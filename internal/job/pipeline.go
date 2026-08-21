// Package job — pipeline.go implements the full job execution pipeline:
//
//  1. Resolve Docker image (from HuggingFace URL, well-known name, or explicit)
//  2. Pull image (cached — skip if already present)
//  3. Download verified custom files (safe model assets and workflows) into staging
//  4. Start container with GPU, mounts, env vars
//  5. Wait for completion (or keep running for workspace mode)
//  6. Collect output files from container
//  7. Upload outputs to storage (Supabase Storage / S3 pre-signed URL)
//  8. Report result with output URLs back to the coordinator
//
// Progress is reported at each stage via the WebSocket so the Next.js frontend
// can show a live progress bar.
package job

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

// Deterministic container names so a job's container can be found and torn down
// by job id alone (on cancel or shutdown), even from a different goroutine.
func CustomContainerName(jobID string) string    { return "tokenize-custom-" + jobID }
func WorkspaceContainerName(jobID string) string { return "tokenize-ws-" + jobID }

// maxDownloadBytes caps the size of a single custom-file download (0 = unlimited).
// Set from config via SetMaxDownloadBytes so a job can't fill the host disk.
var maxDownloadBytes int64
var customAssetCacheMu sync.RWMutex
var customAssetDownloadLocks sync.Map

// SetMaxDownloadBytes sets the per-file download cap in bytes (0 = unlimited).
func SetMaxDownloadBytes(n int64) { atomic.StoreInt64(&maxDownloadBytes, n) }

// configuredHFToken is the HuggingFace token from config, used as a fallback
// when the HF_TOKEN env var is not set.
var configuredHFToken atomic.Value // string

// SetHFToken records the HuggingFace token from config.
func SetHFToken(t string) { configuredHFToken.Store(t) }

// hfToken returns the effective HuggingFace token: env var takes precedence,
// then the configured token.
func hfToken() string {
	if t := os.Getenv("HF_TOKEN"); t != "" {
		return t
	}
	if v, ok := configuredHFToken.Load().(string); ok {
		return v
	}
	return ""
}

// ProgressFunc is called at each pipeline stage so the caller can relay
// progress to the WebSocket (and thus to the user's browser).
type ProgressFunc func(stage string, progress float64, message string)

var cachedFileChtimes = os.Chtimes

func StageInlineWorkflow(workflow, expectedSHA256, stagingDir string) error {
	data := []byte(workflow)
	if len(data) == 0 || len(data) > 256*1024 || !json.Valid(data) {
		return fmt.Errorf("workflow must be valid JSON no larger than 256 KiB")
	}
	actual := sha256.Sum256(data)
	if hex.EncodeToString(actual[:]) != expectedSHA256 {
		return fmt.Errorf("workflow SHA-256 mismatch")
	}
	workflowDir := filepath.Join(stagingDir, "workflows")
	if err := os.MkdirAll(workflowDir, 0o700); err != nil {
		return fmt.Errorf("create workflow directory: %w", err)
	}
	if err := os.WriteFile(filepath.Join(workflowDir, "workflow.json"), data, 0o600); err != nil {
		return fmt.Errorf("write workflow: %w", err)
	}
	return nil
}

// ResolveImage determines the explicit Docker batch worker to use for a job.
func ResolveImage(a types.JobAssignment) (string, error) {
	// 1. Explicit DockerImage field
	if a.DockerImage != "" {
		return a.DockerImage, nil
	}

	// 2. Parameters["docker_image"]
	if a.Parameters != nil {
		if img, ok := a.Parameters["docker_image"].(string); ok && img != "" {
			return img, nil
		}
	}

	return "", fmt.Errorf("cannot resolve a batch worker for model %q — provide a trusted docker_image that consumes INPUT_DATA, writes /output, and exits", a.ModelName)
}

// TrustedFileHosts are the only domains we allow custom file downloads from.
// Only curated platforms with community moderation and content review.
var TrustedFileHosts = []string{
	"huggingface.co",
	"hf.co",
	"cdn-lfs.huggingface.co",
	"civitai.com",
	"civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com",
	"raw.githubusercontent.com",
	"github.com",
	"storage.googleapis.com",
}

var TrustedUploadHosts = []string{"storage.googleapis.com"}

var allowInsecureLoopbackForTests bool

// SafeFileExtensions are the only file types we allow downloading.
var SafeFileExtensions = []string{
	".safetensors",                    // model weights without executable pickle payloads
	".json", ".yaml", ".yml", ".toml", // configs, workflows
	".png", ".jpg", ".jpeg", ".webp", // reference images
	".wav", ".mp3",                   // reference audio
	".txt", ".csv", // prompts, metadata
}

// ValidateCustomFileURL checks that a download URL is from a trusted source
// and the target path doesn't escape the staging directory.
func ValidateCustomFileURL(url, path string) error {
	if err := validateHTTPSHost(url, TrustedFileHosts); err != nil {
		return fmt.Errorf("custom file URL %q is not from a trusted source. Allowed: %s",
			url, strings.Join(TrustedFileHosts, ", "))
	}

	// Check file extension
	ext := strings.ToLower(filepath.Ext(path))
	if ext == "" {
		ext = strings.ToLower(filepath.Ext(url))
	}
	validExt := false
	for _, safe := range SafeFileExtensions {
		if ext == safe {
			validExt = true
			break
		}
	}
	if !validExt {
		return fmt.Errorf("file extension %q is not allowed. Allowed: %s",
			ext, strings.Join(SafeFileExtensions, ", "))
	}

	// Prevent path traversal (e.g. "../../etc/passwd")
	cleaned := filepath.Clean(path)
	if filepath.IsAbs(path) || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return fmt.Errorf("file path %q contains path traversal", path)
	}

	return nil
}

func validateHTTPSHost(rawURL string, trustedHosts []string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Hostname() == "" {
		return fmt.Errorf("URL must use HTTPS on the default port")
	}
	loopbackTestURL := allowInsecureLoopbackForTests && parsed.Scheme == "http" &&
		(parsed.Hostname() == "127.0.0.1" || parsed.Hostname() == "::1")
	if !loopbackTestURL && (parsed.Scheme != "https" || (parsed.Port() != "" && parsed.Port() != "443")) {
		return fmt.Errorf("URL must use HTTPS on the default port")
	}
	host := strings.ToLower(parsed.Hostname())
	for _, trustedHost := range trustedHosts {
		if host == trustedHost || strings.HasSuffix(host, "."+trustedHost) {
			return nil
		}
	}
	return fmt.Errorf("untrusted URL host")
}

func trustedDownloadClient() *http.Client {
	return &http.Client{CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return fmt.Errorf("too many redirects")
		}
		return validateHTTPSHost(req.URL.String(), TrustedFileHosts)
	}}
}

// DownloadCustomFiles downloads safe model assets and workflows into a staging
// directory. Only downloads from trusted sources with safe file extensions.
func DownloadCustomFiles(ctx context.Context, files []types.CustomFile, stagingDir string, progress ProgressFunc) error {
	return DownloadCustomFilesCached(ctx, files, stagingDir, "", progress)
}

// DownloadCustomFilesCached stores one immutable copy per digest in assetCacheDir
// and stages links/copies for each job. An empty cache directory preserves the
// legacy per-job behavior used by callers that do not own a persistent cache.
func DownloadCustomFilesCached(ctx context.Context, files []types.CustomFile, stagingDir, assetCacheDir string, progress ProgressFunc) error {
	if len(files) == 0 {
		return nil
	}

	// Validate ALL files before downloading any
	for _, f := range files {
		if err := ValidateCustomFileURL(f.URL, f.Path); err != nil {
			return fmt.Errorf("security: %w", err)
		}
		if !validSHA256(f.SHA256) {
			return fmt.Errorf("security: custom file %q requires a lowercase SHA-256", f.Path)
		}
	}

	if err := os.MkdirAll(stagingDir, 0o755); err != nil {
		return fmt.Errorf("create staging dir: %w", err)
	}
	if assetCacheDir != "" {
		customAssetCacheMu.RLock()
		defer customAssetCacheMu.RUnlock()
		if err := os.MkdirAll(assetCacheDir, 0o755); err != nil {
			return fmt.Errorf("create asset cache: %w", err)
		}
	}

	for i, f := range files {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		name := f.Name
		if name == "" {
			name = filepath.Base(f.Path)
		}

		fileBase := float64(i) / float64(len(files))
		fileSpan := 1 / float64(len(files))
		if progress != nil {
			progress("downloading_files", fileBase, fmt.Sprintf("Downloading %s (%d/%d)", name, i+1, len(files)))
		}
		downloadProgress := func(downloaded, total int64) {
			if progress == nil || total <= 0 {
				return
			}
			fraction := math.Min(1, float64(downloaded)/float64(total))
			progress("downloading_files", fileBase+fileSpan*fraction,
				fmt.Sprintf("Downloading %s (%s / %s)", name, formatDownloadBytes(downloaded), formatDownloadBytes(total)))
		}

		destPath := filepath.Join(stagingDir, f.Path)
		relative, err := filepath.Rel(stagingDir, destPath)
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || filepath.IsAbs(relative) {
			return fmt.Errorf("security: file path %q escapes staging directory", f.Path)
		}
		if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
			return fmt.Errorf("create dir for %s: %w", f.Path, err)
		}

		// Skip if this job already has the exact staged content.
		if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
			if verifyFileSHA256(destPath, f.SHA256) == nil {
				continue
			}
			_ = os.Remove(destPath)
		}

		cachePath := ""
		if assetCacheDir != "" {
			cachePath = filepath.Join(assetCacheDir, f.SHA256+strings.ToLower(filepath.Ext(f.Path)))
		}
		sourcePath := cachePath
		if sourcePath == "" {
			sourcePath = destPath
		}
		if cachePath != "" {
			if err := ensureCachedFileWithProgress(ctx, f.URL, cachePath, f.SHA256, downloadProgress); err != nil {
				return fmt.Errorf("download %s: %w", name, err)
			}
			if err := stageCachedFile(cachePath, destPath); err != nil {
				return fmt.Errorf("stage %s: %w", name, err)
			}
			if err := verifyFileSHA256(destPath, f.SHA256); err != nil {
				_ = os.Remove(destPath)
				return fmt.Errorf("verify %s: %w", name, err)
			}
		} else if info, statErr := os.Stat(sourcePath); statErr != nil || info.Size() == 0 {
			if err := downloadFileWithProgress(ctx, f.URL, sourcePath, downloadProgress); err != nil {
				return fmt.Errorf("download %s: %w", name, err)
			}
			if err := verifyFileSHA256(sourcePath, f.SHA256); err != nil {
				_ = os.Remove(sourcePath)
				return fmt.Errorf("verify %s: %w", name, err)
			}
		}
	}

	if progress != nil {
		progress("downloading_files", 1.0, fmt.Sprintf("Downloaded %d file(s)", len(files)))
	}
	return nil
}

func ensureCachedFile(ctx context.Context, sourceURL, cachePath, expectedSHA256 string) error {
	return ensureCachedFileWithProgress(ctx, sourceURL, cachePath, expectedSHA256, nil)
}

func ensureCachedFileWithProgress(ctx context.Context, sourceURL, cachePath, expectedSHA256 string, progress func(int64, int64)) error {
	lockValue, _ := customAssetDownloadLocks.LoadOrStore(cachePath, &sync.Mutex{})
	assetLock := lockValue.(*sync.Mutex)
	assetLock.Lock()
	defer assetLock.Unlock()
	if info, err := os.Stat(cachePath); err != nil || info.Size() == 0 || verifyFileSHA256(cachePath, expectedSHA256) != nil {
		_ = os.Remove(cachePath)
		if err := downloadFileWithProgress(ctx, sourceURL, cachePath, progress); err != nil {
			return err
		}
	}
	if err := verifyFileSHA256(cachePath, expectedSHA256); err != nil {
		_ = os.Remove(cachePath)
		return err
	}
	now := time.Now()
	_ = cachedFileChtimes(cachePath, now, now)
	return nil
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && hex.EncodeToString(decoded) == value
}

func verifyFileSHA256(path, expected string) error {
	if !validSHA256(expected) {
		return fmt.Errorf("invalid expected SHA-256")
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return err
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if actual != expected {
		return fmt.Errorf("SHA-256 mismatch: expected %s, got %s", expected, actual)
	}
	return nil
}

// PruneCustomAssets removes expired files, then evicts least-recently-used
// files until the cache is within maxBytes. Modification time is refreshed on
// every cache hit, so the policy survives process restarts without an index.
func PruneCustomAssets(cacheDir string, ttl time.Duration, maxBytes int64, now time.Time) (files int, bytes int64, err error) {
	if ttl <= 0 && maxBytes <= 0 {
		return 0, 0, nil
	}
	customAssetCacheMu.Lock()
	defer customAssetCacheMu.Unlock()
	cutoff := now.Add(-ttl)

	type candidate struct {
		path     string
		size     int64
		lastUsed time.Time
	}
	candidates := []candidate{}
	var totalBytes int64
	assetsDir := filepath.Join(cacheDir, "assets")
	err = filepath.Walk(assetsDir, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			if os.IsNotExist(walkErr) {
				return nil
			}
			return walkErr
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		candidates = append(candidates, candidate{path: path, size: info.Size(), lastUsed: info.ModTime()})
		totalBytes += info.Size()
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return files, bytes, err
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].lastUsed.Before(candidates[j].lastUsed) })
	for _, item := range candidates {
		expired := ttl > 0 && item.lastUsed.Before(cutoff)
		overBudget := maxBytes > 0 && totalBytes > maxBytes
		if !expired && !overBudget {
			continue
		}
		if removeErr := os.Remove(item.path); removeErr != nil && !os.IsNotExist(removeErr) {
			return files, bytes, removeErr
		}
		files++
		bytes += item.size
		totalBytes -= item.size
	}
	return files, bytes, nil
}

// PruneBatchStaging removes completed batch-job staging while preserving
// staging used by detached workspace containers.
func PruneBatchStaging(cacheDir string, workspaceIDs map[string]bool) error {
	customAssetCacheMu.Lock()
	defer customAssetCacheMu.Unlock()
	stagingDir := filepath.Join(cacheDir, "staging")
	entries, err := os.ReadDir(stagingDir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if !entry.IsDir() || workspaceIDs[entry.Name()] {
			continue
		}
		if err := os.RemoveAll(filepath.Join(stagingDir, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}

func stageCachedFile(source, destination string) error {
	_ = os.Remove(destination)
	if err := os.Link(source, destination); err == nil {
		return nil
	}
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(destination)
		return copyErr
	}
	return closeErr
}

func downloadFile(ctx context.Context, url, dest string) error {
	return downloadFileWithProgress(ctx, url, dest, nil)
}

type downloadProgressWriter struct {
	written     int64
	total       int64
	lastPercent int64
	progress    func(int64, int64)
}

func (w *downloadProgressWriter) Write(chunk []byte) (int, error) {
	w.written += int64(len(chunk))
	if w.progress != nil && w.total > 0 {
		percent := w.written * 100 / w.total
		if percent >= w.lastPercent+1 || w.written >= w.total {
			w.lastPercent = percent
			w.progress(w.written, w.total)
		}
	}
	return len(chunk), nil
}

func formatDownloadBytes(bytes int64) string {
	const mebibyte = 1024 * 1024
	if bytes >= mebibyte {
		return fmt.Sprintf("%.1f GiB", float64(bytes)/(1024*mebibyte))
	}
	return fmt.Sprintf("%.1f MiB", float64(bytes)/mebibyte)
}

func downloadFileWithProgress(ctx context.Context, url, dest string, progress func(int64, int64)) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	// Add HuggingFace auth token if downloading from HF (gated models need this)
	if strings.Contains(url, "huggingface.co") || strings.Contains(url, "hf.co") {
		if token := hfToken(); token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
	}

	resp, err := trustedDownloadClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("HTTP %d from %s — this model may be gated. Set HF_TOKEN env var with your HuggingFace access token (https://huggingface.co/settings/tokens)", resp.StatusCode, url)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}

	limit := atomic.LoadInt64(&maxDownloadBytes)
	if limit > 0 && resp.ContentLength > limit {
		return fmt.Errorf("file %s is %d bytes, exceeds max allowed %d bytes", url, resp.ContentLength, limit)
	}

	f, err := os.CreateTemp(filepath.Dir(dest), "."+filepath.Base(dest)+".*.part")
	if err != nil {
		return err
	}
	temporary := f.Name()
	complete := false
	defer func() {
		_ = f.Close()
		if !complete {
			_ = os.Remove(temporary)
		}
	}()

	progressWriter := &downloadProgressWriter{total: resp.ContentLength, lastPercent: -1, progress: progress}
	writer := io.MultiWriter(f, progressWriter)
	if limit > 0 {
		// Read one byte past the limit so we can detect an over-cap stream even
		// when Content-Length was missing or understated.
		n, copyErr := io.Copy(writer, io.LimitReader(resp.Body, limit+1))
		if copyErr != nil {
			return copyErr
		}
		if n > limit {
			return fmt.Errorf("download from %s exceeded max allowed %d bytes", url, limit)
		}
	} else if _, err = io.Copy(writer, resp.Body); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporary, dest); err != nil {
		return err
	}
	complete = true
	return nil
}

// UploadOutput uploads a file to the given pre-signed URL (PUT).
// Returns the public URL of the uploaded file.
func UploadOutput(ctx context.Context, filePath, uploadURL string) error {
	if uploadURL == "" {
		return nil
	}

	if err := validateHTTPSHost(uploadURL, TrustedUploadHosts); err != nil {
		return fmt.Errorf("upload URL rejected: %w", err)
	}
	info, err := os.Lstat(filePath)
	if err != nil || !info.Mode().IsRegular() {
		return fmt.Errorf("output must be a regular file")
	}
	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("open output file: %w", err)
	}
	defer f.Close()

	openedInfo, err := f.Stat()
	if err != nil || !openedInfo.Mode().IsRegular() || !os.SameFile(info, openedInfo) {
		return fmt.Errorf("output file changed before upload")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, f)
	if err != nil {
		return err
	}
	req.ContentLength = openedInfo.Size()

	req.Header.Set("Content-Type", outputContentType(filePath))

	client := &http.Client{CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("upload failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("upload returned HTTP %d", resp.StatusCode)
	}
	return nil
}

func outputContentType(filePath string) string {
	switch strings.ToLower(filepath.Ext(filePath)) {
	case ".mp4":
		return "video/mp4"
	case ".webm":
		return "video/webm"
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	default:
		return "application/octet-stream"
	}
}

// BuildMounts creates the Docker volume mount arguments from the staging dir.
// Maps custom file paths into the container at the expected locations.
func BuildMounts(cacheDir, stagingDir string, files []types.CustomFile) []string {
	mounts := []string{
		cacheDir + ":/cache",
	}

	if len(files) > 0 && stagingDir != "" {
		// Mount the entire staging dir so all custom files are accessible
		mounts = append(mounts, stagingDir+":/custom:ro")
	}

	return mounts
}
