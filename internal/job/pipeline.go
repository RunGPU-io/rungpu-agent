// Package job — pipeline.go implements the full job execution pipeline:
//
//  1. Resolve Docker image (from HuggingFace URL, well-known name, or explicit)
//  2. Pull image (cached — skip if already present)
//  3. Download custom files (LoRAs, workflows, checkpoints) into a staging dir
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
	"fmt"
	"io"
	"net/http"
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
	"raw.githubusercontent.com",
	"github.com",
	"storage.googleapis.com",
}

// SafeFileExtensions are the only file types we allow downloading.
var SafeFileExtensions = []string{
	".safetensors", ".ckpt", ".pt", ".pth", ".bin", // model weights
	".json", ".yaml", ".yml", ".toml", // configs, workflows
	".png", ".jpg", ".jpeg", ".webp", // reference images
	".txt", ".csv", // prompts, metadata
}

// ValidateCustomFileURL checks that a download URL is from a trusted source
// and the target path doesn't escape the staging directory.
func ValidateCustomFileURL(url, path string) error {
	lower := strings.ToLower(url)

	// Must be HTTPS (no HTTP, no file://, no ftp://)
	if !strings.HasPrefix(lower, "https://") {
		return fmt.Errorf("custom file URL must use HTTPS: %s", url)
	}

	// Check against trusted hosts
	host := strings.TrimPrefix(lower, "https://")
	if idx := strings.IndexByte(host, '/'); idx > 0 {
		host = host[:idx]
	}
	// Remove port if present
	if idx := strings.IndexByte(host, ':'); idx > 0 {
		host = host[:idx]
	}

	trusted := false
	for _, th := range TrustedFileHosts {
		if host == th || strings.HasSuffix(host, "."+th) {
			trusted = true
			break
		}
	}
	if !trusted {
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
	if !validExt && ext != "" {
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

// DownloadCustomFiles downloads LoRAs, workflows, checkpoints into a staging
// directory. Only downloads from trusted sources with safe file extensions.
func DownloadCustomFiles(ctx context.Context, files []types.CustomFile, stagingDir string, progress ProgressFunc) error {
	return DownloadCustomFilesCached(ctx, files, stagingDir, "", progress)
}

// DownloadCustomFilesCached stores one immutable copy per URL in assetCacheDir
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

		pct := float64(i) / float64(len(files))
		if progress != nil {
			progress("downloading_files", pct, fmt.Sprintf("Downloading %s (%d/%d)", name, i+1, len(files)))
		}

		destPath := filepath.Join(stagingDir, f.Path)
		relative, err := filepath.Rel(stagingDir, destPath)
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || filepath.IsAbs(relative) {
			return fmt.Errorf("security: file path %q escapes staging directory", f.Path)
		}
		if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
			return fmt.Errorf("create dir for %s: %w", f.Path, err)
		}

		// Skip if this job already has a complete staged file.
		if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
			continue
		}

		cachePath := ""
		if assetCacheDir != "" {
			digest := sha256.Sum256([]byte(f.URL))
			cachePath = filepath.Join(assetCacheDir, fmt.Sprintf("%x%s", digest, strings.ToLower(filepath.Ext(f.Path))))
		}
		sourcePath := cachePath
		if sourcePath == "" {
			sourcePath = destPath
		}
		if cachePath != "" {
			if err := ensureCachedFile(ctx, f.URL, cachePath); err != nil {
				return fmt.Errorf("download %s: %w", name, err)
			}
			if err := stageCachedFile(cachePath, destPath); err != nil {
				return fmt.Errorf("stage %s: %w", name, err)
			}
		} else if info, statErr := os.Stat(sourcePath); statErr != nil || info.Size() == 0 {
			if err := downloadFile(ctx, f.URL, sourcePath); err != nil {
				return fmt.Errorf("download %s: %w", name, err)
			}
		}
	}

	if progress != nil {
		progress("downloading_files", 1.0, fmt.Sprintf("Downloaded %d file(s)", len(files)))
	}
	return nil
}

func ensureCachedFile(ctx context.Context, sourceURL, cachePath string) error {
	lockValue, _ := customAssetDownloadLocks.LoadOrStore(cachePath, &sync.Mutex{})
	assetLock := lockValue.(*sync.Mutex)
	assetLock.Lock()
	defer assetLock.Unlock()
	if info, err := os.Stat(cachePath); err != nil || info.Size() == 0 {
		if err := downloadFile(ctx, sourceURL, cachePath); err != nil {
			return err
		}
	}
	now := time.Now()
	return os.Chtimes(cachePath, now, now)
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

	resp, err := http.DefaultClient.Do(req)
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

	if limit > 0 {
		// Read one byte past the limit so we can detect an over-cap stream even
		// when Content-Length was missing or understated.
		n, copyErr := io.Copy(f, io.LimitReader(resp.Body, limit+1))
		if copyErr != nil {
			return copyErr
		}
		if n > limit {
			return fmt.Errorf("download from %s exceeded max allowed %d bytes", url, limit)
		}
	} else if _, err = io.Copy(f, resp.Body); err != nil {
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

	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("open output file: %w", err)
	}
	defer f.Close()

	info, _ := f.Stat()

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, f)
	if err != nil {
		return err
	}
	req.ContentLength = info.Size()

	// Detect content type from extension
	switch strings.ToLower(filepath.Ext(filePath)) {
	case ".mp4":
		req.Header.Set("Content-Type", "video/mp4")
	case ".webm":
		req.Header.Set("Content-Type", "video/webm")
	case ".png":
		req.Header.Set("Content-Type", "image/png")
	case ".jpg", ".jpeg":
		req.Header.Set("Content-Type", "image/jpeg")
	case ".webp":
		req.Header.Set("Content-Type", "image/webp")
	default:
		req.Header.Set("Content-Type", "application/octet-stream")
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("upload failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("upload returned HTTP %d", resp.StatusCode)
	}
	return nil
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
