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
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/RunGPU-io/gpu-agent/internal/types"
)

// Deterministic container names so a job's container can be found and torn down
// by job id alone (on cancel or shutdown), even from a different goroutine.
func CustomContainerName(jobID string) string    { return "tokenize-custom-" + jobID }
func WorkspaceContainerName(jobID string) string { return "tokenize-ws-" + jobID }

// maxDownloadBytes caps the size of a single custom-file download (0 = unlimited).
// Set from config via SetMaxDownloadBytes so a job can't fill the host disk.
var maxDownloadBytes int64

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

// ResolveImage determines the Docker image to use for a job.
// Priority: job.DockerImage > job.Parameters["docker_image"] > well-known mapping > HuggingFace URL
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

	// 3. Well-known model → image mapping
	img := wellKnownImage(a.ModelName)
	if img != "" {
		return img, nil
	}

	// 4. HuggingFace model URL → HF Docker registry
	if a.ModelURL != "" && strings.Contains(a.ModelURL, "huggingface") {
		// HuggingFace Spaces Docker images: registry.hf.space/org-model
		return hfURLToImage(a.ModelURL), nil
	}

	// 5. Try treating ModelName as a HuggingFace repo → Docker image
	if strings.Contains(a.ModelName, "/") {
		return "registry.hf.space/" + strings.ReplaceAll(a.ModelName, "/", "-"), nil
	}

	return "", fmt.Errorf("cannot resolve Docker image for model %q — provide docker_image or a HuggingFace URL", a.ModelName)
}

// wellKnownImage maps model names to Docker images.
// All image/video generation models route through ComfyUI as the universal runtime.
// Text LLMs route through Ollama (handled by runtime selection, not here).
func wellKnownImage(name string) string {
	// ComfyUI is the universal runtime for all image/video generation.
	// Different models are loaded via ComfyUI workflows, not separate Docker images.
	comfyUI := "ghcr.io/ai-dock/comfyui:latest"

	m := map[string]string{
		// Video generation (via ComfyUI + model-specific nodes)
		"ltx-video":        comfyUI,
		"ltx2-video":       comfyUI,
		"ltx2":             comfyUI,
		"wan2":             comfyUI,
		"wan2.1":           comfyUI,
		"wan2-video":       comfyUI,

		// Image generation (via ComfyUI)
		"stable-diffusion": comfyUI,
		"sdxl":             comfyUI,
		"sd":               comfyUI,
		"flux":             comfyUI,
		"flux.1":           comfyUI,

		// ComfyUI directly
		"comfyui":          comfyUI,

		// Other workspaces (not ComfyUI)
		"a1111":            "ghcr.io/ai-dock/stable-diffusion-webui:latest",
		"invokeai":         "ghcr.io/invoke-ai/invokeai:latest",
		"jupyter":          "jupyter/pytorch-notebook:latest",
	}
	return m[strings.ToLower(name)]
}

func hfURLToImage(url string) string {
	// https://huggingface.co/spaces/org/model → registry.hf.space/org-model
	url = strings.TrimPrefix(url, "https://")
	url = strings.TrimPrefix(url, "http://")
	url = strings.TrimPrefix(url, "huggingface.co/spaces/")
	url = strings.TrimPrefix(url, "huggingface.co/")
	return "registry.hf.space/" + strings.ReplaceAll(url, "/", "-")
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
	".safetensors", ".ckpt", ".pt", ".pth", ".bin",  // model weights
	".json", ".yaml", ".yml", ".toml",                 // configs, workflows
	".png", ".jpg", ".jpeg", ".webp",                  // reference images
	".txt", ".csv",                                     // prompts, metadata
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
	if strings.Contains(cleaned, "..") {
		return fmt.Errorf("file path %q contains path traversal", path)
	}

	return nil
}

// DownloadCustomFiles downloads LoRAs, workflows, checkpoints into a staging
// directory. Only downloads from trusted sources with safe file extensions.
func DownloadCustomFiles(ctx context.Context, files []types.CustomFile, stagingDir string, progress ProgressFunc) error {
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
		if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
			return fmt.Errorf("create dir for %s: %w", f.Path, err)
		}

		// Skip if already downloaded (cache by path)
		if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
			continue
		}

		if err := downloadFile(ctx, f.URL, destPath); err != nil {
			return fmt.Errorf("download %s: %w", name, err)
		}
	}

	if progress != nil {
		progress("downloading_files", 1.0, fmt.Sprintf("Downloaded %d file(s)", len(files)))
	}
	return nil
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

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	if limit > 0 {
		// Read one byte past the limit so we can detect an over-cap stream even
		// when Content-Length was missing or understated.
		n, copyErr := io.Copy(f, io.LimitReader(resp.Body, limit+1))
		if copyErr != nil {
			return copyErr
		}
		if n > limit {
			_ = os.Remove(dest)
			return fmt.Errorf("download from %s exceeded max allowed %d bytes", url, limit)
		}
		return nil
	}

	_, err = io.Copy(f, resp.Body)
	return err
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
