package job

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

// customDockerRuntime runs jobs in a user-specified Docker image. Handles:
//   - HuggingFace Docker image URLs
//   - Any custom Docker image the user provides
//   - Custom files (LoRAs, workflows) mounted into the container
type customDockerRuntime struct {
	docker     *dockermgr.Manager
	cacheDir   string
	useGPU     bool
	gpuDevice  string
	jobTimeout time.Duration
}

func newCustomDockerRuntime(cacheDir string, useGPU bool, gpuDevice string, jobTimeout time.Duration, policy dockermgr.SecurityPolicy) *customDockerRuntime {
	if jobTimeout <= 0 {
		jobTimeout = 60 * time.Minute
	}
	return &customDockerRuntime{
		docker:     dockermgr.NewWithPolicy(policy),
		cacheDir:   cacheDir,
		useGPU:     useGPU,
		gpuDevice:  gpuDevice,
		jobTimeout: jobTimeout,
	}
}

func (r *customDockerRuntime) Name() string { return "docker-custom" }

// resolveJobImage determines the Docker image for this job using the full
// resolution chain: DockerImage field, then the explicit parameters field.
func resolveJobImage(a types.JobAssignment) (string, error) {
	return ResolveImage(a)
}

func (r *customDockerRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	img, err := resolveJobImage(a)
	if err != nil {
		return err
	}

	// Enforce the image allowlist BEFORE pulling — otherwise a job could make
	// the host pull an arbitrary (large/untrusted) image before Run rejects it.
	if err := dockermgr.ValidateImage(img, r.docker.Policy); err != nil {
		return fmt.Errorf("security: %w", err)
	}

	if !r.docker.Available(ctx) {
		return fmt.Errorf("docker is not available — install Docker to run custom model images")
	}

	// Pull the image if not already present (cached)
	if !r.imageExists(ctx, img) {
		cmd := exec.CommandContext(ctx, "docker", "pull", img)
		if out, pullErr := cmd.CombinedOutput(); pullErr != nil {
			return fmt.Errorf("docker pull %s failed: %v: %s", img, pullErr, strings.TrimSpace(string(out)))
		}
		if err := trackManagedAsset(r.cacheDir, "docker", img); err != nil {
			return fmt.Errorf("track pulled Docker image %s: %w", img, err)
		}
	}

	return nil
}

func (r *customDockerRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	img, err := resolveJobImage(a)
	if err != nil {
		return nil, err
	}

	inputJSON, _ := json.Marshal(a.Input)
	containerName := CustomContainerName(a.JobID)

	env := map[string]string{
		"JOB_ID":     a.JobID,
		"MODEL_NAME": a.ModelName,
		"INPUT_DATA": string(inputJSON),
	}
	if a.Parameters != nil {
		for k, v := range a.Parameters {
			if k == "docker_image" || k == "workspace" {
				continue
			}
			if s, ok := v.(string); ok {
				env[strings.ToUpper(k)] = s
			}
		}
	}

	// Build mounts: cache dir + custom files staging dir (if any)
	stagingDir := filepath.Join(r.cacheDir, "staging", a.JobID)
	mounts := BuildMounts(r.cacheDir, stagingDir, a.CustomFiles)

	// Output dir — container writes results here, agent reads them after
	outputDir := filepath.Join(r.cacheDir, "output", a.JobID)
	mounts = append(mounts, outputDir+":/output")

	// Persistent Docker named volume for the shared model cache.
	// Models downloaded by any job are available to all subsequent jobs
	// without re-downloading. This is the key optimization — a 10 GB
	// checkpoint is downloaded once and reused forever.
	volumes := []string{
		"tokenize-model-cache:/models",
	}

	if _, runErr := r.docker.Run(ctx, dockermgr.RunOptions{
		Image:     img,
		Name:      containerName,
		Network:   "none",
		UseGPU:    r.useGPU,
		GPUDevice: r.gpuDevice,
		Mounts:    mounts,
		Volumes:   volumes,
		Env:       env,
		ShmSize:   "8g",
	}); runErr != nil {
		return nil, fmt.Errorf("failed to start container %s: %w", img, runErr)
	}

	defer func() {
		_ = r.docker.Stop(context.Background(), containerName)
		_ = r.docker.Remove(context.Background(), containerName)
	}()

	if isKokoroFastAPIImage(img) || isChatterboxLitServeImage(img) {
		var speechErr error
		if isKokoroFastAPIImage(img) {
			speechErr = r.driveKokoroFastAPI(ctx, containerName, outputDir, r.jobTimeoutFor(a))
		} else {
			speechErr = r.driveChatterboxLitServe(ctx, containerName, outputDir, r.jobTimeoutFor(a))
		}
		if speechErr != nil {
			return nil, speechErr
		}
		return r.collectSpeechOutput(img, outputDir)
	}

	output, exitCode, waitErr := r.waitForCompletion(ctx, containerName, r.jobTimeoutFor(a))
	if waitErr != nil {
		return nil, waitErr
	}

	parsed := parseOutput(output)
	if exitCode != 0 {
		if parsed != nil {
			if msg, ok := parsed["error"].(string); ok && msg != "" {
				return nil, fmt.Errorf("%s", msg)
			}
		}
		lines := strings.Split(strings.TrimSpace(output), "\n")
		tail := output
		if len(lines) > 5 {
			tail = strings.Join(lines[len(lines)-5:], "\n")
		}
		return nil, fmt.Errorf("container exited with code %d:\n%s", exitCode, tail)
	}

	if parsed == nil {
		parsed = map[string]interface{}{
			"status": "completed",
			"output": output,
		}
	}
	parsed["backend"] = "docker-custom"
	parsed["image"] = img

	// Check for output files in the output dir
	if outputFile := findOutputFile(outputDir); outputFile != "" {
		parsed["output_file"] = outputFile
	}

	return parsed, nil
}

func (r *customDockerRuntime) Cleanup(force bool) error { return nil }

// jobTimeoutFor permits a renter to request a shorter timeout, never one above
// the host-configured maximum.
func (r *customDockerRuntime) jobTimeoutFor(a types.JobAssignment) time.Duration {
	requested := time.Duration(0)
	if a.Parameters != nil {
		switch v := a.Parameters["timeout_minutes"].(type) {
		case float64:
			if v > 0 {
				requested = time.Duration(v) * time.Minute
			}
		case int:
			if v > 0 {
				requested = time.Duration(v) * time.Minute
			}
		case string:
			if mins, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && mins > 0 {
				requested = time.Duration(mins) * time.Minute
			}
		}
	}
	if requested > 0 && requested < r.jobTimeout {
		return requested
	}
	return r.jobTimeout
}

func (r *customDockerRuntime) imageExists(ctx context.Context, img string) bool {
	out, err := exec.CommandContext(ctx, "docker", "image", "inspect", img).CombinedOutput()
	return err == nil && len(out) > 2
}

func (r *customDockerRuntime) waitForCompletion(ctx context.Context, name string, timeout time.Duration) (string, int, error) {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return "", 0, ctx.Err()
		case <-ticker.C:
			running, exitCode, err := r.docker.Inspect(ctx, name)
			if err != nil {
				return "", 0, fmt.Errorf("inspect failed: %w", err)
			}
			if !running {
				logs, _ := r.docker.Logs(ctx, name, 2000)
				return logs, exitCode, nil
			}
			if time.Now().After(deadline) {
				return "", 0, fmt.Errorf("job execution timeout after %s", timeout)
			}
		}
	}
}

// findOutputFile looks for the first media file in the output directory.
func findOutputFile(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	mediaExts := map[string]bool{
		".mp4": true, ".webm": true, ".avi": true, ".mov": true,
		".png": true, ".jpg": true, ".jpeg": true, ".webp": true, ".gif": true,
		".wav": true, ".mp3": true, ".flac": true,
	}
	for _, e := range entries {
		if !e.IsDir() {
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if mediaExts[ext] {
				return filepath.Join(dir, e.Name())
			}
		}
	}
	// Fallback: return first file
	if len(entries) > 0 && !entries[0].IsDir() {
		return filepath.Join(dir, entries[0].Name())
	}
	return ""
}
