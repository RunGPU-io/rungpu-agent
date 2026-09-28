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

func resolveJobImage(a types.JobAssignment) (string, error) {
	return ResolveImage(a)
}

func buildContainerEnv(a types.JobAssignment, drive coordinatorDrive) map[string]string {
	env := map[string]string{}
	if a.Parameters != nil {
		for key, value := range a.Parameters {
			if key == "docker_image" || key == "workspace" || isCoordinatorParam(key) {
				continue
			}
			if text, ok := value.(string); ok {
				env[strings.ToUpper(key)] = text
			}
		}
	}
	for key, value := range drive.ExtraEnv {
		env[key] = value
	}

	inputJSON, _ := json.Marshal(a.Input)
	env["JOB_ID"] = a.JobID
	env["MODEL_NAME"] = a.ModelName
	env["INPUT_DATA"] = string(inputJSON)
	return env
}

func (r *customDockerRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	img, err := resolveJobImage(a)
	if err != nil {
		return err
	}

	if err := dockermgr.ValidateImage(img, dockermgr.WithAssignedImage(r.docker.Policy, img)); err != nil {
		return fmt.Errorf("security: %w", err)
	}

	if !r.docker.Available(ctx) {
		return fmt.Errorf("docker is not available — install Docker to run custom model images")
	}

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

	containerName := CustomContainerName(a.JobID)
	if a.ResourceName != "" {
		containerName = a.ResourceName
	}

	drive := parseCoordinatorDrive(a.Parameters)
	env := buildContainerEnv(a, drive)

	stagingDir := filepath.Join(r.cacheDir, "staging", a.JobID)
	mounts := BuildMounts(stagingDir, a.CustomFiles)
	if assignmentNeedsCustomMount(a, drive) && !hasCustomMount(mounts) {
		mounts = append(mounts, stagingDir+":/custom:ro")
	}

	outputDir := filepath.Join(r.cacheDir, "output", a.JobID)
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		return nil, fmt.Errorf("create output directory: %w", err)
	}
	mounts = append(mounts, outputDir+":/output")
	if drive.OutputMount != "" && drive.OutputMount != "/output" {
		mounts = append(mounts, outputDir+":"+drive.OutputMount)
	}
	mounts = applyStageMounts(mounts, stagingDir, drive.StageMounts)

	if _, runErr := r.docker.Run(ctx, dockermgr.RunOptions{
		Image:            img,
		Name:             containerName,
		Labels:           a.ResourceLabels,
		ExpectedDaemonID: a.ResourceDaemonID,
		Network:          "none",
		UseGPU:           r.useGPU,
		GPUDevice:        r.gpuDevice,
		Mounts:           mounts,
		Env:              env,
		ShmSize:          "8g",
		Entrypoint:       drive.Entrypoint,
		Command:          drive.Command,
		UseHostMemory:    drive.UseHostMemory,
	}); runErr != nil {
		return nil, fmt.Errorf("failed to start container %s: %w", img, runErr)
	}

	defer func() {
		if a.ResourceName != "" {
			_ = r.docker.RemoveOwnedAndVerify(context.Background(), dockermgr.OwnedResource{Name: containerName, Labels: a.ResourceLabels, DaemonID: a.ResourceDaemonID})
		} else {
			_ = r.docker.Stop(context.Background(), containerName)
			_ = r.docker.Remove(context.Background(), containerName)
		}
	}()

	if drive.Script != "" {
		if err := r.execDriveScript(ctx, containerName, outputDir, drive.Script, r.jobTimeoutFor(a)); err != nil {
			return nil, err
		}
		return r.collectDriveOutput(img, outputDir)
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

	if outputFile := findOutputFile(outputDir); outputFile != "" {
		parsed["output_file"] = outputFile
	}

	return parsed, nil
}

func (r *customDockerRuntime) Cleanup(force bool) error { return nil }

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

	if len(entries) > 0 && !entries[0].IsDir() {
		return filepath.Join(dir, entries[0].Name())
	}
	return ""
}

func parseOutput(logs string) map[string]interface{} {
	idx := strings.LastIndex(logs, "OUTPUT:")
	if idx < 0 {
		return nil
	}
	rest := logs[idx+len("OUTPUT:"):]
	if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
		rest = rest[:nl]
	}
	var out map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(rest)), &out); err != nil {
		return nil
	}
	return out
}
