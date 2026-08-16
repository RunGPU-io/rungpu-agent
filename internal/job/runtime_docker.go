package job

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const workerImage = "tokenize-worker:latest"

// dockerRuntime runs jobs in the CUDA worker container on NVIDIA hosts. It
// mirrors the worker contract (env JOB_ID/MODEL_PATH/INPUT_DATA, output line
// "OUTPUT:{json}").
type dockerRuntime struct {
	docker     *dockermgr.Manager
	cacheDir   string
	maxCacheGB int
}

func newDockerRuntime(cacheDir string, maxCacheGB int) *dockerRuntime {
	return &dockerRuntime{docker: dockermgr.New(), cacheDir: cacheDir, maxCacheGB: maxCacheGB}
}

func (r *dockerRuntime) Name() string { return "docker-cuda" }

func (r *dockerRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	if err := os.MkdirAll(r.cacheDir, 0o755); err != nil {
		return err
	}
	id := modelID(a.ModelName)

	// Check cache first — skip download if model already exists
	dest := filepath.Join(r.cacheDir, id)
	if info, err := os.Stat(dest); err == nil && info.Size() > 0 {
		return nil // cached
	}

	url := a.ModelURL
	if url == "" {
		// Use HuggingFace API to resolve the actual model download URL
		url = "https://huggingface.co/api/models/" + a.ModelName
	}
	_, err := r.downloadModel(ctx, url, id)
	return err
}

func (r *dockerRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	id := modelID(a.ModelName)
	inputJSON, _ := json.Marshal(a.Input)
	containerName := "tokenize-job-" + a.JobID

	if _, err := r.docker.Run(ctx, dockermgr.RunOptions{
		Image:  workerImage,
		Name:   containerName,
		UseGPU: true,
		Mounts: []string{r.cacheDir + ":/models:ro"},
		Env: map[string]string{
			"JOB_ID":     a.JobID,
			"MODEL_PATH": "/models/" + id,
			"INPUT_DATA": string(inputJSON),
		},
	}); err != nil {
		return nil, err
	}
	// Always clean up the container.
	defer func() {
		_ = r.docker.Stop(context.Background(), containerName)
		_ = r.docker.Remove(context.Background(), containerName)
	}()

	output, exitCode, err := r.waitForCompletion(ctx, containerName, 120*time.Second)
	if err != nil {
		return nil, err
	}

	parsed := parseOutput(output)
	if exitCode != 0 {
		if parsed != nil {
			if msg, ok := parsed["error"].(string); ok && msg != "" {
				return nil, fmt.Errorf("%s", msg)
			}
		}
		return nil, fmt.Errorf("worker exited with code %d", exitCode)
	}
	return parsed, nil
}

func (r *dockerRuntime) Cleanup(force bool) error {
	if !force {
		return nil
	}
	if err := os.RemoveAll(r.cacheDir); err != nil {
		return err
	}
	return os.MkdirAll(r.cacheDir, 0o755)
}

// downloadModel fetches the model to the cache dir if not already present.
func (r *dockerRuntime) downloadModel(ctx context.Context, url, id string) (string, error) {
	dest := filepath.Join(r.cacheDir, id)
	if _, err := os.Stat(dest); err == nil {
		return dest, nil // cached
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status %d downloading %s", resp.StatusCode, url)
	}

	f, err := os.Create(dest)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := io.Copy(f, resp.Body); err != nil {
		return "", err
	}
	return dest, nil
}

// waitForCompletion polls the container until it exits or the timeout elapses,
// then returns its logs and exit code.
func (r *dockerRuntime) waitForCompletion(ctx context.Context, name string, timeout time.Duration) (string, int, error) {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(1 * time.Second)
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
				logs, _ := r.docker.Logs(ctx, name, 1000)
				return logs, exitCode, nil
			}
			if time.Now().After(deadline) {
				return "", 0, fmt.Errorf("job execution timeout after %s", timeout)
			}
		}
	}
}

// parseOutput extracts the JSON object from the worker's "OUTPUT:{...}" line.
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
