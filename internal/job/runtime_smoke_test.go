package job

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const runtimeSmokeModel = "qwen2.5:0.5b"

func requireRuntimeSmoke(t *testing.T) {
	t.Helper()
	if testing.Short() || os.Getenv("RUNGPU_RUNTIME_SMOKE") != "1" {
		t.Skip("set RUNGPU_RUNTIME_SMOKE=1 and omit -short to run real inference")
	}
}

func TestRuntimeSmoke_NativeOllamaInference(t *testing.T) {
	requireRuntimeSmoke(t)

	runtime := newOllamaRuntime(t.TempDir())
	prepareContext, cancelPrepare := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancelPrepare()
	if err := runtime.Prepare(prepareContext, types.JobAssignment{ModelName: runtimeSmokeModel}); err != nil {
		t.Fatalf("prepare native Ollama runtime: %v", err)
	}

	runContext, cancelRun := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelRun()
	result, err := runtime.Run(runContext, types.JobAssignment{
		JobID:     "runtime-smoke-native",
		ModelName: runtimeSmokeModel,
		Input: map[string]interface{}{
			"prompt": "What is 2+2? Answer with only the number.",
		},
		Parameters: map[string]interface{}{
			"temperature": 0.0,
			"num_predict":  8,
		},
	})
	if err != nil {
		t.Fatalf("run native Ollama inference: %v", err)
	}
	response, ok := result["response"].(string)
	if !ok || !strings.Contains(response, "4") {
		t.Fatalf("unexpected native inference response: %q", response)
	}
}

func TestRuntimeSmoke_DockerInference(t *testing.T) {
	requireRuntimeSmoke(t)
	if err := exec.Command("docker", "version").Run(); err != nil {
		t.Fatalf("Docker daemon is required for runtime smoke tests: %v", err)
	}

	const (
		image    = "ghcr.io/ggml-org/llama.cpp:light"
		modelURL = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
	)
	containerName := fmt.Sprintf("rungpu-runtime-smoke-%d", time.Now().UnixNano())
	cleanup := exec.Command("docker", "rm", "--force", containerName)
	t.Cleanup(func() { _ = cleanup.Run() })

	pullContext, cancelPull := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancelPull()
	if output, err := exec.CommandContext(pullContext, "docker", "pull", image).CombinedOutput(); err != nil {
		t.Fatalf("pull llama.cpp image: %v\n%s", err, output)
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("resolve home directory: %v", err)
	}
	modelDir, err := os.MkdirTemp(homeDir, ".rungpu-runtime-smoke-")
	if err != nil {
		t.Fatalf("create shared model directory: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(modelDir) })
	modelPath := filepath.Join(modelDir, "model.gguf")
	downloadContext, cancelDownload := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancelDownload()
	req, err := http.NewRequestWithContext(downloadContext, http.MethodGet, modelURL, nil)
	if err != nil {
		t.Fatalf("create model request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("download public GGUF model: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("download public GGUF model: HTTP %d", resp.StatusCode)
	}
	modelFile, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("create model file: %v", err)
	}
	written, copyErr := io.Copy(modelFile, io.LimitReader(resp.Body, 150<<20))
	closeErr := modelFile.Close()
	if copyErr != nil || closeErr != nil {
		t.Fatalf("save model: copy=%v close=%v", copyErr, closeErr)
	}
	if written < 50<<20 || written >= 150<<20 {
		t.Fatalf("unexpected model size: %d bytes", written)
	}

	inferenceContext, cancelInference := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelInference()
	output, err := exec.CommandContext(inferenceContext, "docker", "run", "--rm",
		"--name", containerName, "--log-driver", "none",
		"--volume", modelPath+":/models/model.gguf:ro", image,
		"--model", "/models/model.gguf",
		"--prompt", "What is 2+2? Answer with only the number.",
		"--predict", "8", "--temp", "0", "--no-display-prompt",
		"--single-turn", "--simple-io").CombinedOutput()
	if err != nil {
		t.Fatalf("run Docker llama.cpp inference: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), "4") {
		t.Fatalf("unexpected Docker inference response: %q", output)
	}
}