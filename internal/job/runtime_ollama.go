package job

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	rt "runtime"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const defaultOllamaEndpoint = "http://localhost:11434"
const managedWarmTTL = "3m"

type ollamaRuntime struct {
	cacheDir string
	endpoint string
	backend  string
	client   *http.Client
}

func newOllamaRuntime(cacheDir, hostBackend string) *ollamaRuntime {
	backend := strings.ToLower(strings.TrimSpace(hostBackend))
	switch backend {
	case "cuda", "metal", "cpu":
	default:
		backend = "cpu"
		if rt.GOOS == "darwin" && rt.GOARCH == "arm64" {
			backend = "metal"
		} else if _, err := exec.LookPath("nvidia-smi"); err == nil {
			backend = "cuda"
		}
	}
	return &ollamaRuntime{
		cacheDir: cacheDir,
		endpoint: defaultOllamaEndpoint,
		backend:  backend,
		client:   &http.Client{Timeout: 5 * time.Minute},
	}
}

func (r *ollamaRuntime) Name() string { return "ollama" }

func ollamaModel(a types.JobAssignment) string {
	if a.Parameters != nil {
		if v, ok := a.Parameters["ollama_model"].(string); ok && v != "" {
			return v
		}
	}
	return a.ModelName
}

func extractPrompt(a types.JobAssignment) string {
	if a.Input != nil {
		if p, ok := a.Input["prompt"].(string); ok && p != "" {
			return p
		}
		if p, ok := a.Input["text"].(string); ok && p != "" {
			return p
		}
		if msgs, ok := a.Input["messages"]; ok {
			if b, err := json.Marshal(msgs); err == nil {
				return string(b)
			}
		}
	}
	b, _ := json.Marshal(a.Input)
	return string(b)
}

func (r *ollamaRuntime) isServerRunning() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.endpoint, nil)
	if err != nil {
		return false
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

func (r *ollamaRuntime) ensureServerRunning(ctx context.Context) error {

	if r.isServerRunning() {
		return nil
	}

	if r.endpoint != defaultOllamaEndpoint {
		return fmt.Errorf("ollama server not responding at %s", r.endpoint)
	}

	fmt.Println("[ollama] Server not running — starting it automatically...")

	cmd := exec.Command("ollama", "serve")
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Env = os.Environ()

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ollama server: %w\n"+
			"  Start it manually with: ollama serve", err)
	}

	go func() {
		cmd.Wait()
	}()

	fmt.Println("[ollama] Waiting for server to be ready...")
	deadline := time.Now().Add(30 * time.Second)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("ollama server did not start within 30 seconds\n" +
				"  Try starting it manually: ollama serve")
		}
		if r.isServerRunning() {
			fmt.Println("[ollama] Server is ready.")
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func (r *ollamaRuntime) isModelCached(ctx context.Context, model string) bool {
	body, _ := json.Marshal(map[string]string{"name": model})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint+"/api/show", bytes.NewReader(body))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func installOllama(ctx context.Context) error {
	_ = ctx
	if _, err := exec.LookPath("ollama"); err == nil {
		return nil
	}
	return fmt.Errorf("ollama is not installed; run rungpu-agent setup")
}

func (r *ollamaRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	model := ollamaModel(a)

	if _, err := exec.LookPath("ollama"); err != nil {
		return fmt.Errorf("ollama is not installed; run rungpu-agent setup")
	}

	if err := r.ensureServerRunning(ctx); err != nil {
		return err
	}

	if r.isModelCached(ctx, model) {
		return nil
	}

	fmt.Printf("[ollama] Pulling model %s (this may take a few minutes on first run)...\n", model)
	cmd := exec.CommandContext(ctx, "ollama", "pull", model)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ollama pull %s failed: %w", model, err)
	}
	if err := trackManagedAsset(r.cacheDir, "ollama", model); err != nil {
		return fmt.Errorf("track pulled Ollama model %s: %w", model, err)
	}
	return nil
}

func (r *ollamaRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {

	if err := r.ensureServerRunning(ctx); err != nil {
		return nil, err
	}

	model := ollamaModel(a)
	prompt := extractPrompt(a)

	reqBody, _ := json.Marshal(map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"stream":     false,
		"keep_alive": managedWarmTTL,
		"options":    a.Parameters,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint+"/api/generate", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w\n"+
			"  Is the ollama server running? Start with: ollama serve", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	var gen struct {
		Response  string `json:"response"`
		Model     string `json:"model"`
		EvalCount int    `json:"eval_count"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&gen); err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"status":     "completed",
		"response":   gen.Response,
		"model":      gen.Model,
		"eval_count": gen.EvalCount,
		"backend":    r.backend,
	}, nil
}

func (r *ollamaRuntime) Cleanup(force bool) error { return nil }
