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
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const defaultOllamaEndpoint = "http://localhost:11434"

type ollamaRuntime struct {
	cacheDir string
	endpoint string
	backend  string
	client   *http.Client
}

func newOllamaRuntime(cacheDir string) *ollamaRuntime {
	backend := "cpu"
	if rt.GOOS == "darwin" && rt.GOARCH == "arm64" {
		backend = "metal"
	}
	return &ollamaRuntime{
		cacheDir: cacheDir,
		endpoint: defaultOllamaEndpoint,
		backend:  backend,
		client:   &http.Client{Timeout: 5 * time.Minute},
	}
}

func (r *ollamaRuntime) Name() string { return "ollama" }

// ollamaModel resolves the Ollama tag for a job.
func ollamaModel(a types.JobAssignment) string {
	if a.Parameters != nil {
		if v, ok := a.Parameters["ollama_model"].(string); ok && v != "" {
			return v
		}
	}
	return a.ModelName
}

// extractPrompt pulls a text prompt out of the job input.
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

// isServerRunning checks if the Ollama HTTP server is responding.
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
	// Ollama returns 200 on GET / with "Ollama is running"
	return resp.StatusCode == http.StatusOK
}

// ensureServerRunning checks if the Ollama server is up. If not, it starts
// `ollama serve` in the background and waits for it to become ready.
// This handles the common case where ollama is installed but the user hasn't
// started the server (especially on macOS where it's not auto-started).
//
// Only auto-starts when using the default endpoint (localhost:11434).
// Custom endpoints (e.g. in tests) are assumed to be managed externally.
func (r *ollamaRuntime) ensureServerRunning(ctx context.Context) error {
	// Already running? Great.
	if r.isServerRunning() {
		return nil
	}

	// Only auto-start for the default local endpoint.
	// Custom endpoints (tests, remote servers) are managed externally.
	if r.endpoint != defaultOllamaEndpoint {
		return fmt.Errorf("ollama server not responding at %s", r.endpoint)
	}

	fmt.Println("[ollama] Server not running — starting it automatically...")

	// Start `ollama serve` in the background.
	// On macOS, the Ollama app may also work — but `ollama serve` is the
	// CLI way that works everywhere.
	cmd := exec.Command("ollama", "serve")
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Env = os.Environ()

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ollama server: %w\n"+
			"  Start it manually with: ollama serve", err)
	}

	// Don't wait for the process — let it run in the background.
	go func() {
		cmd.Wait()
	}()

	// Wait for the server to become ready (up to 30 seconds).
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

// isModelCached checks if the model is already downloaded by querying the
// Ollama /api/show endpoint. Returns true if the model exists locally.
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

// installOllama preserves the existing setup probe for diagnostics and tests.
// Installation is intentionally restricted to the explicit setup command.
func installOllama(ctx context.Context) error {
	_ = ctx
	if _, err := exec.LookPath("ollama"); err == nil {
		return nil
	}
	return fmt.Errorf("ollama is not installed; run rungpu-agent setup")
}

// Prepare ensures the model is available locally, starts the server if needed,
// checks the cache, and pulls the model if not present.
func (r *ollamaRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	model := ollamaModel(a)

	// 1. Check if ollama binary exists.
	if _, err := exec.LookPath("ollama"); err != nil {
		return fmt.Errorf("ollama is not installed; run rungpu-agent setup")
	}

	// 2. Ensure the Ollama server is running (auto-start if needed)
	if err := r.ensureServerRunning(ctx); err != nil {
		return err
	}

	// 3. Check if model is already cached via Ollama API (fast, no download)
	if r.isModelCached(ctx, model) {
		return nil // already downloaded — skip pull
	}

	// 4. Model not cached — pull it (downloads from Ollama registry)
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

// Run sends a generate request to the local Ollama server.
func (r *ollamaRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	// Ensure server is still running (it may have been stopped between Prepare and Run)
	if err := r.ensureServerRunning(ctx); err != nil {
		return nil, err
	}

	model := ollamaModel(a)
	prompt := extractPrompt(a)

	reqBody, _ := json.Marshal(map[string]interface{}{
		"model":   model,
		"prompt":  prompt,
		"stream":  false,
		"options": a.Parameters,
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
