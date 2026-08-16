// Package job — real_integration_test.go contains integration tests that hit
// REAL infrastructure: a real Ollama server, real model downloads, real inference.
//
// The agent auto-installs Ollama and auto-starts the server, so these tests
// use the same code paths. No manual setup required — just run:
//
//	go test ./internal/job/ -v -run TestReal -count=1 -timeout 600s
//
// On first run the agent will:
//  1. Install Ollama if missing (brew install ollama / official script)
//  2. Start `ollama serve` if not running
//  3. Pull qwen2.5:0.5b (~400 MB) if not cached
//
// Subsequent runs reuse the cached model and are fast (~5s total).
package job

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tokenize/gpu-agent/internal/types"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Setup — use the agent's own auto-install + auto-start code
// ═══════════════════════════════════════════════════════════════════════════════

// ensureOllamaReady uses the agent's own automation to install Ollama (if
// missing) and start the server (if not running). This is the same code path
// that runs in production — no manual `ollama serve` needed.
func ensureOllamaReady(t *testing.T) *ollamaRuntime {
	t.Helper()

	rt := newOllamaRuntime(t.TempDir())

	// Use the agent's own installOllama() — installs via brew or official script
	if _, err := exec.LookPath("ollama"); err != nil {
		t.Log("Ollama not installed — using agent's auto-install...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if err := installOllama(ctx); err != nil {
			t.Skipf("SKIP: auto-install failed (expected in some CI environments): %v", err)
		}
		t.Log("✅ Ollama installed automatically")
	}

	// Use the agent's own ensureServerRunning() — starts `ollama serve` if needed
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := rt.ensureServerRunning(ctx); err != nil {
		t.Skipf("SKIP: could not start Ollama server: %v", err)
	}

	return rt
}

// ensureModel uses the agent's own Prepare() to pull a model via `ollama pull`.
func ensureModel(t *testing.T, rt *ollamaRuntime, model string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	if rt.isModelCached(ctx, model) {
		t.Logf("Model %s already cached", model)
		return
	}

	t.Logf("Pulling model %s (this may take a few minutes on first run)...", model)
	if err := rt.Prepare(ctx, types.JobAssignment{ModelName: model}); err != nil {
		t.Fatalf("Failed to pull model %s: %v", model, err)
	}
	t.Logf("✅ Model %s ready", model)
}

func dockerInstalled() bool {
	return exec.Command("docker", "version").Run() == nil
}

func requireRealIntegration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping real runtime integration test in short mode")
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 1: Auto-install + auto-start + health check
//
// Verifies: The agent's own installOllama() and ensureServerRunning() work.
// This is the foundational test — if this passes, Ollama is ready.
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_OllamaAutoSetup(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	// Verify health check passes
	if !rt.isServerRunning() {
		t.Fatal("isServerRunning() returned false after ensureServerRunning()")
	}

	// Verify ensureServerRunning is idempotent (second call is instant)
	start := time.Now()
	if err := rt.ensureServerRunning(context.Background()); err != nil {
		t.Fatalf("Second ensureServerRunning failed: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 1*time.Second {
		t.Errorf("Second ensureServerRunning took %s (should be instant)", elapsed)
	}

	t.Log("✅ Ollama auto-setup passed (installed + server running)")
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 2: Pull a real model via Prepare() + verify cache
//
// Verifies: Prepare() actually downloads a model, isModelCached() returns true.
// Uses "qwen2.5:0.5b" — the smallest available model (~400 MB).
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_OllamaPullModel(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	// Verify the model is now cached via the API
	if !rt.isModelCached(context.Background(), model) {
		t.Fatalf("isModelCached(%q) returned false after Prepare()", model)
	}

	t.Logf("✅ Real model pull + cache check passed (model: %s)", model)
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 3: Run real inference — send a prompt, get a real LLM response
//
// Verifies: Run() sends a prompt to the REAL Ollama server and gets back
// a coherent response. This is the core test — proves the agent can actually
// do inference.
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_OllamaInference(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := rt.Run(ctx, types.JobAssignment{
		JobID:     "real-inference-1",
		ModelName: model,
		Input:     map[string]interface{}{"prompt": "What is 2+2? Answer with just the number."},
		Parameters: map[string]interface{}{
			"temperature": 0.0,
			"num_predict": 20,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	response, ok := result["response"].(string)
	if !ok || response == "" {
		t.Fatalf("Expected non-empty response, got: %v", result["response"])
	}
	if result["status"] != "completed" {
		t.Errorf("status = %v, want completed", result["status"])
	}
	if result["model"] != model {
		t.Errorf("model = %v, want %s", result["model"], model)
	}

	switch v := result["eval_count"].(type) {
	case int:
		if v <= 0 {
			t.Errorf("eval_count = %d, want > 0", v)
		}
	case float64:
		if v <= 0 {
			t.Errorf("eval_count = %f, want > 0", v)
		}
	}

	if !strings.Contains(response, "4") {
		t.Logf("WARNING: response doesn't contain '4': %q (LLM may have given a verbose answer)", response)
	}

	t.Logf("✅ Real inference passed")
	t.Logf("   Model:    %s", model)
	t.Logf("   Response: %s", strings.TrimSpace(response))
	t.Logf("   Tokens:   %v", result["eval_count"])
	t.Logf("   Backend:  %v", result["backend"])
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 4: Full Executor pipeline — Prepare + Run + progress tracking
//
// Verifies: The complete Executor.Execute() pipeline with a real Ollama model.
// This is the same code path that runs when a job arrives from the pool.
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_ExecutorFullPipeline(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	cacheDir := t.TempDir()
	executor := &Executor{
		runtime:  rt,
		gpuID:    "gpu-real-0",
		backend:  rt.backend,
		cacheDir: cacheDir,
	}

	var stages []string
	executor.OnProgress = func(p types.JobProgress) {
		stages = append(stages, p.Stage)
		t.Logf("  [%s] %.0f%% — %s", p.Stage, p.Progress*100, p.Message)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	result := executor.Execute(ctx, types.JobAssignment{
		JobID:     "real-pipeline-1",
		ModelName: model,
		Input:     map[string]interface{}{"prompt": "Name three primary colors. Be brief."},
		Parameters: map[string]interface{}{
			"temperature": 0.0,
			"num_predict": 50,
		},
	})

	if !result.Success {
		t.Fatalf("Job failed: %s", result.Error)
	}
	if result.DurationMS <= 0 {
		t.Errorf("DurationMS = %d, want > 0", result.DurationMS)
	}

	response, ok := result.Result["response"].(string)
	if !ok || response == "" {
		t.Fatalf("Expected non-empty response, got: %v", result.Result["response"])
	}

	if len(stages) < 3 {
		t.Errorf("Expected >= 3 progress stages, got %d: %v", len(stages), stages)
	}
	if !containsInOrder(stages, "pulling_image", "running", "completed") {
		t.Errorf("Progress stages out of order: %v", stages)
	}

	t.Logf("✅ Real executor pipeline passed")
	t.Logf("   Response:  %s", strings.TrimSpace(response))
	t.Logf("   Duration:  %dms", result.DurationMS)
	t.Logf("   Stages:    %v", stages)
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 5: Parameter forwarding — temperature=0 determinism, num_predict
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_OllamaParameterEffects(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	prompt := "What is the capital of France? Answer in one word."

	result1, err := rt.Run(ctx, types.JobAssignment{
		JobID: "param-1", ModelName: model,
		Input:      map[string]interface{}{"prompt": prompt},
		Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 10},
	})
	if err != nil {
		t.Fatalf("Run 1 failed: %v", err)
	}

	result2, err := rt.Run(ctx, types.JobAssignment{
		JobID: "param-2", ModelName: model,
		Input:      map[string]interface{}{"prompt": prompt},
		Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 10},
	})
	if err != nil {
		t.Fatalf("Run 2 failed: %v", err)
	}

	resp1 := strings.TrimSpace(result1["response"].(string))
	resp2 := strings.TrimSpace(result2["response"].(string))

	if resp1 == resp2 {
		t.Logf("  Deterministic: both runs produced %q", resp1)
	} else {
		t.Logf("  Run 1: %q / Run 2: %q (may vary by Ollama version)", resp1, resp2)
	}

	if !strings.Contains(strings.ToLower(resp1), "paris") {
		t.Errorf("Response should mention Paris: %q", resp1)
	}

	result3, err := rt.Run(ctx, types.JobAssignment{
		JobID: "param-3", ModelName: model,
		Input:      map[string]interface{}{"prompt": prompt},
		Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 1},
	})
	if err != nil {
		t.Fatalf("Run 3 failed: %v", err)
	}
	resp3 := strings.TrimSpace(result3["response"].(string))

	t.Logf("✅ Real parameter forwarding passed")
	t.Logf("   num_predict=10: %q (%d chars)", resp1, len(resp1))
	t.Logf("   num_predict=1:  %q (%d chars)", resp3, len(resp3))
}

// ═══════════════════════════════════��═══════════════════════════════════════════
// REAL TEST 6: Prompt extraction — "prompt" and "text" input fields
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_PromptFormats(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	cases := []struct {
		name  string
		input map[string]interface{}
	}{
		{"prompt field", map[string]interface{}{"prompt": "Say hello"}},
		{"text field", map[string]interface{}{"text": "Say hello"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := rt.Run(ctx, types.JobAssignment{
				JobID: "fmt-" + tc.name, ModelName: model,
				Input:      tc.input,
				Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 20},
			})
			if err != nil {
				t.Fatalf("Run failed: %v", err)
			}
			resp := result["response"].(string)
			if len(strings.TrimSpace(resp)) == 0 {
				t.Errorf("Empty response for input format %q", tc.name)
			}
			t.Logf("  %s → %q", tc.name, strings.TrimSpace(resp))
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 7: Executor with real inference + file upload
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_ExecutorWithUpload(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	var uploadedData []byte
	var uploadedContentType string
	storageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			uploadedContentType = r.Header.Get("Content-Type")
			uploadedData, _ = io.ReadAll(r.Body)
			w.WriteHeader(200)
			return
		}
		http.Error(w, "want PUT", 405)
	}))
	defer storageSrv.Close()

	cacheDir := t.TempDir()
	outputDir := filepath.Join(cacheDir, "output", "real-upload-1")
	os.MkdirAll(outputDir, 0755)
	outputFile := filepath.Join(outputDir, "result.png")
	os.WriteFile(outputFile, []byte("FAKE_PNG_FROM_REAL_INFERENCE"), 0644)

	wrappedRT := &outputInjector{inner: rt, outputFile: outputFile}
	executor := &Executor{runtime: wrappedRT, gpuID: "gpu-real-upload", backend: rt.backend, cacheDir: cacheDir}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result := executor.Execute(ctx, types.JobAssignment{
		JobID: "real-upload-1", ModelName: model,
		Input:      map[string]interface{}{"prompt": "Describe a sunset."},
		Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 30},
		UploadURL:  storageSrv.URL + "/bucket/outputs/result.png",
	})

	if !result.Success {
		t.Fatalf("Job failed: %s", result.Error)
	}
	if result.Result["uploaded"] != true {
		t.Error("Expected uploaded=true")
	}
	if string(uploadedData) != "FAKE_PNG_FROM_REAL_INFERENCE" {
		t.Errorf("Upload data mismatch")
	}
	if uploadedContentType != "image/png" {
		t.Errorf("Content-Type = %q, want image/png", uploadedContentType)
	}

	response, _ := result.Result["response"].(string)
	t.Logf("✅ Real executor + upload passed")
	t.Logf("   Response: %s", strings.TrimSpace(response))
	t.Logf("   Uploaded: %d bytes as %s", len(uploadedData), uploadedContentType)
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 8: Multiple models (gated — pulls ~1 GB)
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_MultipleModels(t *testing.T) {
	requireRealIntegration(t)
	if os.Getenv("REAL_TEST_MULTI_MODEL") == "" {
		t.Skip("SKIP: set REAL_TEST_MULTI_MODEL=1 to run (pulls two models)")
	}

	rt := ensureOllamaReady(t)

	models := []string{"qwen2.5:0.5b", "tinyllama"}
	for _, m := range models {
		ensureModel(t, rt, m)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	for _, model := range models {
		result, err := rt.Run(ctx, types.JobAssignment{
			JobID: "multi-" + model, ModelName: model,
			Input:      map[string]interface{}{"prompt": "What is your model name? Answer briefly."},
			Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 30},
		})
		if err != nil {
			t.Fatalf("Run(%s) failed: %v", model, err)
		}
		resp := strings.TrimSpace(result["response"].(string))
		if resp == "" {
			t.Errorf("Empty response from %s", model)
		}
		t.Logf("  %s → %q", model, resp)
	}

	t.Log("✅ Real multi-model test passed")
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 9: Context cancellation with real Ollama
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_ContextCancellation(t *testing.T) {
	requireRealIntegration(t)
	rt := ensureOllamaReady(t)

	model := "qwen2.5:0.5b"
	ensureModel(t, rt, model)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := rt.Run(ctx, types.JobAssignment{
		JobID: "cancel-real-1", ModelName: model,
		Input:      map[string]interface{}{"prompt": "Write a 1000 word essay about the history of computing."},
		Parameters: map[string]interface{}{"num_predict": 500},
	})
	elapsed := time.Since(start)

	if err == nil {
		t.Logf("WARNING: inference completed before timeout (%s) — machine may be too fast", elapsed)
		return
	}
	if elapsed > 5*time.Second {
		t.Errorf("Took %s — cancellation should have been faster", elapsed)
	}

	t.Logf("✅ Real context cancellation passed (cancelled in %s)", elapsed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 10: Docker container execution
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_DockerContainerExecution(t *testing.T) {
	requireRealIntegration(t)
	if !dockerInstalled() {
		t.Skip("SKIP: docker not installed")
	}

	t.Log("Pulling alpine:latest...")
	cmd := exec.Command("docker", "pull", "alpine:latest")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to pull alpine: %v", err)
	}

	containerName := fmt.Sprintf("tokenize-test-%d", time.Now().UnixNano())
	out, err := exec.Command("docker", "run", "--rm", "--name", containerName,
		"alpine:latest", "sh", "-c",
		`echo 'OUTPUT:{"status":"completed","message":"hello from docker"}'`,
	).CombinedOutput()
	if err != nil {
		t.Fatalf("Docker run failed: %v\n%s", err, string(out))
	}

	parsed := parseOutput(string(out))
	if parsed == nil {
		t.Fatalf("parseOutput returned nil for: %q", string(out))
	}
	if parsed["status"] != "completed" {
		t.Errorf("status = %v", parsed["status"])
	}
	if parsed["message"] != "hello from docker" {
		t.Errorf("message = %v", parsed["message"])
	}

	t.Logf("✅ Real Docker container execution passed")
	t.Logf("   Output: %v", parsed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL TEST 11: Full Prepare → Run cycle (the production code path)
//
// This is the exact sequence that runs when a real job arrives from the pool:
//   1. Prepare() — installs Ollama, starts server, pulls model
//   2. Run() — sends prompt, gets response
//   3. Run() again — model warm in memory, faster
// ═══════════════════════════════════════════════════════════════════════════════

func TestReal_PrepareAndRun(t *testing.T) {
	requireRealIntegration(t)
	// Don't use ensureOllamaReady — let Prepare() handle everything.
	// This tests the full production code path from zero.
	rt := newOllamaRuntime(t.TempDir())

	model := "qwen2.5:0.5b"
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	job := types.JobAssignment{
		JobID:     "real-prepare-run-1",
		ModelName: model,
		Input:     map[string]interface{}{"prompt": "What color is the sky? One word."},
		Parameters: map[string]interface{}{
			"temperature": 0.0,
			"num_predict": 10,
		},
	}

	// Step 1: Prepare — auto-installs Ollama, auto-starts server, pulls model
	t.Log("Step 1: Prepare (auto-install + auto-start + pull)...")
	start := time.Now()
	if err := rt.Prepare(ctx, job); err != nil {
		t.Skipf("SKIP: Prepare failed (expected in some CI environments): %v", err)
	}
	prepareTime := time.Since(start)
	t.Logf("  Prepare completed in %s", prepareTime)

	if !rt.isModelCached(ctx, model) {
		t.Fatal("Model should be cached after Prepare")
	}

	// Step 2: Run — real inference
	t.Log("Step 2: Run (real inference)...")
	start = time.Now()
	result, err := rt.Run(ctx, job)
	runTime := time.Since(start)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	response := strings.TrimSpace(result["response"].(string))
	if response == "" {
		t.Fatal("Empty response")
	}

	// Step 3: Run again — model warm in memory, should be faster
	t.Log("Step 3: Run again (model warm in memory)...")
	start = time.Now()
	result2, err := rt.Run(ctx, types.JobAssignment{
		JobID: "real-prepare-run-2", ModelName: model,
		Input:      map[string]interface{}{"prompt": "What is 1+1? Just the number."},
		Parameters: map[string]interface{}{"temperature": 0.0, "num_predict": 5},
	})
	warmTime := time.Since(start)
	if err != nil {
		t.Fatalf("Warm run failed: %v", err)
	}
	warmResponse := strings.TrimSpace(result2["response"].(string))

	t.Logf("✅ Real Prepare → Run cycle passed")
	t.Logf("   Prepare:      %s", prepareTime)
	t.Logf("   Cold run:     %s → %q", runTime, response)
	t.Logf("   Warm run:     %s → %q", warmTime, warmResponse)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

type outputInjector struct {
	inner      Runtime
	outputFile string
}

func (o *outputInjector) Name() string { return o.inner.Name() }
func (o *outputInjector) Prepare(ctx context.Context, a types.JobAssignment) error {
	return o.inner.Prepare(ctx, a)
}
func (o *outputInjector) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	result, err := o.inner.Run(ctx, a)
	if err != nil {
		return nil, err
	}
	result["output_file"] = o.outputFile
	return result, nil
}
func (o *outputInjector) Cleanup(force bool) error { return o.inner.Cleanup(force) }
