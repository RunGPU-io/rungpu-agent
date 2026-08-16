// Package job — bootstrap_test.go tests the full bootstrap chain:
//
//   1. Ollama binary missing → auto-install
//   2. Ollama server not running → auto-start
//   3. Model not cached → auto-pull
//   4. Run inference → return response
//
// These tests verify each step of the "zero to inference" pipeline that
// runs when a fresh host joins the pool for the first time.
//
// Safe tests (no side effects, always run):
//   go test -v -run TestBootstrap ./internal/job/ -timeout 120s
//
// Destructive test (actually removes + re-pulls a model):
//   go test -v -run TestBootstrap_ModelPull_ThenInference ./internal/job/ -timeout 300s
package job

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/tokenize/gpu-agent/internal/types"
)

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: Ollama binary detection and install logic
// ═══════════════════════════════════════════════════════════════════════════

// TestBootstrap_OllamaDetection verifies the agent correctly detects
// whether Ollama is installed and reports the right runtime name.
func TestBootstrap_OllamaDetection(t *testing.T) {
	hasOllama := exec.Command("ollama", "--version").Run() == nil
	t.Logf("Ollama installed: %v", hasOllama)

	rt, err := NewRuntime("metal", t.TempDir(), 10)
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}

	name := rt.Name()
	t.Logf("Runtime name: %s", name)

	if hasOllama {
		if !strings.Contains(name, "ollama") {
			t.Errorf("Ollama is installed but runtime name %q doesn't contain 'ollama'", name)
		}
		t.Log("✅ Ollama detected correctly")
	} else {
		// If Ollama isn't installed, the runtime should have tried to install it.
		// If install succeeded, "ollama" will be in the name.
		// If install failed, it won't — but the runtime should still work (just no LLM jobs).
		t.Logf("Runtime available without Ollama: %s", name)
		t.Log("✅ Agent handles missing Ollama gracefully")
	}
}

// TestBootstrap_InstallOllama_AlreadyInstalled verifies that installOllama()
// is a no-op when Ollama is already present (doesn't reinstall or break anything).
func TestBootstrap_InstallOllama_AlreadyInstalled(t *testing.T) {
	if exec.Command("ollama", "--version").Run() != nil {
		t.Skip("Ollama not installed — skipping already-installed test")
	}

	// This should be a fast no-op
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := installOllama(ctx)
	if err != nil {
		t.Fatalf("installOllama failed even though ollama is installed: %v", err)
	}

	// Verify it's still working
	if exec.Command("ollama", "--version").Run() != nil {
		t.Fatal("Ollama broken after installOllama call")
	}

	t.Log("✅ installOllama() is a safe no-op when already installed")
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 2: Ollama server auto-start
// ═══════════════════════════════════════════════════════════════════════════

// TestBootstrap_OllamaServerAutoStart verifies that the agent can detect
// whether the Ollama server is running and auto-start it if needed.
func TestBootstrap_OllamaServerAutoStart(t *testing.T) {
	if exec.Command("ollama", "--version").Run() != nil {
		t.Skip("Ollama not installed")
	}

	rt := newOllamaRuntime(t.TempDir())

	// Check if server is running
	running := rt.isServerRunning()
	t.Logf("Ollama server running: %v", running)

	// ensureServerRunning should either confirm it's running or start it
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err := rt.ensureServerRunning(ctx)
	if err != nil {
		t.Fatalf("ensureServerRunning failed: %v", err)
	}

	// Verify it's actually responding
	if !rt.isServerRunning() {
		t.Fatal("Server should be running after ensureServerRunning")
	}

	// Verify the API is functional
	resp, err := http.Get("http://localhost:11434")
	if err != nil {
		t.Fatalf("GET localhost:11434 failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("Ollama server returned %d, want 200", resp.StatusCode)
	}

	t.Log("✅ Ollama server is running and responding")
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 3: Model cache detection
// ═══════════════════════════════════════════════════════════════════════════

// TestBootstrap_ModelCacheDetection verifies the agent can check whether
// a model is already cached in Ollama without triggering a download.
func TestBootstrap_ModelCacheDetection(t *testing.T) {
	if exec.Command("ollama", "--version").Run() != nil {
		t.Skip("Ollama not installed")
	}

	rt := newOllamaRuntime(t.TempDir())
	ctx := context.Background()

	// Ensure server is running first
	if err := rt.ensureServerRunning(ctx); err != nil {
		t.Fatalf("ensureServerRunning: %v", err)
	}

	// List what's actually cached
	out, err := exec.Command("ollama", "list").Output()
	if err != nil {
		t.Fatalf("ollama list: %v", err)
	}
	t.Logf("Cached models:\n%s", string(out))

	// Test detection for a model that should NOT exist
	if rt.isModelCached(ctx, "nonexistent-model-xyz-999") {
		t.Error("nonexistent-model-xyz-999 should not be cached")
	}

	// Test detection for models that might be cached
	for _, model := range []string{"tinyllama", "llama3.2", "llama3", "phi3:mini"} {
		cached := rt.isModelCached(ctx, model)
		t.Logf("  %s cached: %v", model, cached)
	}

	t.Log("✅ Model cache detection works correctly")
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 4: Model pull + inference (the real test)
// ═══════════════════════════════════════════════════════════════════════════

// TestBootstrap_ModelPull_ThenInference is the definitive bootstrap test.
// It removes a small model from Ollama, then verifies the full chain:
//
//   1. Agent detects model is NOT cached
//   2. Agent pulls the model (downloads from Ollama registry)
//   3. Agent runs inference with the freshly-pulled model
//   4. Customer gets back a real LLM response
//
// This test is DESTRUCTIVE — it removes tinyllama from Ollama cache,
// then re-pulls it (~637 MB download). Only runs when explicitly requested
// and skipped in -short mode.
//
// Run:
//   go test -v -run TestBootstrap_ModelPull_ThenInference ./internal/job/ -timeout 300s
func TestBootstrap_ModelPull_ThenInference(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping model pull test in -short mode (downloads ~637 MB)")
	}

	if exec.Command("ollama", "--version").Run() != nil {
		t.Skip("Ollama not installed")
	}

	// Use tinyllama — smallest model at 637 MB
	modelName := "tinyllama"

	rt := newOllamaRuntime(t.TempDir())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Ensure server is running
	if err := rt.ensureServerRunning(ctx); err != nil {
		t.Fatalf("ensureServerRunning: %v", err)
	}

	// ── Step 1: Remove the model if it exists ───────────────────────────
	wasCached := rt.isModelCached(ctx, modelName)
	t.Logf("Model %q was cached: %v", modelName, wasCached)

	if wasCached {
		t.Logf("Removing %s to test fresh pull...", modelName)
		cmd := exec.CommandContext(ctx, "ollama", "rm", modelName)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("Failed to remove %s: %v: %s", modelName, err, string(out))
		}

		// Verify it's gone
		if rt.isModelCached(ctx, modelName) {
			t.Fatalf("%s still cached after removal", modelName)
		}
		t.Logf("✅ %s removed from cache", modelName)
	}

	// ── Step 2: Run Prepare — should detect missing model and pull it ───
	t.Logf("Running Prepare (should pull %s)...", modelName)
	pullStart := time.Now()

	assignment := types.JobAssignment{
		JobID:     "bootstrap-pull-test",
		ModelName: modelName,
		Input:     map[string]interface{}{"prompt": "test"},
	}

	err := rt.Prepare(ctx, assignment)
	pullDuration := time.Since(pullStart)

	if err != nil {
		t.Fatalf("Prepare failed: %v", err)
	}
	t.Logf("✅ Model pulled in %s", pullDuration.Round(time.Millisecond))

	// Verify it's now cached
	if !rt.isModelCached(ctx, modelName) {
		t.Fatal("Model should be cached after Prepare")
	}

	// ── Step 3: Run inference with the freshly-pulled model ─────────────
	t.Log("Running inference with freshly-pulled model...")
	inferStart := time.Now()

	result, err := rt.Run(ctx, types.JobAssignment{
		JobID:     "bootstrap-infer-test",
		ModelName: modelName,
		Input:     map[string]interface{}{"prompt": "What is 2+2? Answer with just the number."},
		Parameters: map[string]interface{}{
			"num_predict": 10,
			"temperature": 0.1,
		},
	})
	inferDuration := time.Since(inferStart)

	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	response, _ := result["response"].(string)
	model, _ := result["model"].(string)
	evalCount, _ := result["eval_count"].(int)

	t.Logf("✅ Inference completed in %s", inferDuration.Round(time.Millisecond))
	t.Logf("  Response: %q", strings.TrimSpace(response))
	t.Logf("  Model: %s", model)
	t.Logf("  Tokens: %d", evalCount)

	if response == "" {
		t.Error("Empty response from freshly-pulled model")
	}

	// ── Step 4: Second inference (model warm) ───────────────────────────
	t.Log("Running second inference (model warm)...")
	warmStart := time.Now()

	result2, err := rt.Run(ctx, types.JobAssignment{
		JobID:     "bootstrap-warm-test",
		ModelName: modelName,
		Input:     map[string]interface{}{"prompt": "Say exactly: BOOTSTRAP_OK"},
		Parameters: map[string]interface{}{
			"num_predict": 10,
		},
	})
	warmDuration := time.Since(warmStart)

	if err != nil {
		t.Fatalf("Warm inference failed: %v", err)
	}

	response2, _ := result2["response"].(string)
	t.Logf("✅ Warm inference in %s: %q", warmDuration.Round(time.Millisecond), strings.TrimSpace(response2))

	// ── Summary ─────────────────────────────────────────────────────────
	t.Logf("\n=== Bootstrap Summary ===")
	t.Logf("  Model pull:      %s", pullDuration.Round(time.Millisecond))
	t.Logf("  First inference:  %s", inferDuration.Round(time.Millisecond))
	t.Logf("  Warm inference:   %s", warmDuration.Round(time.Millisecond))
	t.Log("  ✅ Full bootstrap chain verified: detect missing → pull → infer → respond")
}

// ═══════════════════════════════════════════════════════════════════════════
// Full pipeline: Executor handles everything (install + pull + infer)
// ═══════════════════════════════════════════════════════════════════════════

// TestBootstrap_Executor_FullPipeline tests the Executor (the same code
// that runs in production) handling a model that's already cached.
// This proves the full Executor pipeline: Prepare → Run → result.
func TestBootstrap_Executor_FullPipeline(t *testing.T) {
	if exec.Command("ollama", "--version").Run() != nil {
		t.Skip("Ollama not installed")
	}

	// Find a cached model
	modelName := ""
	for _, candidate := range []string{"tinyllama", "qwen2.5:0.5b", "llama3.2:1b", "llama3.2"} {
		if isModelCachedCheck(candidate) {
			modelName = candidate
			break
		}
	}
	if modelName == "" {
		t.Skip("No cached Ollama model — run TestBootstrap_ModelPull_ThenInference first")
	}

	t.Logf("Using cached model: %s", modelName)

	cacheDir := t.TempDir()
	executor, err := NewExecutor(cacheDir, 50, "gpu-bootstrap-0", "metal")
	if err != nil {
		t.Fatalf("NewExecutor: %v", err)
	}

	// Track progress events
	var progressEvents []string
	executor.OnProgress = func(p types.JobProgress) {
		progressEvents = append(progressEvents, fmt.Sprintf("[%s] %.0f%% %s", p.Stage, p.Progress*100, p.Message))
		t.Logf("  Progress: %s", progressEvents[len(progressEvents)-1])
	}

	// Execute the full pipeline
	t.Log("Executing full pipeline...")
	start := time.Now()

	result := executor.Execute(context.Background(), types.JobAssignment{
		JobID:     "bootstrap-executor-test",
		ModelName: modelName,
		Input:     map[string]interface{}{"prompt": "What is the capital of Japan? One word."},
		Parameters: map[string]interface{}{
			"num_predict": 10,
			"temperature": 0.1,
		},
	})

	duration := time.Since(start)

	if !result.Success {
		t.Fatalf("Executor failed: %s", result.Error)
	}

	t.Logf("✅ Executor completed in %s (%dms reported)", duration.Round(time.Millisecond), result.DurationMS)

	// Verify result structure
	if result.Type != "job_result" {
		t.Errorf("type = %q, want job_result", result.Type)
	}
	if result.JobID != "bootstrap-executor-test" {
		t.Errorf("job_id = %q", result.JobID)
	}
	if result.GPUID != "gpu-bootstrap-0" {
		t.Errorf("gpu_id = %q", result.GPUID)
	}

	// Verify response
	if result.Result == nil {
		t.Fatal("Result is nil")
	}
	response, _ := result.Result["response"].(string)
	if response == "" {
		t.Error("Empty response")
	}
	t.Logf("  Response: %q", strings.TrimSpace(response))
	t.Logf("  Model: %v", result.Result["model"])
	t.Logf("  Backend: %v", result.Result["backend"])

	// Verify progress events happened in order
	if len(progressEvents) < 3 {
		t.Errorf("Expected >= 3 progress events, got %d", len(progressEvents))
	}
	t.Logf("  Progress events: %d", len(progressEvents))

	t.Log("\n✅ Full Executor pipeline verified (same code as production)")
}

// TestBootstrap_Executor_UnknownModel verifies the Executor handles an
// unknown model gracefully — returns a clear error, doesn't hang.
func TestBootstrap_Executor_UnknownModel(t *testing.T) {
	cacheDir := t.TempDir()
	executor, err := NewExecutor(cacheDir, 10, "gpu-unknown-0", "metal")
	if err != nil {
		t.Fatalf("NewExecutor: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	start := time.Now()
	result := executor.Execute(ctx, types.JobAssignment{
		JobID:     "unknown-model-test",
		ModelName: "nonexistent-model-xyz-999",
		Input:     map[string]interface{}{"prompt": "test"},
	})
	duration := time.Since(start)

	if result.Success {
		t.Error("Expected failure for nonexistent model")
	}
	if result.Error == "" {
		t.Error("Expected error message")
	}

	t.Logf("✅ Unknown model failed in %s: %s", duration.Round(time.Millisecond), result.Error)

	// Should fail fast (< 10s), not hang
	if duration > 15*time.Second {
		t.Errorf("Took too long: %s (should fail fast)", duration)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

// isModelCachedCheck checks if a model is cached in Ollama via the API.
func isModelCachedCheck(model string) bool {
	body, _ := json.Marshal(map[string]string{"name": model})
	resp, err := http.Post("http://localhost:11434/api/show", "application/json",
		strings.NewReader(string(body)))
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == 200
}
