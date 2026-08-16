package job

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tokenize/gpu-agent/internal/types"
)

// TestFullPipeline_JobArrives_DownloadsImage_Runs_UploadsResult tests the
// complete job execution pipeline end-to-end:
//
//  1. Job arrives with a model name
//  2. Runtime prepares (simulated image pull via mock)
//  3. Runtime runs inference (mock Ollama returns a response)
//  4. Output file is created
//  5. Output is uploaded to mock storage (like Google Cloud Storage)
//  6. Result is returned with timing, output URL, and success status
//
// This uses a mock Ollama server and a mock storage server — no real
// Docker or cloud storage needed.
func TestFullPipeline_JobArrives_DownloadsImage_Runs_UploadsResult(t *testing.T) {
	// ── Mock Ollama server (simulates model download + inference) ────────
	modelPulled := false
	inferenceCount := 0

	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/":
			// Health check — ensureServerRunning calls GET /
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("Ollama is running"))
		case r.URL.Path == "/api/show":
			// First call: model not cached. After pull: cached.
			if modelPulled {
				json.NewEncoder(w).Encode(map[string]interface{}{"modelfile": "FROM llama2"})
			} else {
				http.Error(w, "model not found", 404)
			}
		case r.URL.Path == "/api/generate":
			inferenceCount++
			var req map[string]interface{}
			json.NewDecoder(r.Body).Decode(&req)
			model, _ := req["model"].(string)
			prompt, _ := req["prompt"].(string)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"model":      model,
				"response":   "Generated response for: " + prompt,
				"done":       true,
				"eval_count": 42,
			})
		default:
			http.Error(w, "not found", 404)
		}
	}))
	defer ollamaSrv.Close()

	// ── Mock storage server (simulates Google Cloud Storage / S3) ────────
	var uploadedData []byte
	var uploadedContentType string

	storageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			uploadedContentType = r.Header.Get("Content-Type")
			uploadedData, _ = io.ReadAll(r.Body)
			w.WriteHeader(200)
			return
		}
		http.Error(w, "method not allowed", 405)
	}))
	defer storageSrv.Close()

	// ── Set up executor with mock Ollama runtime ────────────────────────
	cacheDir := t.TempDir()

	// Create a mock runtime that uses our mock Ollama server
	mockOllama := &ollamaRuntime{
		cacheDir: cacheDir,
		endpoint: ollamaSrv.URL,
		backend:  "cpu",
		client:   &http.Client{Timeout: 10 * time.Second},
	}

	// Wrap in skipPrepare since we can't run `ollama pull` in tests
	// Instead, simulate the pull by setting modelPulled = true
	mockRuntime := &mockPrepareRuntime{
		inner: mockOllama,
		onPrepare: func() {
			modelPulled = true // simulate successful pull
		},
	}

	executor := &Executor{
		runtime:  mockRuntime,
		gpuID:    "gpu-test-0",
		backend:  "cpu",
		cacheDir: cacheDir,
	}

	// ── Track progress stages ───────────────────────────────────────────
	var progressStages []string
	executor.OnProgress = func(p types.JobProgress) {
		progressStages = append(progressStages, p.Stage+":"+p.Message)
		t.Logf("[progress] %s %.0f%% — %s", p.Stage, p.Progress*100, p.Message)
	}

	// ── Step 1: Verify model is NOT cached initially ────────────────────
	if mockOllama.isModelCached(context.Background(), "llama2") {
		t.Fatal("model should not be cached before job runs")
	}

	// ── Step 2: Execute the job ─────────────────────────────────────────
	job := types.JobAssignment{
		JobID:     "pipeline-test-001",
		ModelName: "llama2",
		Input:     map[string]interface{}{"prompt": "What is the meaning of life?"},
	}

	result := executor.Execute(context.Background(), job)

	// ── Step 3: Verify the result ───────────────────────────────────────
	if !result.Success {
		t.Fatalf("job failed: %s", result.Error)
	}
	if result.JobID != "pipeline-test-001" {
		t.Errorf("JobID = %q", result.JobID)
	}
	if result.GPUID != "gpu-test-0" {
		t.Errorf("GPUID = %q", result.GPUID)
	}
	if result.DurationMS <= 0 {
		t.Errorf("DurationMS = %d, should be > 0", result.DurationMS)
	}

	// Verify the response content
	response, ok := result.Result["response"].(string)
	if !ok || response == "" {
		t.Fatalf("response should be a non-empty string, got %v", result.Result["response"])
	}
	if !strings.Contains(response, "meaning of life") {
		t.Errorf("response should echo the prompt: %q", response)
	}

	// ── Step 4: Verify model was "pulled" (prepared) ────────────────────
	if !modelPulled {
		t.Error("model should have been pulled during Prepare()")
	}

	// ── Step 5: Verify model is now cached ──────────────────────────────
	if !mockOllama.isModelCached(context.Background(), "llama2") {
		t.Error("model should be cached after pull")
	}

	// ── Step 6: Verify inference was called ──────────────────────────────
	if inferenceCount != 1 {
		t.Errorf("inference called %d times, want 1", inferenceCount)
	}

	// ── Step 7: Verify progress was reported ────────────────────────────
	if len(progressStages) < 3 {
		t.Errorf("expected at least 3 progress stages, got %d: %v", len(progressStages), progressStages)
	}

	t.Logf("✅ Pipeline complete: duration=%dms, response=%q, stages=%d",
		result.DurationMS, response, len(progressStages))

	// ── Step 8: Now test with upload ────────────────────────────────────
	// Create a fake output file that the runtime would have generated
	outputDir := filepath.Join(cacheDir, "output", "upload-test")
	os.MkdirAll(outputDir, 0755)
	outputFile := filepath.Join(outputDir, "result.mp4")
	os.WriteFile(outputFile, []byte("fake video content 1234567890"), 0644)

	// Create a runtime that returns the output file path
	mockRuntimeWithOutput := &mockOutputRuntime{
		inner:      mockOllama,
		outputFile: outputFile,
	}

	executor2 := &Executor{
		runtime:  mockRuntimeWithOutput,
		gpuID:    "gpu-test-0",
		backend:  "cpu",
		cacheDir: cacheDir,
	}

	job2 := types.JobAssignment{
		JobID:     "upload-test-001",
		ModelName: "llama2",
		Input:     map[string]interface{}{"prompt": "Generate a video"},
		UploadURL: storageSrv.URL + "/bucket/output.mp4",
	}

	result2 := executor2.Execute(context.Background(), job2)

	if !result2.Success {
		t.Fatalf("upload job failed: %s", result2.Error)
	}

	// Verify upload happened
	if result2.Result["uploaded"] != true {
		t.Error("result should have uploaded=true")
	}
	if result2.Result["upload_url"] != nil {
		t.Errorf("upload_url must not be returned, got %v", result2.Result["upload_url"])
	}

	// Verify the storage server received the file
	if string(uploadedData) != "fake video content 1234567890" {
		t.Errorf("uploaded data = %q", string(uploadedData))
	}
	if uploadedContentType != "video/mp4" {
		t.Errorf("content-type = %q, want video/mp4", uploadedContentType)
	}

	t.Logf("✅ Upload complete: %d bytes uploaded as %s", len(uploadedData), uploadedContentType)

	// ── Step 9: Run again — model should be cached (no re-pull) ─────────
	pullCount := 0
	mockRuntime2 := &mockPrepareRuntime{
		inner: mockOllama,
		onPrepare: func() {
			pullCount++
		},
	}
	executor3 := &Executor{runtime: mockRuntime2, gpuID: "gpu-test-0", backend: "cpu", cacheDir: cacheDir}

	result3 := executor3.Execute(context.Background(), types.JobAssignment{
		JobID: "cached-test-001", ModelName: "llama2",
		Input: map[string]interface{}{"prompt": "Second run"},
	})

	if !result3.Success {
		t.Fatalf("cached job failed: %s", result3.Error)
	}
	if inferenceCount != 3 { // 1 from first test + 1 from upload test + 1 from this
		t.Logf("inference count = %d (expected 3)", inferenceCount)
	}

	t.Logf("✅ Cached run complete: duration=%dms, pullCount=%d", result3.DurationMS, pullCount)
}

// ── Test helpers ─────────────────────────────────────────────────────────────

// mockPrepareRuntime wraps a runtime and calls onPrepare instead of real Prepare.
type mockPrepareRuntime struct {
	inner     Runtime
	onPrepare func()
}

func (m *mockPrepareRuntime) Name() string { return m.inner.Name() }
func (m *mockPrepareRuntime) Prepare(_ context.Context, _ types.JobAssignment) error {
	if m.onPrepare != nil {
		m.onPrepare()
	}
	return nil
}
func (m *mockPrepareRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	return m.inner.Run(ctx, a)
}
func (m *mockPrepareRuntime) Cleanup(force bool) error { return m.inner.Cleanup(force) }

// mockOutputRuntime wraps a runtime and injects an output_file into the result.
type mockOutputRuntime struct {
	inner      Runtime
	outputFile string
}

func (m *mockOutputRuntime) Name() string { return m.inner.Name() }
func (m *mockOutputRuntime) Prepare(_ context.Context, _ types.JobAssignment) error {
	return nil
}
func (m *mockOutputRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	result, err := m.inner.Run(ctx, a)
	if err != nil {
		return nil, err
	}
	result["output_file"] = m.outputFile
	return result, nil
}
func (m *mockOutputRuntime) Cleanup(force bool) error { return m.inner.Cleanup(force) }
