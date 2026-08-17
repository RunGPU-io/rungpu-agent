package job

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

// ── Unit tests for pure functions ───────────────────────────────────────────

func TestOllamaModel(t *testing.T) {
	cases := []struct {
		name string
		job  types.JobAssignment
		want string
	}{
		{"simple", types.JobAssignment{ModelName: "llama2"}, "llama2"},
		{"path style", types.JobAssignment{ModelName: "meta-llama/Llama-2-7b-hf"}, "llama-2-7b-hf"},
		{"explicit param", types.JobAssignment{ModelName: "x", Parameters: map[string]interface{}{"ollama_model": "mistral:7b"}}, "mistral:7b"},
		{"uppercase", types.JobAssignment{ModelName: "GPT4All"}, "gpt4all"},
		{"empty param fallback", types.JobAssignment{ModelName: "phi3", Parameters: map[string]interface{}{"ollama_model": ""}}, "phi3"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ollamaModel(tc.job); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestExtractPrompt(t *testing.T) {
	cases := []struct {
		name string
		job  types.JobAssignment
		want string
	}{
		{"prompt field", types.JobAssignment{Input: map[string]interface{}{"prompt": "Hello"}}, "Hello"},
		{"text field", types.JobAssignment{Input: map[string]interface{}{"text": "Sum"}}, "Sum"},
		{"prompt over text", types.JobAssignment{Input: map[string]interface{}{"prompt": "A", "text": "B"}}, "A"},
		{"nil input", types.JobAssignment{}, "null"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := extractPrompt(tc.job); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestNewRuntime(t *testing.T) {
	dir := t.TempDir()
	for _, backend := range []string{"cuda", "metal", "cpu", ""} {
		rt, err := NewRuntime(backend, dir, 10)
		if err != nil {
			t.Fatalf("NewRuntime(%q): %v", backend, err)
		}
		if rt.Name() == "" {
			t.Errorf("NewRuntime(%q).Name() should not be empty", backend)
		}
		t.Logf("NewRuntime(%q) → %q", backend, rt.Name())
	}
}

func TestIsKnownDockerModel(t *testing.T) {
	docker := []string{"ltx-video", "ltx2", "wan2", "wan2.1", "stable-diffusion", "sdxl", "flux", "flux.1", "whisper"}
	for _, m := range docker {
		if !isKnownDockerModel(m) {
			t.Errorf("%q should be a Docker model", m)
		}
	}
	ollama := []string{"llama2", "mistral", "phi3", "codellama", "gemma"}
	for _, m := range ollama {
		if isKnownDockerModel(m) {
			t.Errorf("%q should NOT be a Docker model", m)
		}
	}
	if !isKnownDockerModel("my-custom-video-gen") {
		t.Error("model with 'video' should be Docker")
	}
}

func TestIsWorkspaceJob(t *testing.T) {
	// Known workspace names
	for _, name := range []string{"comfyui", "jupyter", "a1111", "invokeai"} {
		a := types.JobAssignment{ModelName: name}
		if !isWorkspaceJob(a) {
			t.Errorf("%q should be a workspace job", name)
		}
	}
	// Workspace bool field
	if !isWorkspaceJob(types.JobAssignment{ModelName: "anything", Workspace: true}) {
		t.Error("Workspace=true field should make it a workspace job")
	}
	// Explicit workspace=true param (backward compat)
	a := types.JobAssignment{
		ModelName:  "custom-thing",
		Parameters: map[string]interface{}{"workspace": true},
	}
	if !isWorkspaceJob(a) {
		t.Error("workspace=true param should make it a workspace job")
	}
	// Regular LLM is NOT a workspace
	if isWorkspaceJob(types.JobAssignment{ModelName: "llama2"}) {
		t.Error("llama2 should not be a workspace job")
	}
}

func TestHasExplicitDockerImage(t *testing.T) {
	// DockerImage field
	if !hasExplicitDockerImage(types.JobAssignment{DockerImage: "myimg:v1"}) {
		t.Error("DockerImage field should be detected")
	}
	// Parameters docker_image
	if !hasExplicitDockerImage(types.JobAssignment{Parameters: map[string]interface{}{"docker_image": "img:v2"}}) {
		t.Error("Parameters docker_image should be detected")
	}
	// Neither
	if hasExplicitDockerImage(types.JobAssignment{ModelName: "llama2"}) {
		t.Error("llama2 should not have explicit docker image")
	}
	// Empty string
	if hasExplicitDockerImage(types.JobAssignment{DockerImage: ""}) {
		t.Error("empty DockerImage should not count")
	}
}

func TestIsComfyUIBatchJob(t *testing.T) {
	job := types.JobAssignment{Parameters: map[string]interface{}{"runtime": "comfyui-batch"}}
	if !isComfyUIBatchJob(job) {
		t.Fatal("explicit comfyui-batch runtime should be detected")
	}
	if isComfyUIBatchJob(types.JobAssignment{}) {
		t.Fatal("job without an explicit runtime should not use ComfyUI batch")
	}

	comfy := &comfyUIBatchRuntime{}
	runtime, err := (&multiRuntime{comfyUIBatch: comfy, dockerCustom: &customDockerRuntime{}}).selectRuntime(types.JobAssignment{
		ModelName: "stable-diffusion", DockerImage: "renter/image:latest",
		Parameters: map[string]interface{}{"runtime": "comfyui-batch"},
	})
	if err != nil || runtime != comfy {
		t.Fatalf("ComfyUI batch selection = %T, %v", runtime, err)
	}
	if _, err := (&multiRuntime{}).selectRuntime(job); err == nil || !strings.Contains(err.Error(), "require Docker") {
		t.Fatalf("missing ComfyUI batch runtime error = %v", err)
	}
}

func TestExplicitRuntimeSelection(t *testing.T) {
	ollama := &ollamaRuntime{}
	docker := &customDockerRuntime{}
	workspace := &workspaceRuntime{}
	comfy := &comfyUIBatchRuntime{}
	runtimes := &multiRuntime{ollama: ollama, dockerCustom: docker, workspace: workspace, comfyUIBatch: comfy}

	for runtimeName, expected := range map[string]Runtime{
		"ollama": ollama, "docker-custom": docker, "workspace": workspace, "comfyui-batch": comfy,
	} {
		selected, err := runtimes.selectRuntime(types.JobAssignment{Runtime: runtimeName, ModelName: "misleading-video-name"})
		if err != nil || selected != expected {
			t.Fatalf("runtime %q selected %T, %v", runtimeName, selected, err)
		}
	}
	if _, err := runtimes.selectRuntime(types.JobAssignment{Runtime: "unknown"}); err == nil {
		t.Fatal("unknown explicit runtime should be rejected")
	}
}

func TestInjectComfyPrompt(t *testing.T) {
	withPlaceholder := map[string]interface{}{
		"1": map[string]interface{}{"inputs": map[string]interface{}{"text": "Create {{prompt}} now"}},
	}
	if !injectComfyPrompt(withPlaceholder, "a red kite") {
		t.Fatal("placeholder workflow should accept a prompt")
	}
	text := withPlaceholder["1"].(map[string]interface{})["inputs"].(map[string]interface{})["text"]
	if text != "Create a red kite now" {
		t.Fatalf("placeholder result = %q", text)
	}

	clipWorkflow := map[string]interface{}{
		"negative": map[string]interface{}{
			"class_type": "CLIPTextEncode",
			"_meta":      map[string]interface{}{"title": "Negative Prompt"},
			"inputs":     map[string]interface{}{"text": "blurry"},
		},
		"positive": map[string]interface{}{
			"class_type": "CLIPTextEncode",
			"_meta":      map[string]interface{}{"title": "Positive Prompt"},
			"inputs":     map[string]interface{}{"text": "old prompt"},
		},
	}
	if !injectComfyPrompt(clipWorkflow, "a glass city") {
		t.Fatal("CLIPTextEncode workflow should accept a prompt")
	}
	positive := clipWorkflow["positive"].(map[string]interface{})["inputs"].(map[string]interface{})["text"]
	negative := clipWorkflow["negative"].(map[string]interface{})["inputs"].(map[string]interface{})["text"]
	if positive != "a glass city" || negative != "blurry" {
		t.Fatalf("unexpected prompt injection: positive=%q negative=%q", positive, negative)
	}
}

func TestSafeComfyOutputPath(t *testing.T) {
	root := t.TempDir()
	path, err := safeComfyOutputPath(root, "videos", "result.mp4")
	if err != nil || path != filepath.Join(root, "videos", "result.mp4") {
		t.Fatalf("safe output path = %q, %v", path, err)
	}
	if _, err := safeComfyOutputPath(root, "../../outside", "result.mp4"); err == nil {
		t.Fatal("traversal output path should be rejected")
	}
}

func TestComfyUIBatchRejectsInvalidWorkflow(t *testing.T) {
	cacheDir := t.TempDir()
	workflowDir := filepath.Join(cacheDir, "staging", "invalid-workflow", "workflows")
	if err := os.MkdirAll(workflowDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workflowDir, "workflow.json"), []byte("not-json"), 0o600); err != nil {
		t.Fatal(err)
	}
	runtime := &comfyUIBatchRuntime{cacheDir: cacheDir}
	_, err := runtime.Run(context.Background(), types.JobAssignment{
		JobID: "invalid-workflow", Parameters: map[string]interface{}{"runtime": "comfyui-batch"},
	})
	if err == nil || !strings.Contains(err.Error(), "not valid JSON") {
		t.Fatalf("invalid workflow error = %v", err)
	}
}

func TestDockerImageResolution(t *testing.T) {
	cases := []struct {
		name    string
		job     types.JobAssignment
		want    string
		wantErr bool
	}{
		{"explicit DockerImage field", types.JobAssignment{DockerImage: "myimg:v1"}, "myimg:v1", false},
		{"explicit param", types.JobAssignment{ModelName: "x", Parameters: map[string]interface{}{"docker_image": "myimg:v1"}}, "myimg:v1", false},
		{"ltx-video requires worker", types.JobAssignment{ModelName: "ltx-video"}, "", true},
		{"wan2 requires worker", types.JobAssignment{ModelName: "wan2"}, "", true},
		{"HF repo style requires worker", types.JobAssignment{ModelName: "org/model"}, "", true},
		{"unknown no slash", types.JobAssignment{ModelName: "llama2"}, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveImage(tc.job)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestResolveWorkspace(t *testing.T) {
	// ComfyUI
	img, ports, shm, vols, _ := resolveWorkspace(types.JobAssignment{ModelName: "comfyui"})
	if img == "" {
		t.Error("comfyui should resolve to an image")
	}
	if len(ports) == 0 {
		t.Error("comfyui should have ports")
	}
	if shm == "" {
		t.Error("comfyui should have shm_size")
	}
	if len(vols) == 0 {
		t.Error("comfyui should have persistent volumes for model storage")
	}
	// Verify the models volume exists (most important for caching)
	if _, ok := vols["tokenize-comfyui-models"]; !ok {
		t.Error("comfyui should have a tokenize-comfyui-models volume")
	}

	// Explicit override
	img2, ports2, _, _, _ := resolveWorkspace(types.JobAssignment{
		ModelName:  "custom",
		Parameters: map[string]interface{}{"docker_image": "myimg:v1", "ports": "9090"},
	})
	if img2 != "myimg:v1" {
		t.Errorf("image = %q, want myimg:v1", img2)
	}
	if len(ports2) != 1 || ports2[0] != "9090:9090" {
		t.Errorf("ports = %v, want [9090:9090]", ports2)
	}
	img3, ports3, _, _, _ := resolveWorkspace(types.JobAssignment{
		ModelName: "custom", DockerImage: "top-level:v2", Ports: []string{"8188"},
	})
	if img3 != "top-level:v2" || len(ports3) != 1 || ports3[0] != "8188:8188" {
		t.Errorf("top-level workspace fields ignored: image=%q ports=%v", img3, ports3)
	}
}

func TestModelID(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"llama2", "llama2"},
		{"meta-llama/Llama-2-7b", "meta-llama_Llama-2-7b"},
		{"org/sub/model", "org_sub_model"},
	} {
		if got := modelID(tc.in); got != tc.want {
			t.Errorf("modelID(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// ── Mock Ollama integration test ────────────────────────────────────────────

func TestOllamaRuntimeWithMockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)
		model, _ := req["model"].(string)
		prompt, _ := req["prompt"].(string)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"model": model, "response": "I am " + model + ". You said: " + prompt,
			"done": true, "eval_count": 42,
		})
	}))
	defer srv.Close()

	rt := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 10 * time.Second},
	}

	result, err := rt.Run(context.Background(), types.JobAssignment{
		JobID: "m1", ModelName: "llama2",
		Input: map[string]interface{}{"prompt": "What is 2+2?"},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result["status"] != "completed" {
		t.Errorf("status = %v", result["status"])
	}
	if result["model"] != "llama2" {
		t.Errorf("model = %v", result["model"])
	}
	resp := result["response"].(string)
	if !strings.Contains(resp, "llama2") {
		t.Errorf("response should mention model: %q", resp)
	}
	// eval_count may be int or float64 depending on Go version's JSON decoder
	switch v := result["eval_count"].(type) {
	case float64:
		if v != 42 {
			t.Errorf("eval_count = %v", v)
		}
	case int:
		if v != 42 {
			t.Errorf("eval_count = %v", v)
		}
	default:
		t.Errorf("eval_count unexpected type %T = %v", result["eval_count"], result["eval_count"])
	}
	t.Logf("Response: %s", resp)
}

func TestOllamaRuntimeServerError(t *testing.T) {
	// Server that returns 500 for /api/generate but 200 for health check (GET /)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			// Health check — server is "running" but broken
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("Ollama is running"))
			return
		}
		http.Error(w, "boom", 500)
	}))
	defer srv.Close()

	rt := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 5 * time.Second},
	}
	_, err := rt.Run(context.Background(), types.JobAssignment{
		JobID: "e1", ModelName: "llama2", Input: map[string]interface{}{"prompt": "x"},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should mention 500: %v", err)
	}
}

func TestOllamaRuntimeServerDown(t *testing.T) {
	rt := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: "http://127.0.0.1:1",
		backend: "cpu", client: &http.Client{Timeout: 2 * time.Second},
	}
	_, err := rt.Run(context.Background(), types.JobAssignment{
		JobID: "d1", ModelName: "llama2", Input: map[string]interface{}{"prompt": "x"},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	// Non-default endpoint: ensureServerRunning returns "not responding" error
	if !strings.Contains(err.Error(), "not responding") {
		t.Errorf("error should mention 'not responding': %v", err)
	}
}

// ── Full executor end-to-end with mock ──────────────────────────────────────

// skipPrepare wraps a runtime and skips Prepare (no real ollama in tests).
type skipPrepare struct{ inner Runtime }

func (s *skipPrepare) Name() string                                           { return s.inner.Name() }
func (s *skipPrepare) Prepare(_ context.Context, _ types.JobAssignment) error { return nil }
func (s *skipPrepare) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	return s.inner.Run(ctx, a)
}
func (s *skipPrepare) Cleanup(force bool) error { return s.inner.Cleanup(force) }

func TestExecutorEndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"model": req["model"], "response": "The answer is 4.",
			"done": true, "eval_count": 10,
		})
	}))
	defer srv.Close()

	mockRT := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 10 * time.Second},
	}
	exec := &Executor{runtime: &skipPrepare{inner: mockRT}, gpuID: "gpu-e2e", backend: "cpu"}

	result := exec.Execute(context.Background(), types.JobAssignment{
		JobID: "e2e-1", ModelName: "llama2",
		Input: map[string]interface{}{"prompt": "What is 2+2?"},
	})

	if !result.Success {
		t.Fatalf("failed: %s", result.Error)
	}
	if result.JobID != "e2e-1" {
		t.Errorf("JobID = %q", result.JobID)
	}
	if result.GPUID != "gpu-e2e" {
		t.Errorf("GPUID = %q", result.GPUID)
	}
	if result.Result["response"] != "The answer is 4." {
		t.Errorf("response = %v", result.Result["response"])
	}
	t.Logf("E2E: success=%v duration=%dms response=%v", result.Success, result.DurationMS, result.Result["response"])
}

func TestExecutorHandlesFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", 500)
	}))
	defer srv.Close()

	mockRT := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 5 * time.Second},
	}
	exec := &Executor{runtime: &skipPrepare{inner: mockRT}, gpuID: "gpu-fail", backend: "cpu"}

	result := exec.Execute(context.Background(), types.JobAssignment{
		JobID: "fail-1", ModelName: "llama2",
		Input: map[string]interface{}{"prompt": "x"},
	})

	if result.Success {
		t.Fatal("expected failure")
	}
	if result.Error == "" {
		t.Error("Error should not be empty")
	}
	if result.JobID != "fail-1" {
		t.Errorf("JobID = %q", result.JobID)
	}
}

// ── Cache behavior tests ────────────────────────────────────────────────────

func TestOllamaCacheHit(t *testing.T) {
	// Mock: /api/show returns 200 (model cached)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			json.NewEncoder(w).Encode(map[string]interface{}{"modelfile": "FROM llama2"})
			return
		}
		if r.URL.Path == "/api/generate" {
			json.NewEncoder(w).Encode(map[string]interface{}{
				"model": "llama2", "response": "cached!", "done": true, "eval_count": 1,
			})
			return
		}
	}))
	defer srv.Close()

	rt := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 5 * time.Second},
	}

	if !rt.isModelCached(context.Background(), "llama2") {
		t.Fatal("should report cached when /api/show returns 200")
	}

	res, err := rt.Run(context.Background(), types.JobAssignment{
		JobID: "ch-1", ModelName: "llama2",
		Input: map[string]interface{}{"prompt": "test"},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res["response"] != "cached!" {
		t.Errorf("response = %v", res["response"])
	}
}

func TestOllamaCacheMiss(t *testing.T) {
	// Mock: /api/show returns 404 (model not cached)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			http.Error(w, "not found", 404)
			return
		}
	}))
	defer srv.Close()

	rt := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 5 * time.Second},
	}

	if rt.isModelCached(context.Background(), "nonexistent") {
		t.Error("should report NOT cached when /api/show returns 404")
	}
}

func TestDockerCacheSkipsDownload(t *testing.T) {
	downloads := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		downloads++
		w.Write([]byte("fake model data"))
	}))
	defer srv.Close()

	cacheDir := t.TempDir()
	dr := &dockerRuntime{cacheDir: cacheDir, maxCacheGB: 10}
	job := types.JobAssignment{
		JobID: "dc-1", ModelName: "org/model", ModelURL: srv.URL + "/m.bin",
	}

	// First: downloads
	if err := dr.Prepare(context.Background(), job); err != nil {
		t.Fatalf("Prepare 1: %v", err)
	}
	if downloads != 1 {
		t.Errorf("expected 1 download, got %d", downloads)
	}

	// Verify cached
	cached := filepath.Join(cacheDir, modelID("org/model"))
	if info, err := os.Stat(cached); err != nil || info.Size() == 0 {
		t.Fatal("model should be cached on disk")
	}

	// Second: skips download
	if err := dr.Prepare(context.Background(), job); err != nil {
		t.Fatalf("Prepare 2: %v", err)
	}
	if downloads != 1 {
		t.Errorf("expected still 1 download (cached), got %d", downloads)
	}
}

func TestDockerCleanup(t *testing.T) {
	dir := t.TempDir()
	dr := &dockerRuntime{cacheDir: dir, maxCacheGB: 10}
	os.WriteFile(filepath.Join(dir, "model-a"), []byte("data"), 0644)

	dr.Cleanup(false)
	if _, err := os.Stat(filepath.Join(dir, "model-a")); err != nil {
		t.Error("Cleanup(false) should not remove files")
	}

	dr.Cleanup(true)
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Errorf("Cleanup(true) should empty cache, got %d entries", len(entries))
	}
}

func TestFullPipelineCacheAndRun(t *testing.T) {
	generates := 0
	showCached := false

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			if showCached {
				json.NewEncoder(w).Encode(map[string]interface{}{"modelfile": "ok"})
			} else {
				http.Error(w, "not found", 404)
			}
		case "/api/generate":
			generates++
			var req map[string]interface{}
			json.NewDecoder(r.Body).Decode(&req)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"model": req["model"], "response": "Answer: 42",
				"done": true, "eval_count": 5,
			})
		}
	}))
	defer srv.Close()

	rt := &ollamaRuntime{
		cacheDir: t.TempDir(), endpoint: srv.URL,
		backend: "cpu", client: &http.Client{Timeout: 10 * time.Second},
	}
	job := types.JobAssignment{
		JobID: "fp-1", ModelName: "llama2",
		Input: map[string]interface{}{"prompt": "meaning of life?"},
	}

	// 1. Not cached
	if rt.isModelCached(context.Background(), "llama2") {
		t.Fatal("should not be cached yet")
	}

	// 2. Simulate pull completed
	showCached = true

	// 3. Now cached
	if !rt.isModelCached(context.Background(), "llama2") {
		t.Fatal("should be cached after pull")
	}

	// 4. Run inference
	res, err := rt.Run(context.Background(), job)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res["response"] != "Answer: 42" {
		t.Errorf("response = %v", res["response"])
	}

	// 5. Run again — still cached
	res2, _ := rt.Run(context.Background(), job)
	if res2["response"] != "Answer: 42" {
		t.Errorf("response 2 = %v", res2["response"])
	}
	if generates != 2 {
		t.Errorf("expected 2 generates, got %d", generates)
	}

	t.Logf("Pipeline: generates=%d, response=%v", generates, res["response"])
}
