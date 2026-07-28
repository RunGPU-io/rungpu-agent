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

	"github.com/RunGPU-io/gpu-agent/internal/types"
)

func TestResolveImage(t *testing.T) {
	cases := []struct {
		name    string
		job     types.JobAssignment
		want    string
		wantErr bool
	}{
		{
			"explicit DockerImage field",
			types.JobAssignment{DockerImage: "myregistry/mymodel:v1"},
			"myregistry/mymodel:v1", false,
		},
		{
			"docker_image in parameters",
			types.JobAssignment{Parameters: map[string]interface{}{"docker_image": "img:v2"}},
			"img:v2", false,
		},
		{
			"well-known ltx-video routes to ComfyUI",
			types.JobAssignment{ModelName: "ltx-video"},
			"ghcr.io/ai-dock/comfyui:latest", false,
		},
		{
			"well-known wan2 routes to ComfyUI",
			types.JobAssignment{ModelName: "wan2"},
			"ghcr.io/ai-dock/comfyui:latest", false,
		},
		{
			"well-known comfyui",
			types.JobAssignment{ModelName: "comfyui"},
			"ghcr.io/ai-dock/comfyui:latest", false,
		},
		{
			"HuggingFace URL",
			types.JobAssignment{ModelName: "test", ModelURL: "https://huggingface.co/spaces/org/model"},
			"registry.hf.space/org-model", false,
		},
		{
			"HuggingFace repo style",
			types.JobAssignment{ModelName: "stabilityai/stable-diffusion-xl"},
			"registry.hf.space/stabilityai-stable-diffusion-xl", false,
		},
		{
			"unknown model no slash",
			types.JobAssignment{ModelName: "unknownmodel"},
			"", true,
		},
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

func TestValidateCustomFileURL(t *testing.T) {
	// Trusted sources should pass
	trusted := []struct{ url, path string }{
		{"https://huggingface.co/user/model/resolve/main/lora.safetensors", "models/loras/lora.safetensors"},
		{"https://civitai.com/api/download/models/123", "models/loras/anime.safetensors"},
		{"https://raw.githubusercontent.com/user/repo/main/workflow.json", "workflows/wf.json"},
		{"https://storage.googleapis.com/bucket/model.bin", "models/model.bin"},
		{"https://cdn-lfs.huggingface.co/repos/abc/model.safetensors", "models/m.safetensors"},
	}
	for _, tc := range trusted {
		if err := ValidateCustomFileURL(tc.url, tc.path); err != nil {
			t.Errorf("trusted URL %q rejected: %v", tc.url, err)
		}
	}

	// Untrusted sources should fail
	untrusted := []struct{ url, path string }{
		{"https://evil.io/malware.safetensors", "models/m.safetensors"},
		{"https://randomsite.com/lora.safetensors", "models/lora.safetensors"},
		{"http://huggingface.co/model.safetensors", "models/m.safetensors"},  // HTTP not HTTPS
		{"ftp://huggingface.co/model.safetensors", "models/m.safetensors"},   // FTP
	}
	for _, tc := range untrusted {
		if err := ValidateCustomFileURL(tc.url, tc.path); err == nil {
			t.Errorf("untrusted URL %q should be rejected", tc.url)
		}
	}

	// Unsafe extensions should fail
	if err := ValidateCustomFileURL("https://huggingface.co/file.exe", "models/file.exe"); err == nil {
		t.Error(".exe should be rejected")
	}
	if err := ValidateCustomFileURL("https://huggingface.co/file.sh", "scripts/file.sh"); err == nil {
		t.Error(".sh should be rejected")
	}

	// Path traversal should fail
	if err := ValidateCustomFileURL("https://huggingface.co/lora.safetensors", "../../etc/passwd"); err == nil {
		t.Error("path traversal should be rejected")
	}
}

func TestDownloadCustomFilesDirectly(t *testing.T) {
	// Test downloadFile directly (bypasses URL validation which is tested above)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/lora.safetensors":
			w.Write([]byte("fake lora weights"))
		case "/workflow.json":
			w.Write([]byte(`{"nodes": []}`))
		default:
			http.Error(w, "not found", 404)
		}
	}))
	defer srv.Close()

	stagingDir := t.TempDir()

	// Download LoRA
	loraPath := filepath.Join(stagingDir, "models/loras/my_lora.safetensors")
	os.MkdirAll(filepath.Dir(loraPath), 0755)
	if err := downloadFile(context.Background(), srv.URL+"/lora.safetensors", loraPath); err != nil {
		t.Fatalf("download lora: %v", err)
	}
	if data, err := os.ReadFile(loraPath); err != nil {
		t.Errorf("lora missing: %v", err)
	} else if string(data) != "fake lora weights" {
		t.Errorf("lora content = %q", string(data))
	}

	// Download workflow
	wfPath := filepath.Join(stagingDir, "workflows/my_workflow.json")
	os.MkdirAll(filepath.Dir(wfPath), 0755)
	if err := downloadFile(context.Background(), srv.URL+"/workflow.json", wfPath); err != nil {
		t.Fatalf("download workflow: %v", err)
	}
	if data, err := os.ReadFile(wfPath); err != nil {
		t.Errorf("workflow missing: %v", err)
	} else if string(data) != `{"nodes": []}` {
		t.Errorf("workflow content = %q", string(data))
	}

	// Second download of same file — should overwrite (downloadFile doesn't cache)
	if err := downloadFile(context.Background(), srv.URL+"/lora.safetensors", loraPath); err != nil {
		t.Fatalf("re-download: %v", err)
	}
}

func TestDownloadCustomFilesEmpty(t *testing.T) {
	err := DownloadCustomFiles(context.Background(), nil, t.TempDir(), nil)
	if err != nil {
		t.Errorf("empty files should not error: %v", err)
	}
}

func TestUploadOutput(t *testing.T) {
	var receivedBody []byte
	var receivedContentType string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			http.Error(w, "want PUT", 405)
			return
		}
		receivedContentType = r.Header.Get("Content-Type")
		body, _ := os.ReadFile(r.URL.Path) // won't work, but we read from request body
		_ = body
		buf := make([]byte, 1024)
		n, _ := r.Body.Read(buf)
		receivedBody = buf[:n]
		w.WriteHeader(200)
	}))
	defer srv.Close()

	// Create a fake output file
	tmpFile := filepath.Join(t.TempDir(), "output.mp4")
	os.WriteFile(tmpFile, []byte("fake video data"), 0644)

	err := UploadOutput(context.Background(), tmpFile, srv.URL+"/upload")
	if err != nil {
		t.Fatalf("UploadOutput: %v", err)
	}

	if string(receivedBody) != "fake video data" {
		t.Errorf("received body = %q", string(receivedBody))
	}
	if receivedContentType != "video/mp4" {
		t.Errorf("content-type = %q, want video/mp4", receivedContentType)
	}
}

func TestUploadOutputEmptyURL(t *testing.T) {
	// Should be a no-op
	err := UploadOutput(context.Background(), "/nonexistent", "")
	if err != nil {
		t.Errorf("empty URL should be no-op: %v", err)
	}
}

func TestBuildMounts(t *testing.T) {
	mounts := BuildMounts("/cache", "/staging/job1", []types.CustomFile{
		{URL: "x", Path: "models/lora.safetensors"},
	})

	if len(mounts) != 2 {
		t.Fatalf("expected 2 mounts, got %d: %v", len(mounts), mounts)
	}
	if mounts[0] != "/cache:/cache" {
		t.Errorf("mount[0] = %q", mounts[0])
	}
	if mounts[1] != "/staging/job1:/custom:ro" {
		t.Errorf("mount[1] = %q", mounts[1])
	}

	// No custom files → only cache mount
	mounts2 := BuildMounts("/cache", "", nil)
	if len(mounts2) != 1 {
		t.Errorf("expected 1 mount without custom files, got %d", len(mounts2))
	}
}

func TestFindOutputFile(t *testing.T) {
	dir := t.TempDir()

	// Empty dir
	if f := findOutputFile(dir); f != "" {
		t.Errorf("empty dir should return empty, got %q", f)
	}

	// Non-media file
	os.WriteFile(filepath.Join(dir, "log.txt"), []byte("log"), 0644)
	if f := findOutputFile(dir); f != filepath.Join(dir, "log.txt") {
		// Should return first file as fallback
		t.Logf("fallback file: %q", f)
	}

	// Media file takes priority
	os.WriteFile(filepath.Join(dir, "output.mp4"), []byte("video"), 0644)
	f := findOutputFile(dir)
	if f != filepath.Join(dir, "output.mp4") {
		t.Errorf("should find mp4, got %q", f)
	}

	// PNG also works
	dir2 := t.TempDir()
	os.WriteFile(filepath.Join(dir2, "result.png"), []byte("image"), 0644)
	if f := findOutputFile(dir2); f != filepath.Join(dir2, "result.png") {
		t.Errorf("should find png, got %q", f)
	}
}

func TestDownloadFileWithHFToken(t *testing.T) {
	var receivedAuth string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Write([]byte("model weights"))
	}))
	defer srv.Close()

	// Set HF_TOKEN env var
	os.Setenv("HF_TOKEN", "hf_test_token_123")
	defer os.Unsetenv("HF_TOKEN")

	dest := filepath.Join(t.TempDir(), "model.safetensors")

	// Download from a URL that contains "huggingface.co" — should add auth
	// We fake the URL by using our test server but including "huggingface.co" in the path
	// Since downloadFile checks the URL string, we need to test the logic directly
	err := downloadFile(context.Background(), srv.URL+"/huggingface.co/model.safetensors", dest)
	if err != nil {
		t.Fatalf("download error: %v", err)
	}

	if receivedAuth != "Bearer hf_test_token_123" {
		t.Errorf("Authorization = %q, want 'Bearer hf_test_token_123'", receivedAuth)
	}

	data, _ := os.ReadFile(dest)
	if string(data) != "model weights" {
		t.Errorf("content = %q", string(data))
	}
}

func TestDownloadFileNoTokenForNonHF(t *testing.T) {
	var receivedAuth string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Write([]byte("data"))
	}))
	defer srv.Close()

	os.Setenv("HF_TOKEN", "hf_should_not_be_sent")
	defer os.Unsetenv("HF_TOKEN")

	dest := filepath.Join(t.TempDir(), "file.bin")
	err := downloadFile(context.Background(), srv.URL+"/civitai/model.safetensors", dest)
	if err != nil {
		t.Fatalf("download error: %v", err)
	}

	// Non-HF URLs should NOT get the token
	if receivedAuth != "" {
		t.Errorf("non-HF URL should not get auth header, got %q", receivedAuth)
	}
}

func TestDownloadFileGatedModelError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Access denied", 403)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "gated.safetensors")
	err := downloadFile(context.Background(), srv.URL+"/huggingface.co/gated-model", dest)

	if err == nil {
		t.Fatal("expected error for 403")
	}
	if !strings.Contains(err.Error(), "gated") {
		t.Errorf("error should mention 'gated': %v", err)
	}
	if !strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("error should mention HF_TOKEN: %v", err)
	}
}

func TestExecutorRejectsUntrustedCustomFileURL(t *testing.T) {
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"model": "llama2", "response": "done", "done": true, "eval_count": 1,
		})
	}))
	defer ollamaSrv.Close()

	cacheDir := t.TempDir()
	mockRT := &ollamaRuntime{
		cacheDir: cacheDir, endpoint: ollamaSrv.URL,
		backend: "cpu", client: &http.Client{Timeout: 10 * time.Second},
	}
	exec := &Executor{
		runtime:  &skipPrepare{inner: mockRT},
		gpuID:    "gpu-sec",
		backend:  "cpu",
		cacheDir: cacheDir,
	}

	// Try to download from an untrusted source — should be rejected
	result := exec.Execute(context.Background(), types.JobAssignment{
		JobID:     "sec-1",
		ModelName: "llama2",
		Input:     map[string]interface{}{"prompt": "test"},
		CustomFiles: []types.CustomFile{
			{URL: "https://evil.io/malware.safetensors", Path: "models/loras/bad.safetensors", Name: "Bad LoRA"},
		},
	})

	if result.Success {
		t.Fatal("should have failed — untrusted URL")
	}
	if !strings.Contains(result.Error, "not from a trusted source") {
		t.Errorf("error should mention trusted source: %s", result.Error)
	}
}

func TestExecutorRejectsPathTraversal(t *testing.T) {
	cacheDir := t.TempDir()
	exec := &Executor{
		runtime:  &skipPrepare{inner: &ollamaRuntime{cacheDir: cacheDir, endpoint: "http://localhost:1", backend: "cpu", client: &http.Client{}}},
		gpuID:    "gpu-pt",
		backend:  "cpu",
		cacheDir: cacheDir,
	}

	result := exec.Execute(context.Background(), types.JobAssignment{
		JobID:     "pt-1",
		ModelName: "llama2",
		Input:     map[string]interface{}{"prompt": "test"},
		CustomFiles: []types.CustomFile{
			{URL: "https://huggingface.co/lora.safetensors", Path: "../../etc/passwd", Name: "Traversal"},
		},
	})

	if result.Success {
		t.Fatal("should have failed — path traversal")
	}
	if !strings.Contains(result.Error, "path traversal") {
		t.Errorf("error should mention path traversal: %s", result.Error)
	}
}
