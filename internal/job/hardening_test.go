package job

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tokenize/gpu-agent/internal/dockermgr"
	"github.com/tokenize/gpu-agent/internal/types"
)

func TestBindPortsLoopback(t *testing.T) {
	got := bindPortsLoopback([]string{"8188:8188", "7860", "0.0.0.0:9000:9000", "  "})
	want := []string{
		"127.0.0.1:8188:8188",
		"127.0.0.1:7860:7860",
		"127.0.0.1:9000:9000",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("bindPortsLoopback = %v, want %v", got, want)
	}
}

func TestHostPortOf(t *testing.T) {
	cases := map[string]string{
		"8188":                "8188",
		"8188:8188":           "8188",
		"127.0.0.1:8188:8188": "8188",
		"0.0.0.0:9000:9000":   "9000",
	}
	for in, want := range cases {
		if got := hostPortOf(in); got != want {
			t.Errorf("hostPortOf(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestJobTimeoutFor(t *testing.T) {
	r := newCustomDockerRuntime(t.TempDir(), false, "", 0, dockermgr.DefaultPolicy())

	if d := r.jobTimeoutFor(types.JobAssignment{}); d != 60*time.Minute {
		t.Errorf("default timeout = %v, want 60m", d)
	}
	if d := r.jobTimeoutFor(types.JobAssignment{
		Parameters: map[string]interface{}{"timeout_minutes": float64(5)},
	}); d != 5*time.Minute {
		t.Errorf("float override = %v, want 5m", d)
	}
	if d := r.jobTimeoutFor(types.JobAssignment{
		Parameters: map[string]interface{}{"timeout_minutes": "10"},
	}); d != 10*time.Minute {
		t.Errorf("string override = %v, want 10m", d)
	}
	if d := r.jobTimeoutFor(types.JobAssignment{
		Parameters: map[string]interface{}{"timeout_minutes": float64(120)},
	}); d != 60*time.Minute {
		t.Errorf("renter timeout above host cap = %v, want 60m", d)
	}

	// A custom construction timeout is respected when no per-job override.
	r2 := newCustomDockerRuntime(t.TempDir(), false, "", 90*time.Minute, dockermgr.DefaultPolicy())
	if d := r2.jobTimeoutFor(types.JobAssignment{}); d != 90*time.Minute {
		t.Errorf("configured timeout = %v, want 90m", d)
	}
}

func TestValidateCustomFileRejectsAbsolutePath(t *testing.T) {
	if err := ValidateCustomFileURL("https://huggingface.co/model.safetensors", "/etc/passwd"); err == nil {
		t.Fatal("expected absolute custom file path to be rejected")
	}
}

func TestDownloadFileSizeCap(t *testing.T) {
	defer SetMaxDownloadBytes(0) // reset global cap for other tests

	body := make([]byte, 500)

	// Server that advertises Content-Length (default behavior).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	defer srv.Close()

	// Server that streams without a known length (chunked).
	streamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fl, _ := w.(http.Flusher)
		for i := 0; i < 5; i++ {
			w.Write(make([]byte, 100))
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	defer streamSrv.Close()

	dir := t.TempDir()

	// Over the cap via Content-Length → error, no file left behind.
	SetMaxDownloadBytes(100)
	dest := filepath.Join(dir, "over.bin")
	if err := downloadFile(context.Background(), srv.URL, dest); err == nil {
		t.Error("expected error for file exceeding cap (content-length)")
	}

	// Over the cap via streamed bytes (unknown length) → error.
	destStream := filepath.Join(dir, "stream.bin")
	if err := downloadFile(context.Background(), streamSrv.URL, destStream); err == nil {
		t.Error("expected error for streamed file exceeding cap")
	}
	if _, err := os.Stat(destStream); err == nil {
		t.Error("over-cap streamed file should be removed")
	}

	// Under the cap → success, full content written.
	SetMaxDownloadBytes(1000)
	okDest := filepath.Join(dir, "ok.bin")
	if err := downloadFile(context.Background(), srv.URL, okDest); err != nil {
		t.Fatalf("under-cap download failed: %v", err)
	}
	if info, err := os.Stat(okDest); err != nil || info.Size() != 500 {
		t.Errorf("expected 500-byte file, got size=%v err=%v", info.Size(), err)
	}
}

func TestHFTokenFallback(t *testing.T) {
	defer SetHFToken("")
	os.Unsetenv("HF_TOKEN")

	// Configured token used when env is unset.
	SetHFToken("hf_from_config")
	if got := hfToken(); got != "hf_from_config" {
		t.Errorf("expected configured token, got %q", got)
	}

	// Env var takes precedence over config.
	os.Setenv("HF_TOKEN", "hf_from_env")
	defer os.Unsetenv("HF_TOKEN")
	if got := hfToken(); got != "hf_from_env" {
		t.Errorf("env token should win, got %q", got)
	}
}

// blockingRuntime blocks in Run until the context is cancelled, so we can
// verify Executor.Cancel interrupts an in-flight job.
type blockingRuntime struct{ started chan struct{} }

func (b *blockingRuntime) Name() string                                             { return "blocking" }
func (b *blockingRuntime) Prepare(ctx context.Context, a types.JobAssignment) error { return nil }
func (b *blockingRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	close(b.started)
	<-ctx.Done()
	return nil, ctx.Err()
}
func (b *blockingRuntime) Cleanup(force bool) error { return nil }

func TestPruneBatchStagingPreservesWorkspaces(t *testing.T) {
	cacheDir := t.TempDir()
	for _, jobID := range []string{"batch-job", "workspace-job"} {
		path := filepath.Join(cacheDir, "staging", jobID, "asset.safetensors")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("asset"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := PruneBatchStaging(cacheDir, map[string]bool{"workspace-job": true}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(cacheDir, "staging", "batch-job")); !os.IsNotExist(err) {
		t.Fatalf("batch staging still exists: %v", err)
	}
	if _, err := os.Stat(filepath.Join(cacheDir, "staging", "workspace-job")); err != nil {
		t.Fatalf("workspace staging was removed: %v", err)
	}
}

func TestExecutorCancelInterruptsRunningJob(t *testing.T) {
	rt := &blockingRuntime{started: make(chan struct{})}
	e := &Executor{runtime: rt, gpuID: "gpu-cancel", backend: "cpu", cacheDir: t.TempDir()}

	done := make(chan types.JobResult, 1)
	go func() {
		done <- e.Execute(context.Background(), types.JobAssignment{JobID: "cx", ModelName: "m"})
	}()

	select {
	case <-rt.started:
	case <-time.After(5 * time.Second):
		t.Fatal("job never started")
	}

	e.Cancel("cx")

	select {
	case res := <-done:
		if res.Success {
			t.Error("cancelled job should not report success")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Cancel did not interrupt the running job")
	}
}

func TestExecutorEnforcesBatchTimeout(t *testing.T) {
	runtime := &blockingRuntime{started: make(chan struct{})}
	executor := &Executor{
		runtime: runtime, gpuID: "gpu-timeout", backend: "cpu", cacheDir: t.TempDir(),
		jobTimeout: 20 * time.Millisecond, inflight: map[string]context.CancelFunc{}, workspaces: map[string]bool{},
	}
	result := executor.Execute(context.Background(), types.JobAssignment{JobID: "timeout-job"})
	if result.Success || !strings.Contains(result.Error, context.DeadlineExceeded.Error()) {
		t.Fatalf("timeout result: success=%v error=%q", result.Success, result.Error)
	}
}

func TestExecutorTrackingAndCancel(t *testing.T) {
	// Cancel/StopAll must be safe on unknown ids and must clear tracking state.
	e := &Executor{gpuID: "gpu-track", backend: "cpu"}

	e.trackInflight("job-a", func() {})
	e.markWorkspace("job-b")

	e.mu.Lock()
	if len(e.inflight) != 1 || len(e.workspaces) != 1 {
		e.mu.Unlock()
		t.Fatal("tracking maps not populated")
	}
	e.mu.Unlock()

	// Cancel a tracked job (docker teardown is best-effort / no-op without docker).
	e.Cancel("job-a")
	e.Cancel("job-b")
	e.Cancel("does-not-exist") // must not panic

	e.mu.Lock()
	defer e.mu.Unlock()
	if _, ok := e.inflight["job-a"]; ok {
		t.Error("job-a should be removed from inflight after cancel")
	}
	if e.workspaces["job-b"] {
		t.Error("job-b should be removed from workspaces after cancel")
	}
}
