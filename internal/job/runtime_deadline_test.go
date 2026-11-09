package job

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type deadlineTransport func(*http.Request) (*http.Response, error)

func (f deadlineTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func deadlineOllama(t *testing.T, generate func(*http.Request) (*http.Response, error)) *ollamaRuntime {
	t.Helper()
	rt := newOllamaRuntime(t.TempDir(), "cpu")
	rt.endpoint = "http://ollama.fixture"
	rt.client.Transport = deadlineTransport(func(req *http.Request) (*http.Response, error) {
		if req.URL.Path == "/api/generate" {
			return generate(req)
		}
		return &http.Response{
			StatusCode: http.StatusOK, Header: make(http.Header),
			Body: io.NopCloser(strings.NewReader("{}")), Request: req,
		}, nil
	})
	return rt
}

func TestOllamaJobDeadline(t *testing.T) {
	t.Run("generation can exceed five minutes", func(t *testing.T) {
		rt := deadlineOllama(t, func(req *http.Request) (*http.Response, error) {
			deadline, ok := req.Context().Deadline()
			if !ok || time.Until(deadline) <= 5*time.Minute || time.Until(deadline) > 10*time.Minute {
				return nil, fmt.Errorf("generation request did not inherit ten-minute job deadline: %v", deadline)
			}
			return &http.Response{
				StatusCode: http.StatusOK, Header: make(http.Header),
				Body: io.NopCloser(strings.NewReader(`{"done":true,"response":"complete"}`)), Request: req,
			}, nil
		})
		gate := &sync.RWMutex{}
		e := &Executor{cacheDir: t.TempDir(), runtime: &skipPrepare{inner: rt}, executionGate: gate, jobTimeout: 10 * time.Minute}
		result := e.Execute(context.Background(), types.JobAssignment{JobID: "long-native", Runtime: "ollama", ModelName: "fixture"})
		if !result.Success || result.Result["response"] != "complete" {
			t.Fatalf("long-deadline generation failed: %+v", result)
		}
		if !gate.TryLock() {
			t.Fatal("confirmed completion did not release machine reservation")
		}
		gate.Unlock()
	})

	t.Run("job deadline still cancels and retains native reservation", func(t *testing.T) {
		rt := deadlineOllama(t, func(req *http.Request) (*http.Response, error) {
			<-req.Context().Done()
			return nil, req.Context().Err()
		})
		gate := &sync.RWMutex{}
		e := &Executor{cacheDir: t.TempDir(), runtime: &skipPrepare{inner: rt}, executionGate: gate, jobTimeout: 30 * time.Millisecond}
		t.Cleanup(func() {
			e.mu.Lock()
			if state := e.executions["expired-native"]; state != nil {
				e.releaseExecution(state)
			}
			e.mu.Unlock()
		})
		result := e.Execute(context.Background(), types.JobAssignment{JobID: "expired-native", Runtime: "ollama", ModelName: "fixture"})
		if result.Success || !strings.Contains(result.Error, "context deadline exceeded") || !strings.Contains(result.Error, "machine remains reserved") {
			t.Fatalf("job deadline was not safely enforced: %+v", result)
		}
		e.ForgetExecution("expired-native")
		if !e.IsTracked("expired-native") {
			t.Fatal("unverified native execution was forgotten")
		}
		sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
		rejectAdmission(t, sibling, "after-native-timeout", "docker-custom")
	})

	t.Run("metadata probes keep their own bound", func(t *testing.T) {
		rt := newOllamaRuntime(t.TempDir(), "cpu")
		rt.endpoint = "http://ollama.fixture"
		rt.client.Transport = deadlineTransport(func(req *http.Request) (*http.Response, error) {
			deadline, ok := req.Context().Deadline()
			maximum := 2 * time.Second
			if req.URL.Path == "/api/show" {
				maximum = 5 * time.Minute
			}
			if !ok || time.Until(deadline) > maximum {
				return nil, fmt.Errorf("unbounded metadata request: %s", req.URL.Path)
			}
			return &http.Response{
				StatusCode: http.StatusOK, Header: make(http.Header),
				Body: io.NopCloser(strings.NewReader("{}")), Request: req,
			}, nil
		})
		if !rt.isServerRunning() || !rt.isModelCached(context.Background(), "fixture") {
			t.Fatal("bounded metadata probes failed")
		}
	})
}

func deadlineCacheDir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), ".tokenize", "cache")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	return dir
}

func deadlineDockerFixture(t *testing.T) (started, cleanup, release string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake Docker executable uses a POSIX shell")
	}
	dir := t.TempDir()
	started, cleanup, release = filepath.Join(dir, "started"), filepath.Join(dir, "cleanup"), filepath.Join(dir, "release")
	t.Setenv("DEADLINE_STARTED", started)
	t.Setenv("DEADLINE_CLEANUP", cleanup)
	t.Setenv("DEADLINE_RELEASE", release)
	t.Setenv("DEADLINE_RUNNING", "true")
	t.Setenv("DEADLINE_HANG", "true")
	script := `#!/bin/sh
case "$1" in
info) printf 'fixture-daemon\n';;
run) printf 'fixture-container\n';;
inspect)
  printf 'started\n' > "$DEADLINE_STARTED"
  printf '%s 0\n' "$DEADLINE_RUNNING"
  ;;
logs) printf 'OUTPUT: {"status":"completed","response":"fixture"}\n';;
ps)
  printf 'cleanup\n' > "$DEADLINE_CLEANUP"
  if [ "$DEADLINE_HANG" = error ]; then exit 1; fi
  if [ "$DEADLINE_HANG" = true ]; then
    while [ ! -f "$DEADLINE_RELEASE" ]; do /bin/sleep 1; done
    exit 1
  fi
  ;;
*) exit 2;;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	t.Cleanup(func() {
		if err := os.WriteFile(release, []byte("release"), 0o600); err != nil {
			t.Error(err)
		}
	})
	return
}

func awaitDeadlineFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(path); err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("fake Docker did not reach %s", filepath.Base(path))
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestDockerRuntimeCleanup(t *testing.T) {
	for _, owned := range []bool{false, true} {
		t.Run(fmt.Sprintf("hung daemon owned=%t", owned), func(t *testing.T) {
			started, cleanup, release := deadlineDockerFixture(t)
			rt := newCustomDockerRuntime(deadlineCacheDir(t), false, "", time.Minute, dockermgr.DefaultPolicy())
			gate := &sync.RWMutex{}
			e := &Executor{
				cacheDir: rt.cacheDir, runtime: &skipPrepare{inner: rt}, executionGate: gate, jobTimeout: time.Minute,
				teardown: cancellationTeardown(func(ctx context.Context, _ string) error {
					if _, ok := ctx.Deadline(); !ok {
						t.Error("final executor verification has no deadline")
					}
					return errors.New("fixture backend remains unavailable")
				}),
			}
			assignment := types.JobAssignment{JobID: "cleanup-deadline", Runtime: "docker-custom", DockerImage: "fixture-worker:latest"}
			if owned {
				assignment.ResourceName = "fixture-owned"
				assignment.ResourceLabels = map[string]string{"ai.rungpu.owner": "fixture"}
				assignment.ResourceDaemonID = "fixture-daemon"
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			results, err := e.Start(ctx, assignment)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			var result types.JobResult
			go func() { result = <-results; close(done) }()
			t.Cleanup(func() {
				_ = os.WriteFile(release, []byte("release"), 0o600)
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Error("fake Docker job did not stop after fixture release")
				}
				e.mu.Lock()
				if state := e.executions[assignment.JobID]; state != nil {
					e.releaseExecution(state)
				}
				e.mu.Unlock()
			})
			awaitDeadlineFile(t, started)
			cancel()
			awaitDeadlineFile(t, cleanup)
			start := time.Now()
			select {
			case <-done:
			case <-time.After(runtimeContainerCleanupTimeout + 5*time.Second):
				t.Fatal("runtime cleanup exceeded its independent 30-second budget")
			}
			if elapsed := time.Since(start); elapsed < runtimeContainerCleanupTimeout-time.Second {
				t.Fatalf("cleanup inherited the already-cancelled job context instead of its own budget: %s", elapsed)
			}
			if result.Success || !strings.Contains(result.Error, "teardown could not be verified") {
				t.Fatalf("unverified teardown was reported as stopped: %+v", result)
			}
			e.ForgetExecution(assignment.JobID)
			if !e.IsTracked(assignment.JobID) || e.ReservedJobs() != 1 {
				t.Fatal("failed teardown discarded execution reservation")
			}
			rejectAdmission(t, e, "retry-on-same-gpu", "docker-custom")
			sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
			rejectAdmission(t, sibling, "native-after-unverified-stop", "ollama")
		})
	}

	t.Run("verified cleanup still releases capacity", func(t *testing.T) {
		deadlineDockerFixture(t)
		t.Setenv("DEADLINE_RUNNING", "false")
		t.Setenv("DEADLINE_HANG", "false")
		rt := newCustomDockerRuntime(deadlineCacheDir(t), false, "", time.Minute, dockermgr.DefaultPolicy())
		gate := &sync.RWMutex{}
		e := &Executor{
			cacheDir: rt.cacheDir, runtime: &skipPrepare{inner: rt}, executionGate: gate, jobTimeout: time.Minute,
			teardown: cancellationTeardown(func(context.Context, string) error { return nil }),
		}
		result := e.Execute(context.Background(), types.JobAssignment{
			JobID: "cleanup-success", Runtime: "docker-custom", DockerImage: "fixture-worker:latest",
		})
		if !result.Success || result.Result["response"] != "fixture" {
			t.Fatalf("verified cleanup changed successful execution: %+v", result)
		}
		if !gate.TryLock() {
			t.Fatal("verified cleanup retained machine reservation")
		}
		gate.Unlock()
	})

	t.Run("failed workspace startup exposes cleanup failure", func(t *testing.T) {
		_, cleanup, _ := deadlineDockerFixture(t)
		t.Setenv("DEADLINE_RUNNING", "false")
		t.Setenv("DEADLINE_HANG", "error")
		rt := newWorkspaceRuntime(deadlineCacheDir(t), false, "", dockermgr.DefaultPolicy())
		result, err := rt.Run(context.Background(), types.JobAssignment{
			JobID: "workspace-cleanup-failure", Runtime: "workspace", DockerImage: "fixture-worker:latest",
		})
		if result != nil || err == nil || !strings.Contains(err.Error(), "workspace container failed to start") ||
			!strings.Contains(err.Error(), "workspace cleanup could not be verified") {
			t.Fatalf("workspace startup cleanup failure was hidden: result=%v error=%v", result, err)
		}
		if _, err := os.Stat(cleanup); err != nil {
			t.Fatalf("workspace did not attempt bounded runtime cleanup: %v", err)
		}
	})
}
