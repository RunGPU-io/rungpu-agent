package job

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func admissionExecutor(gate *sync.RWMutex, run func(context.Context) (map[string]interface{}, error)) *Executor {
	return &Executor{
		executionGate: gate,
		runtime:       &cancellationRuntime{run: run},
		teardown:      cancellationTeardown(func(context.Context, string) error { return nil }),
	}
}

func startAdmission(t *testing.T, e *Executor, id, runtime string) <-chan types.JobResult {
	t.Helper()
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: id, Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	return results
}

func awaitAdmission(t *testing.T, results <-chan types.JobResult) types.JobResult {
	t.Helper()
	select {
	case result := <-results:
		return result
	case <-time.After(time.Second):
		t.Fatal("execution did not finish")
		return types.JobResult{}
	}
}

func rejectAdmission(t *testing.T, e *Executor, id, runtime string) {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := e.Start(context.Background(), types.JobAssignment{JobID: id, Runtime: runtime})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil || (!strings.Contains(err.Error(), "busy") && !strings.Contains(err.Error(), "already running")) {
			t.Fatalf("expected immediate busy rejection, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("admission blocked instead of rejecting")
	}
}

func TestMachineAdmissionAcrossGPUExecutors(t *testing.T) {
	for _, firstRuntime := range []string{"ollama", "docker-custom", "comfyui-batch", "workspace"} {
		for _, nextRuntime := range []string{"ollama", "docker-custom", "comfyui-batch", "workspace"} {
			t.Run(firstRuntime+"/"+nextRuntime, func(t *testing.T) {
				gate := &sync.RWMutex{}
				release := make(chan struct{})
				var once sync.Once
				unblock := func() { once.Do(func() { close(release) }) }
				defer unblock()
				run := func(context.Context) (map[string]interface{}, error) {
					<-release
					return nil, nil
				}
				first, sibling := admissionExecutor(gate, run), admissionExecutor(gate, run)
				results := startAdmission(t, first, "first", firstRuntime)
				var siblingResults <-chan types.JobResult
				if firstRuntime == "ollama" || nextRuntime == "ollama" {
					rejectAdmission(t, sibling, "sibling", nextRuntime)
				} else {
					siblingResults = startAdmission(t, sibling, "sibling", nextRuntime)
				}
				rejectAdmission(t, first, "same-gpu", "docker-custom")
				unblock()
				if result := awaitAdmission(t, results); !result.Success {
					t.Fatal(result.Error)
				}
				if siblingResults != nil {
					if result := awaitAdmission(t, siblingResults); !result.Success {
						t.Fatal(result.Error)
					}
				}
				if firstRuntime == "workspace" {
					if err := first.Cancel("first"); err != nil {
						t.Fatal(err)
					}
				}
				if siblingResults != nil && nextRuntime == "workspace" {
					if err := sibling.Cancel("sibling"); err != nil {
						t.Fatal(err)
					}
				}
				if result := awaitAdmission(t, startAdmission(t, sibling, "after", "ollama")); !result.Success {
					t.Fatal(result.Error)
				}
			})
		}
	}
}

func TestMachineAdmissionSimultaneousNativeStarts(t *testing.T) {
	for i := 0; i < 50; i++ {
		gate := &sync.RWMutex{}
		release, start := make(chan struct{}), make(chan struct{})
		run := func(context.Context) (map[string]interface{}, error) {
			<-release
			return nil, nil
		}
		type attempt struct {
			results <-chan types.JobResult
			err     error
		}
		attempts := make(chan attempt, 2)
		for _, e := range []*Executor{admissionExecutor(gate, run), admissionExecutor(gate, run)} {
			go func(e *Executor) {
				<-start
				results, err := e.Start(context.Background(), types.JobAssignment{JobID: "native", Runtime: "ollama"})
				attempts <- attempt{results, err}
			}(e)
		}
		close(start)
		first, second := <-attempts, <-attempts
		close(release)
		successes := 0
		for _, attempt := range []attempt{first, second} {
			if attempt.err == nil {
				successes++
				awaitAdmission(t, attempt.results)
			}
		}
		if successes != 1 {
			t.Fatalf("simultaneous native starts admitted %d jobs", successes)
		}
	}
}

func TestMachineAdmissionWorkspaceLifetime(t *testing.T) {
	for _, uploadFails := range []bool{false, true} {
		t.Run(map[bool]string{false: "startup-success", true: "upload-failure"}[uploadFails], func(t *testing.T) {
			gate := &sync.RWMutex{}
			e := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) {
				return map[string]interface{}{"output_file": "nonexistent-workspace-output"}, nil
			})
			a := types.JobAssignment{JobID: "workspace", Runtime: "workspace"}
			if uploadFails {
				a.UploadURL = "http://127.0.0.1:1/upload"
			}
			result := e.Execute(context.Background(), a)
			if result.Success == uploadFails {
				t.Fatalf("unexpected workspace startup result: %+v", result)
			}
			e.ForgetExecution(a.JobID)
			sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
			rejectAdmission(t, sibling, "native", "ollama")
			rejectAdmission(t, e, "same-gpu", "docker-custom")
			if gate.TryLock() {
				gate.Unlock()
				t.Fatal("workspace startup released maintenance exclusion")
			}
			if result := awaitAdmission(t, startAdmission(t, sibling, "docker", "docker-custom")); !result.Success {
				t.Fatal(result.Error)
			}
			e.teardown = cancellationTeardown(func(context.Context, string) error { return errors.New("daemon offline") })
			if err := e.Cancel(a.JobID); err == nil {
				t.Fatal("unverified workspace stop succeeded")
			}
			e.ForgetExecution(a.JobID)
			rejectAdmission(t, sibling, "native-retry", "ollama")
			e.teardown = cancellationTeardown(func(context.Context, string) error { return nil })
			if err := e.Cancel(a.JobID); err != nil {
				t.Fatal(err)
			}
			if result := awaitAdmission(t, startAdmission(t, sibling, "after-stop", "ollama")); !result.Success {
				t.Fatal(result.Error)
			}
		})
	}
}

func TestMachineAdmissionUnverifiedNativeStop(t *testing.T) {
	for _, cause := range []string{"cancel", "timeout", "runtime-error"} {
		t.Run(cause, func(t *testing.T) {
			gate := &sync.RWMutex{}
			started := make(chan struct{})
			e := admissionExecutor(gate, func(ctx context.Context) (map[string]interface{}, error) {
				close(started)
				if cause == "runtime-error" {
					return nil, errors.New("connection lost")
				}
				<-ctx.Done()
				return nil, ctx.Err()
			})
			if cause == "timeout" {
				e.jobTimeout = 20 * time.Millisecond
			}
			results := startAdmission(t, e, "native", "ollama")
			<-started
			if cause == "cancel" {
				if err := e.Cancel("native"); err == nil || !strings.Contains(err.Error(), "cannot confirm backend") {
					t.Fatalf("native cancellation incorrectly confirmed stop: %v", err)
				}
			}
			result := awaitAdmission(t, results)
			if result.Success || !strings.Contains(result.Error, "stop cannot be verified") {
				t.Fatalf("unverified native failure not explicit: %+v", result)
			}
			e.ForgetExecution("native")
			sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
			for _, runtime := range []string{"ollama", "docker-custom", "workspace"} {
				rejectAdmission(t, sibling, "next-"+runtime, runtime)
			}
			if err := e.StopAll(context.Background()); err == nil {
				t.Fatal("shutdown silently discarded unverified native stop")
			}
		})
	}
}

func TestMachineAdmissionDockerTeardownFailure(t *testing.T) {
	for _, runtime := range []string{"docker-custom", "workspace"} {
		t.Run(runtime, func(t *testing.T) {
			gate := &sync.RWMutex{}
			e := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) {
				return nil, errors.New("container start response lost")
			})
			e.teardown = cancellationTeardown(func(context.Context, string) error { return errors.New("cannot inspect") })
			result := awaitAdmission(t, startAdmission(t, e, "uncertain", runtime))
			if result.Success || !strings.Contains(result.Error, "teardown could not be verified") {
				t.Fatalf("missing teardown failure: %+v", result)
			}
			e.ForgetExecution("uncertain")
			sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
			rejectAdmission(t, sibling, "native", "ollama")
			rejectAdmission(t, e, "same-gpu", "docker-custom")
			e.teardown = cancellationTeardown(func(context.Context, string) error { return nil })
			if err := e.Cancel("uncertain"); err != nil {
				t.Fatal(err)
			}
			awaitAdmission(t, startAdmission(t, sibling, "after", "ollama"))
		})
	}
}

func TestMachineAdmissionCancellationTimeoutRetainsLease(t *testing.T) {
	gate := &sync.RWMutex{}
	release, started := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	e := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) {
		close(started)
		<-release
		return nil, nil
	})
	results := startAdmission(t, e, "docker", "docker-custom")
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := e.CancelContext(ctx, "docker"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected bounded join failure: %v", err)
	}
	unblock()
	awaitAdmission(t, results)
	e.ForgetExecution("docker")
	sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
	rejectAdmission(t, sibling, "native", "ollama")
	if err := e.Cancel("docker"); err != nil {
		t.Fatal(err)
	}
	awaitAdmission(t, startAdmission(t, sibling, "after", "ollama"))
}

func TestMachineAdmissionPreparationFailureAndMaintenance(t *testing.T) {
	gate := &sync.RWMutex{}
	e := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) {
		t.Error("run called after preparation failure")
		return nil, nil
	})
	e.runtime.(*cancellationRuntime).prepare = func(context.Context) error { return errors.New("cannot prepare") }
	for _, runtime := range []string{"ollama", "docker-custom", "workspace"} {
		gate.Lock()
		rejectAdmission(t, e, runtime+"-maintenance", runtime)
		gate.Unlock()
		if result := awaitAdmission(t, startAdmission(t, e, runtime, runtime)); result.Success {
			t.Fatal("preparation failure succeeded")
		}
		if !gate.TryLock() {
			t.Fatal("preparation failure leaked machine lease")
		}
		gate.Unlock()
	}
	if _, err := e.Start(context.Background(), types.JobAssignment{JobID: "invalid", Runtime: "unknown"}); err == nil {
		t.Fatal("unknown runtime admitted")
	}
}
