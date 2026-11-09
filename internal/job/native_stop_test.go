package job

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type nativeStopBarrier struct {
	inner     Runtime
	returned  chan struct{}
	cancelled chan struct{}
	release   chan struct{}
}

func (*nativeStopBarrier) Name() string                                       { return "native-stop-fixture" }
func (*nativeStopBarrier) Prepare(context.Context, types.JobAssignment) error { return nil }
func (*nativeStopBarrier) Cleanup(bool) error                                 { return nil }
func (r *nativeStopBarrier) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	result, err := r.inner.Run(ctx, a)
	close(r.returned)
	<-ctx.Done()
	close(r.cancelled)
	<-r.release
	return result, err
}

func nativeStopExecutor(t *testing.T, cache string, gate *sync.RWMutex, runtime Runtime) *Executor {
	t.Helper()
	e := newJournalExecutor(t, cache, "native-proof-gpu", gate, &journalDocker{})
	e.runtime = runtime
	e.teardown = cancellationTeardown(func(context.Context, string) error {
		t.Error("native stop proof must not use Docker resource removal")
		return errors.New("unexpected Docker removal")
	})
	return e
}

func awaitNativeSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(2 * time.Second):
		t.Fatal("native fixture did not reach expected execution state")
	}
}

func TestNativeStopTerminalProof(t *testing.T) {
	for _, scenario := range []string{"terminal-done", "missing-done", "malformed", "http-error", "result-map-only"} {
		t.Run(scenario, func(t *testing.T) {
			inner := Runtime(deadlineOllama(t, func(req *http.Request) (*http.Response, error) {
				body, status := `{"done":true,"response":"fixture"}`, http.StatusOK
				switch scenario {
				case "missing-done":
					body = `{"response":"not confirmed"}`
				case "malformed":
					body = `{invalid`
				case "http-error":
					status = http.StatusInternalServerError
				}
				return &http.Response{
					StatusCode: status, Header: make(http.Header),
					Body: io.NopCloser(strings.NewReader(body)), Request: req,
				}, nil
			}))
			if scenario == "result-map-only" {
				inner = &cancellationRuntime{run: func(context.Context) (map[string]interface{}, error) {
					return map[string]interface{}{"done": true, "status": "completed", "response": "untrusted proof"}, nil
				}}
			}
			barrier := &nativeStopBarrier{
				inner: inner, returned: make(chan struct{}), cancelled: make(chan struct{}), release: make(chan struct{}),
			}
			var once sync.Once
			unblock := func() { once.Do(func() { close(barrier.release) }) }
			defer unblock()
			cache, gate := t.TempDir(), &sync.RWMutex{}
			e := nativeStopExecutor(t, cache, gate, barrier)
			a := types.JobAssignment{JobID: "native-stop", DispatchToken: "attempt-A", Runtime: "ollama", ModelName: "fixture"}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			results, err := e.Start(ctx, a)
			if err != nil {
				t.Fatal(err)
			}
			awaitNativeSignal(t, barrier.returned)
			if err := e.CancelAttempt(a.JobID, "attempt-B"); err == nil {
				t.Fatal("a different dispatch token reused the native completion proof")
			}
			select {
			case <-barrier.cancelled:
				t.Fatal("wrong-token cancellation reached the runtime")
			default:
			}
			cancelResult := make(chan error, 1)
			go func() { cancelResult <- e.CancelAttempt(a.JobID, a.DispatchToken) }()
			awaitNativeSignal(t, barrier.cancelled)
			select {
			case err := <-cancelResult:
				t.Fatalf("stop acknowledged before execution joined: %v", err)
			default:
			}
			if gate.TryLock() {
				gate.Unlock()
				t.Fatal("machine released before execution joined and journal completion committed")
			}
			unblock()
			<-results
			err = <-cancelResult
			wantVerified := scenario == "terminal-done"
			if (err == nil) != wantVerified {
				t.Fatalf("stop confirmation for %s: %v", scenario, err)
			}
			record, ok := e.journal.Find(a.JobID, a.DispatchToken)
			if !ok || record.StopVerified != wantVerified || !record.CancelRequested {
				t.Fatalf("incorrect durable stop state: %+v", record)
			}
			if wantVerified {
				if err := e.CancelAttempt(a.JobID, a.DispatchToken); err != nil {
					t.Fatalf("confirmed native cancellation is not idempotent: %v", err)
				}
				crashJournalExecutor(e)
				restarted := nativeStopExecutor(t, cache, &sync.RWMutex{}, inner)
				if MachineRecoveryBlocked(restarted.executionGate) {
					t.Fatal("durably completed native request became an unverified restart fence")
				}
				if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err != nil {
					t.Fatalf("restart lost exact-attempt completion proof: %v", err)
				}
				if err := restarted.CancelAttempt(a.JobID, "attempt-B"); err == nil {
					t.Fatal("restart applied one attempt's proof to another attempt")
				}
			} else {
				e.ForgetExecution(a.JobID)
				if !e.IsTracked(a.JobID) || e.ReservedJobs() != 1 {
					t.Fatal("unconfirmed native request lost its execution reservation")
				}
				sibling := admissionExecutor(gate, func(context.Context) (map[string]interface{}, error) { return nil, nil })
				rejectAdmission(t, sibling, "after-unconfirmed-native", "docker-custom")
				crashJournalExecutor(e)
				restarted := nativeStopExecutor(t, cache, &sync.RWMutex{}, inner)
				if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err == nil ||
					!MachineRecoveryBlocked(restarted.executionGate) {
					t.Fatal("restart manufactured completion proof for an unverified native request")
				}
			}
		})
	}
}

func TestNativeStopProofRequiresDurableCommit(t *testing.T) {
	inner := deadlineOllama(t, func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK, Header: make(http.Header),
			Body: io.NopCloser(strings.NewReader(`{"done":true}`)), Request: req,
		}, nil
	})
	barrier := &nativeStopBarrier{
		inner: inner, returned: make(chan struct{}), cancelled: make(chan struct{}), release: make(chan struct{}),
	}
	var once sync.Once
	unblock := func() { once.Do(func() { close(barrier.release) }) }
	defer unblock()
	cache, gate := t.TempDir(), &sync.RWMutex{}
	e := nativeStopExecutor(t, cache, gate, barrier)
	a := types.JobAssignment{JobID: "native-proof-write-failure", DispatchToken: "A", Runtime: "ollama"}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	results, err := e.Start(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	awaitNativeSignal(t, barrier.returned)
	cancelResult := make(chan error, 1)
	go func() { cancelResult <- e.CancelAttempt(a.JobID, a.DispatchToken) }()
	awaitNativeSignal(t, barrier.cancelled)
	if err := setJournalDirectoryWritable(e.journal.dir, false); err != nil {
		t.Fatal(err)
	}
	defer setJournalDirectoryWritable(e.journal.dir, true)
	unblock()
	if result := <-results; result.Success {
		t.Fatal("execution reported success without durably committing stop proof")
	}
	if err := <-cancelResult; err == nil {
		t.Fatal("native stop was acknowledged after completion-proof persistence failed")
	}
	if gate.TryLock() {
		gate.Unlock()
		t.Fatal("machine released without durable stop proof")
	}
}

func TestNativeStopBeforeRun(t *testing.T) {
	for _, prepareErr := range []bool{false, true} {
		t.Run(map[bool]string{false: "prepare-returns-after-cancel", true: "prepare-reports-cancel"}[prepareErr], func(t *testing.T) {
			started := make(chan struct{})
			rt := &cancellationRuntime{
				prepare: func(ctx context.Context) error {
					close(started)
					<-ctx.Done()
					if prepareErr {
						return ctx.Err()
					}
					return nil
				},
				run: func(context.Context) (map[string]interface{}, error) {
					t.Error("cancelled preparation entered native inference")
					return nil, nil
				},
			}
			cache, gate := t.TempDir(), &sync.RWMutex{}
			e := nativeStopExecutor(t, cache, gate, rt)
			a := types.JobAssignment{JobID: "native-preparation", DispatchToken: "A", Runtime: "ollama"}
			results, err := e.Start(context.Background(), a)
			if err != nil {
				t.Fatal(err)
			}
			awaitNativeSignal(t, started)
			if err := e.CancelAttempt(a.JobID, a.DispatchToken); err != nil {
				t.Fatalf("never-entered native request remained reserved: %v", err)
			}
			if result := <-results; result.Success {
				t.Fatal("cancelled preparation was reported as successful inference")
			}
			record, _ := e.journal.Find(a.JobID, a.DispatchToken)
			if !record.StopVerified || !record.CancelRequested {
				t.Fatalf("never-started cancellation proof was not persisted: %+v", record)
			}
			if !gate.TryLock() {
				t.Fatal("never-started native execution kept machine admission locked")
			}
			gate.Unlock()
		})
	}
}

func TestNativeStopCompletionIdentity(t *testing.T) {
	for _, mismatch := range []string{"job", "token"} {
		t.Run(mismatch, func(t *testing.T) {
			rt := deadlineOllama(t, func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK, Header: make(http.Header),
					Body: io.NopCloser(strings.NewReader(`{"done":true}`)), Request: req,
				}, nil
			})
			proof := &nativeCompletionProof{jobID: "job", dispatchToken: "A"}
			ctx := context.WithValue(context.Background(), nativeCompletionContextKey{}, proof)
			a := types.JobAssignment{JobID: "job", DispatchToken: "A"}
			if mismatch == "job" {
				a.JobID = "another-job"
			} else {
				a.DispatchToken = "B"
			}
			if _, err := rt.Run(ctx, a); err == nil || proof.completed {
				t.Fatal("terminal response was accepted as proof for a different execution attempt")
			}
		})
	}
}
