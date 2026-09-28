package pool

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/gpu"
	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type restartRuntime struct {
	mu   sync.Mutex
	runs int
}

func (*restartRuntime) Name() string                                       { return "restart-fixture" }
func (*restartRuntime) Prepare(context.Context, types.JobAssignment) error { return nil }
func (*restartRuntime) Cleanup(bool) error                                 { return nil }
func (*restartRuntime) DaemonID(context.Context) (string, error)           { return "fake-daemon", nil }
func (r *restartRuntime) Run(context.Context, types.JobAssignment) (map[string]interface{}, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.runs++
	return map[string]interface{}{"response": "durable output"}, nil
}
func (*restartRuntime) RemoveOwnedAndVerify(context.Context, dockermgr.OwnedResource) error {
	return nil
}
func (*restartRuntime) InspectOwned(context.Context, dockermgr.OwnedResource) (bool, error) {
	return false, nil
}

func restartFixtureClient(t *testing.T, cache string, runtime *restartRuntime) (*Client, *job.Executor) {
	t.Helper()
	t.Setenv("PATH", t.TempDir())
	gate := &sync.RWMutex{}
	executor, err := job.NewExecutorWithOptions(job.ExecutorOptions{
		CacheDir: cache, GPUID: "gpu-restart", Backend: "cpu", Runtime: runtime,
		OwnedTeardown: runtime, ExecutionGate: gate,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = executor.Journal().Close() })
	cfg := &types.Config{ModelCacheDir: cache}
	c, err := clientWithExecutor(cfg, "gpu-restart", "cpu", 0, gpu.NewMonitor(), gate, executor)
	if err != nil {
		t.Fatal(err)
	}
	c.baseCtx = context.Background()
	return c, executor
}

func TestRestartClientRecoversCompletionBeforeOutboxAndAcceptedResult(t *testing.T) {
	cache, runtime := t.TempDir(), &restartRuntime{}
	_, first := restartFixtureClient(t, cache, runtime)
	a := types.JobAssignment{JobID: "done", DispatchToken: "A", Runtime: "docker-custom"}
	result := first.Execute(context.Background(), a)
	if !result.Success {
		t.Fatal(result.Error)
	}

	first.Journal().Close()
	c, second := restartFixtureClient(t, cache, runtime)
	if result := waitRetryResult(t, c, "done", "A"); !result.Success || result.Result["response"] != "durable output" {
		t.Fatalf("durable result not restored: %+v", result)
	}
	dispatchRetry(c, "done", "A")
	if runtime.runs != 1 {
		t.Fatal("completed attempt ran again after a fresh client restart")
	}

	if err := second.Journal().Accept("done", "A"); err != nil {
		t.Fatal(err)
	}
	second.Journal().Close()
	again, _ := restartFixtureClient(t, cache, runtime)
	if len(again.results) != 0 {
		t.Fatal("accepted result was resurrected from a stale outbox file")
	}
	dispatchRetry(again, "done", "A")
	if runtime.runs != 1 || len(again.results) != 0 {
		t.Fatal("accepted attempt was rerun or poisoned by replay")
	}
}

func TestRestartClientCancellationTombstonesAndNeverStartedProof(t *testing.T) {
	cache, runtime := t.TempDir(), &restartRuntime{}
	first, executor := restartFixtureClient(t, cache, runtime)
	a := types.JobAssignment{JobID: "deferred", DispatchToken: "B", Runtime: "docker-custom"}
	if _, err := executor.Journal().Admit(a, "cpu"); err != nil {
		t.Fatal(err)
	}
	first.deferredAssignments = map[string]types.JobAssignment{a.JobID: a}
	cancelRetry(first, "deferred", "B")
	readAttemptCancelAck(t, first, "B", true)
	cancelRetry(first, "unknown", "old")
	readAttemptCancelAck(t, first, "old", false)
	executor.Journal().Close()
	again, _ := restartFixtureClient(t, cache, runtime)
	cancelRetry(again, "deferred", "B")
	readAttemptCancelAck(t, again, "B", true)
	cancelRetry(again, "unknown", "old")
	readAttemptCancelAck(t, again, "old", false)
	ackRetry(again, "deferred", "B")
	dispatchRetry(again, "deferred", "B")
	dispatchRetry(again, "unknown", "old")
	if runtime.runs != 0 {
		t.Fatal("restart forgot cancelled or known-never-started attempts")
	}
}

func TestRestartClientAcceptedReplayCannotCreateRejection(t *testing.T) {
	for _, scenario := range []string{"busy", "invalid", "newer-pending"} {
		t.Run(scenario, func(t *testing.T) {
			cache, runtime := t.TempDir(), &restartRuntime{}
			_, first := restartFixtureClient(t, cache, runtime)
			result := first.Execute(context.Background(), types.JobAssignment{
				JobID: "done", DispatchToken: "A", Runtime: "docker-custom",
			})
			if !result.Success {
				t.Fatal(result.Error)
			}
			if err := first.Journal().Accept("done", "A"); err != nil {
				t.Fatal(err)
			}
			first.Journal().Close()
			c, executor := restartFixtureClient(t, cache, runtime)
			wantPending := 0
			switch scenario {
			case "busy":
				c.jobSlot <- struct{}{}
				c.activeAssignment, c.activeDispatchToken = "other", "B"
			case "newer-pending":
				c.queueResult(types.JobResult{
					Type: "job_result", JobID: "done", GPUID: "gpu-restart", DispatchToken: "B", Success: true,
				})
				wantPending = 1
			}
			for i := 0; i < 3; i++ {
				if scenario == "invalid" {
					c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"done","dispatch_token":"A","custom_files":"invalid"}`))
				} else {
					dispatchRetry(c, "done", "A")
				}
			}
			if runtime.runs != 1 || len(c.results) != wantPending || len(c.deferredAssignments) != 0 {
				t.Fatalf("acknowledged replay reran or created new delivery: runs=%d results=%d deferred=%d",
					runtime.runs, len(c.results), len(c.deferredAssignments))
			}
			if scenario == "newer-pending" && (c.results["done"].result.DispatchToken != "B" || !c.results["done"].result.Success) {
				t.Fatal("retired replay replaced the newer pending result")
			}
			record, exists := executor.Journal().Find("done", "A")
			if !exists || !record.ResultAccepted || record.Result == nil || !record.Result.Success {
				t.Fatal("retired replay changed the durable accepted outcome")
			}
		})
	}
}

func TestRestartClientPreJournalResultDoesNotInventStopProof(t *testing.T) {
	cache := os.Getenv("RUNGPU_TEST_UNVERIFIED_LEGACY_CACHE")
	if runtime.GOOS == "windows" && cache == "" {

		cmd := exec.Command(os.Args[0], "-test.run=^TestRestartClientPreJournalResultDoesNotInventStopProof$")
		cmd.Env = append(os.Environ(), "RUNGPU_TEST_UNVERIFIED_LEGACY_CACHE="+t.TempDir())
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("unverified legacy subprocess failed: %v\n%s", err, output)
		}
		return
	}
	if cache == "" {
		cache = t.TempDir()
	}
	runtime := &restartRuntime{}
	outbox := filepath.Join(cache, "outbox", "gpu-restart")
	if err := os.MkdirAll(outbox, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outbox, "legacy.json"),
		[]byte(`{"type":"job_result","job_id":"legacy","gpu_id":"gpu-restart","success":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	c, executor := restartFixtureClient(t, cache, runtime)
	if !job.MachineRecoveryBlocked(c.maintenanceGate) {
		t.Fatal("unknown legacy execution did not reserve the machine")
	}
	if err := executor.Cancel("legacy"); err == nil {
		t.Fatal("legacy result was invented as verified backend stop")
	}
	if _, err := executor.Start(context.Background(), types.JobAssignment{
		JobID: "other", DispatchToken: "B", Runtime: "docker-custom",
	}); err == nil || errors.Is(err, job.ErrAlreadyTracked) {
		t.Fatalf("legacy recovery did not fence unrelated admission: %v", err)
	}
}
