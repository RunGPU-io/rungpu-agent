package pool

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/gpu"
	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type processDocker struct{ dir string }

func (*processDocker) Name() string                                       { return "process-docker-fixture" }
func (*processDocker) Prepare(context.Context, types.JobAssignment) error { return nil }
func (*processDocker) Cleanup(bool) error                                 { return nil }
func (*processDocker) DaemonID(context.Context) (string, error)           { return "process-fixture-daemon", nil }
func (d *processDocker) Run(_ context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	count, err := os.OpenFile(filepath.Join(d.dir, "fake-runs"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	_, writeErr := count.WriteString("run\n")
	closeErr := count.Close()
	if writeErr != nil {
		return nil, writeErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	owner := dockermgr.OwnedResource{Name: a.ResourceName, Labels: a.ResourceLabels, DaemonID: a.ResourceDaemonID}
	data, err := json.Marshal(owner)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(d.dir, "fake-container.json"), data, 0o600); err != nil {
		return nil, err
	}
	if os.Getenv("RUNGPU_JOURNAL_CRASH_BEFORE_RESULT") == "1" {
		os.Exit(0)
	}
	return map[string]interface{}{"status": "running", "container": owner.Name}, nil
}
func (d *processDocker) InspectOwned(_ context.Context, owner dockermgr.OwnedResource) (bool, error) {
	data, err := os.ReadFile(filepath.Join(d.dir, "fake-container.json"))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	var actual dockermgr.OwnedResource
	if json.Unmarshal(data, &actual) != nil || !reflect.DeepEqual(actual, owner) {
		return false, fmt.Errorf("fake resource ownership mismatch")
	}
	return true, nil
}
func (d *processDocker) RemoveOwnedAndVerify(ctx context.Context, owner dockermgr.OwnedResource) error {
	present, err := d.InspectOwned(ctx, owner)
	if err != nil || !present {
		return err
	}
	if err := os.Remove(filepath.Join(d.dir, "fake-container.json")); err != nil {
		return err
	}
	count, err := os.OpenFile(filepath.Join(d.dir, "fake-removals"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer count.Close()
	_, err = count.WriteString("removed\n")
	return err
}

func processFixtureClient(t *testing.T, cache string) (*Client, *job.Executor) {
	t.Helper()
	t.Setenv("PATH", filepath.Join(cache, "no-executables"))
	docker, gate := &processDocker{dir: cache}, &sync.RWMutex{}
	executor, err := job.NewExecutorWithOptions(job.ExecutorOptions{
		CacheDir: cache, GPUID: "gpu-process", Backend: "cpu", Runtime: docker,
		OwnedTeardown: docker, ExecutionGate: gate,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = executor.Journal().Close() })
	client, err := clientWithExecutor(&types.Config{ModelCacheDir: cache}, "gpu-process", "cpu", 0, gpu.NewMonitor(), gate, executor)
	if err != nil {
		t.Fatal(err)
	}
	client.baseCtx = context.Background()
	return client, executor
}

func TestRestartWorkspaceProcessHelper(t *testing.T) {
	cache := os.Getenv("RUNGPU_JOURNAL_PROCESS_FIXTURE")
	if cache == "" {
		return
	}
	_, executor := processFixtureClient(t, cache)
	result := executor.Execute(context.Background(), types.JobAssignment{
		JobID: "workspace", DispatchToken: "A", Runtime: "workspace",
	})
	if !result.Success {
		t.Fatal(result.Error)
	}

	os.Exit(0)
}

func TestRestartClientReattachesWorkspaceFromExitedProcessAndCancelsExactlyOnce(t *testing.T) {
	cache := t.TempDir()
	cmd := exec.Command(os.Args[0], "-test.run=^TestRestartWorkspaceProcessHelper$")
	cmd.Env = append(os.Environ(), "RUNGPU_JOURNAL_PROCESS_FIXTURE="+cache)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fake workspace process failed: %v\n%s", err, output)
	}

	c, executor := processFixtureClient(t, cache)
	if !waitRetryResult(t, c, "workspace", "A").Success || c.heartbeatMessage().CurrentJobs != 1 {
		t.Fatal("fresh client did not recover the running owned workspace and pending result")
	}
	resend, err := json.Marshal(types.JobAssignment{
		Type: "job_assignment", JobID: "workspace", DispatchToken: "A", Runtime: "workspace",
	})
	if err != nil {
		t.Fatal(err)
	}
	c.dispatch(context.Background(), nil, resend)
	if result := waitRetryResult(t, c, "workspace", "A"); !result.Success || result.Result["status"] != "running" {
		t.Fatalf("assignment resend replaced the durable workspace result: %+v", result)
	}
	ackRetry(c, "workspace", "A")
	c.dispatch(context.Background(), nil, resend)
	if !executor.IsTracked("workspace") || c.heartbeatMessage().CurrentJobs != 1 {
		t.Fatal("acknowledged workspace resend lost the recovered execution reservation")
	}
	cancelRetry(c, "workspace", "wrong")
	readAttemptCancelAck(t, c, "wrong", false)
	if _, err := os.Stat(filepath.Join(cache, "fake-container.json")); err != nil {
		t.Fatal("wrong-token cancellation removed the recovered workspace")
	}
	cancelRetry(c, "workspace", "A")
	readAttemptCancelAck(t, c, "A", true)
	cancelRetry(c, "workspace", "A")
	readAttemptCancelAck(t, c, "A", true)
	removed, err := os.ReadFile(filepath.Join(cache, "fake-removals"))
	if err != nil || string(removed) != "removed\n" {
		t.Fatalf("resource was not removed exactly once: %q %v", removed, err)
	}
	runs, err := os.ReadFile(filepath.Join(cache, "fake-runs"))
	if err != nil || string(runs) != "run\n" {
		t.Fatalf("replayed running attempt executed more than once across restart: %q %v", runs, err)
	}
	ackRetry(c, "workspace", "A")
	executor.Journal().Close()
	again, _ := processFixtureClient(t, cache)
	cancelRetry(again, "workspace", "A")
	readAttemptCancelAck(t, again, "A", true)
	dispatchRetry(again, "workspace", "A")
	if _, err := os.Stat(filepath.Join(cache, "fake-container.json")); !os.IsNotExist(err) {
		t.Fatal("durable cancellation was lost in the next process incarnation")
	}
}
