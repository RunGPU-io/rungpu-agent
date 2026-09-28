package pool

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/job"
)

func TestRestartClientWorkspaceCreatedBeforeResultFailsDurablyWithoutRerun(t *testing.T) {
	cache := t.TempDir()
	cmd := exec.Command(os.Args[0], "-test.run=^TestRestartWorkspaceProcessHelper$")
	cmd.Env = append(os.Environ(), "RUNGPU_JOURNAL_PROCESS_FIXTURE="+cache, "RUNGPU_JOURNAL_CRASH_BEFORE_RESULT=1")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fake workspace process failed: %v\n%s", err, output)
	}
	if _, err := os.Stat(filepath.Join(cache, "fake-container.json")); err != nil {
		t.Fatal("fixture did not crash after container creation")
	}
	proof, err := job.OpenAttemptJournal(cache, "gpu-process")
	if err != nil {
		t.Fatal(err)
	}
	interrupted, exists := proof.Find("workspace", "A")
	proof.Close()
	if !exists || interrupted.Result != nil || interrupted.Phase != "running" {
		t.Fatal("fixture did not leave the exact pre-result execution crash window")
	}
	c, executor := processFixtureClient(t, cache)
	result := waitRetryResult(t, c, "workspace", "A")
	if result.Success || result.Error == "" || c.heartbeatMessage().CurrentJobs != 0 {
		t.Fatalf("result-less workspace did not become a stopped recovery failure: %+v", result)
	}
	record, exists := executor.Journal().Find("workspace", "A")
	if !exists || !record.StopVerified || record.Result == nil || record.Result.Success {
		t.Fatal("workspace removal and recovery failure were not durable")
	}
	executor.Journal().Close()
	again, _ := processFixtureClient(t, cache)
	if replay := waitRetryResult(t, again, "workspace", "A"); replay.Success || replay.Error != result.Error {
		t.Fatalf("recovery failure was not replayed after another restart: %+v", replay)
	}
	dispatchRetry(again, "workspace", "A")
	ackRetry(again, "workspace", "A")
	dispatchRetry(again, "workspace", "A")
	for file, want := range map[string]string{"fake-runs": "run\n", "fake-removals": "removed\n"} {
		data, err := os.ReadFile(filepath.Join(cache, file))
		if err != nil || string(data) != want {
			t.Fatalf("%s: expected exactly once, got %q (%v)", file, data, err)
		}
	}
	if _, err := os.Stat(filepath.Join(cache, "fake-container.json")); !os.IsNotExist(err) {
		t.Fatal("result-less workspace leaked or reran")
	}
}
