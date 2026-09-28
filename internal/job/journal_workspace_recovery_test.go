package job

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestJournalFailedWorkspaceResultRequiresVerifiedStopBeforeReplay(t *testing.T) {
	for _, unavailable := range []bool{false, true} {
		t.Run(map[bool]string{false: "verified-stop", true: "unverified-stop"}[unavailable], func(t *testing.T) {
			cache, docker := t.TempDir(), &journalDocker{}
			first := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
			a := types.JobAssignment{JobID: "workspace", DispatchToken: "A", Runtime: "workspace"}
			if result := first.Execute(context.Background(), a); !result.Success {
				t.Fatal(result.Error)
			}
			failed := types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken,
				GPUID: "gpu", Error: "workspace failed before verified teardown"}
			if err := first.journal.Complete(a.JobID, a.DispatchToken, &failed, false); err != nil {
				t.Fatal(err)
			}
			if record, _ := first.journal.Find(a.JobID, a.DispatchToken); !record.ReplayBlocked {
				t.Fatal("unverified workspace failure was allowed into result replay")
			}
			crashJournalExecutor(first)
			docker.unavailable = unavailable
			restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
			record, _ := restarted.journal.Find(a.JobID, a.DispatchToken)
			if record.Result == nil || record.Result.Success || record.Result.Error != failed.Error {
				t.Fatal("recovery lost the original terminal failure")
			}
			if record.StopVerified == unavailable || record.ReplayBlocked != unavailable {
				t.Fatal("failure replay was not gated by verified resource removal")
			}
			wantRemoved := 1
			if unavailable {
				wantRemoved = 0
			}
			if docker.removed != wantRemoved || docker.runs != 1 {
				t.Fatal("failed workspace recovery reran or incorrectly removed a resource")
			}
			if _, err := restarted.Start(context.Background(), a); !errors.Is(err, ErrAlreadyTracked) {
				t.Fatalf("same failed attempt was admitted again: %v", err)
			}
		})
	}
}
