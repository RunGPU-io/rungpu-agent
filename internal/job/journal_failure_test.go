package job

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestJournalUncataloguedPreRuntimeRecordRecoversWithoutExecution(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	e := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	catalog := filepath.Join(e.journal.dir, "catalog.json")
	original, err := os.ReadFile(catalog)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(catalog); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(catalog, 0o700); err != nil {
		t.Fatal(err)
	}
	a := types.JobAssignment{JobID: "crash", DispatchToken: "A", Runtime: "docker-custom"}
	if _, err := e.Start(context.Background(), a); err == nil || docker.runs != 0 {
		t.Fatal("runtime entered before the catalog checkpoint was durable")
	}
	if err := os.Remove(catalog); err != nil {
		t.Fatal(err)
	}
	var originalCatalog checkedJournalJSON
	if err := json.Unmarshal(original, &originalCatalog); err != nil {
		t.Fatal(err)
	}
	if err := atomicJournalJSON(e.journal.dir, filepath.Base(catalog), originalCatalog); err != nil {
		t.Fatal(err)
	}
	crashJournalExecutor(e)
	restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err != nil {
		t.Fatalf("pre-runtime crash did not retain never-started proof: %v", err)
	}
	if docker.runs != 0 || docker.removed != 0 {
		t.Fatal("pre-runtime recovery executed or removed a backend resource")
	}
}

func TestJournalCancellationWriteFailureCannotTouchOrAcknowledgeResource(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	e := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	a := types.JobAssignment{JobID: "workspace", DispatchToken: "A", Runtime: "workspace"}
	if result := e.Execute(context.Background(), a); !result.Success {
		t.Fatal(result.Error)
	}
	if err := setJournalDirectoryWritable(e.journal.dir, false); err != nil {
		t.Fatal(err)
	}
	if err := e.CancelAttempt(a.JobID, a.DispatchToken); err == nil || docker.removed != 0 {
		t.Fatal("unpersisted cancellation touched or acknowledged a resource")
	}
	if err := setJournalDirectoryWritable(e.journal.dir, true); err != nil {
		t.Fatal(err)
	}
	crashJournalExecutor(e)
	restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if docker.removed != 0 {
		t.Fatal("failed cancellation was invented as durable intent")
	}
	if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err != nil || docker.removed != 1 {
		t.Fatalf("explicit retry did not verify exactly one removal: %v", err)
	}
}

func TestJournalDurableCancellationIntentIsCompletedAtRestart(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	e := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	a := types.JobAssignment{JobID: "workspace", DispatchToken: "A", Runtime: "workspace"}
	if result := e.Execute(context.Background(), a); !result.Success {
		t.Fatal(result.Error)
	}
	if _, err := e.journal.RememberCancellation(a.JobID, a.DispatchToken, false); err != nil {
		t.Fatal(err)
	}
	crashJournalExecutor(e)
	restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if docker.removed != 1 {
		t.Fatal("restart did not finish the durable cancellation")
	}
	if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err != nil || docker.removed != 1 {
		t.Fatalf("recovered cancellation did not use durable stop proof: %v", err)
	}
}
