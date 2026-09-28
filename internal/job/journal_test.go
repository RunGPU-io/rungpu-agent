package job

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type journalDocker struct {
	mu          sync.Mutex
	resources   map[string]dockermgr.OwnedResource
	runs        int
	removed     int
	unavailable bool
	beforeRun   func(types.JobAssignment) error
}

func (*journalDocker) Name() string                                       { return "journal-fake-docker" }
func (*journalDocker) Prepare(context.Context, types.JobAssignment) error { return nil }
func (*journalDocker) Cleanup(bool) error                                 { return nil }
func (*journalDocker) DaemonID(context.Context) (string, error)           { return "fake-daemon", nil }
func (d *journalDocker) Run(_ context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.beforeRun != nil {
		if err := d.beforeRun(a); err != nil {
			return nil, err
		}
	}
	if a.Runtime == "ollama" {
		d.runs++
		return map[string]interface{}{"response": "native fixture"}, nil
	}
	if a.ResourceName == "" || len(a.ResourceLabels) != 6 {
		return nil, errors.New("runtime entered without durable ownership")
	}
	d.runs++
	if d.resources == nil {
		d.resources = make(map[string]dockermgr.OwnedResource)
	}
	d.resources[a.ResourceName] = dockermgr.OwnedResource{Name: a.ResourceName, Labels: a.ResourceLabels, DaemonID: a.ResourceDaemonID}
	return map[string]interface{}{"response": "fixture", "container": a.ResourceName}, nil
}
func (d *journalDocker) RemoveOwnedAndVerify(_ context.Context, owner dockermgr.OwnedResource) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.unavailable {
		return errors.New("fake Docker unavailable")
	}
	if found, exists := d.resources[owner.Name]; exists {
		if !reflect.DeepEqual(found, owner) {
			return errors.New("resource is not owned by this attempt")
		}
		delete(d.resources, owner.Name)
		d.removed++
	}
	return nil
}

func (d *journalDocker) InspectOwned(_ context.Context, owner dockermgr.OwnedResource) (bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.unavailable {
		return false, errors.New("fake Docker unavailable")
	}
	found, exists := d.resources[owner.Name]
	if exists && !reflect.DeepEqual(found, owner) {
		return false, errors.New("resource is not owned by this attempt")
	}
	return exists, nil
}

func newJournalExecutor(t *testing.T, cache, gpuID string, gate *sync.RWMutex, docker *journalDocker) *Executor {
	t.Helper()
	e, err := NewExecutorWithOptions(ExecutorOptions{
		CacheDir: cache, GPUID: gpuID, Backend: "cpu", Runtime: docker,
		OwnedTeardown: docker, ExecutionGate: gate,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		crashJournalExecutor(e)
		restoredLeaseMu.Lock()
		for key, release := range restoredLeases {
			if strings.HasPrefix(key, e.journal.dir+"/") {
				delete(restoredLeases, key)
				release()
			}
		}
		restoredLeaseMu.Unlock()
		recoveryGates.Delete(e.executionGate)
	})
	return e
}

func crashJournalExecutor(e *Executor) {
	e.mu.Lock()
	for _, state := range e.executions {
		e.releaseExecution(state)
	}
	e.mu.Unlock()
	_ = e.Journal().Close()
}

func TestJournalRestartOwnedDockerCancelAndAttemptIsolation(t *testing.T) {
	cache := t.TempDir()
	docker := &journalDocker{}
	first := newJournalExecutor(t, cache, "gpu-1", &sync.RWMutex{}, docker)
	docker.beforeRun = func(a types.JobAssignment) error {
		var record AttemptRecord
		if err := readCheckedJSON(filepath.Join(first.journal.dir, attemptKey(a.JobID, a.DispatchToken)+".json"), &record); err != nil {
			return err
		}
		if record.Phase != "running" || record.Resource.Name != a.ResourceName || record.GPUID != "gpu-1" {
			return errors.New("runtime started before ownership and running phase were durable")
		}
		return nil
	}
	a := types.JobAssignment{JobID: "workspace", DispatchToken: "A", Runtime: "workspace"}
	if result := first.Execute(context.Background(), a); !result.Success {
		t.Fatalf("first execution failed: %+v", result)
	}
	docker.beforeRun = nil
	crashJournalExecutor(first)

	restarted := newJournalExecutor(t, cache, "gpu-1", &sync.RWMutex{}, docker)
	if docker.removed != 0 || len(docker.resources) != 1 {
		t.Fatal("restart did not reattach the exact owned workspace")
	}
	if err := restarted.CancelAttempt("workspace", "A"); err != nil {
		t.Fatal(err)
	}
	if docker.removed != 1 {
		t.Fatal("recovered workspace cancellation did not remove exactly once")
	}
	if err := restarted.journal.Accept("workspace", "A"); err != nil {
		t.Fatal(err)
	}
	restarted.ForgetExecution("workspace")
	a.DispatchToken = "B"
	if result := restarted.Execute(context.Background(), a); !result.Success {
		t.Fatalf("new attempt failed after verified recovery: %+v", result)
	}
	if err := restarted.CancelAttempt("workspace", "A"); err == nil {
		t.Fatal("old token cancelled the newly tracked attempt")
	}
	if len(docker.resources) != 1 {
		t.Fatal("wrong-token cancellation touched B")
	}
	if err := restarted.CancelAttempt("workspace", "B"); err != nil {
		t.Fatal(err)
	}
	if docker.removed != 2 {
		t.Fatalf("expected exactly one removal per owned attempt, got %d", docker.removed)
	}
}

func TestJournalCrashWindowsBeforeAndAfterContainerCreation(t *testing.T) {
	for _, phase := range []string{"unstarted", "before-container", "after-container"} {
		t.Run(phase, func(t *testing.T) {
			cache := t.TempDir()
			docker := &journalDocker{resources: make(map[string]dockermgr.OwnedResource)}
			first := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
			a := types.JobAssignment{JobID: "crash", DispatchToken: "attempt", Runtime: "docker-custom"}
			record, err := first.journal.Admit(a, "cpu")
			if err != nil {
				t.Fatal(err)
			}
			if phase != "unstarted" {
				if err := first.journal.Running(a.JobID, a.DispatchToken); err != nil {
					t.Fatal(err)
				}
				record, err = first.journal.AssignDockerDaemon(a.JobID, a.DispatchToken, "fake-daemon")
				if err != nil {
					t.Fatal(err)
				}
			}
			if phase == "after-container" {
				docker.resources[record.Resource.Name] = record.Resource
			}
			first.journal.Close()
			restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
			if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err != nil {
				t.Fatal(err)
			}
			if _, err := restarted.Start(context.Background(), a); !errors.Is(err, ErrAlreadyTracked) {
				t.Fatalf("recovered attempt was executed again: %v", err)
			}
			if docker.runs != 0 || len(docker.resources) != 0 {
				t.Fatal("crash recovery reran or leaked an attempt")
			}
		})
	}
}

func TestJournalCompletedResultRecoveredWithoutRerun(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	first := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	a := types.JobAssignment{JobID: "done", DispatchToken: "A", Runtime: "docker-custom"}
	result := first.Execute(context.Background(), a)
	if !result.Success {
		t.Fatal(result.Error)
	}
	first.journal.Close()
	restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	records := restarted.journal.Records()
	if len(records) != 1 || records[0].Result == nil || !records[0].Result.Success || records[0].ResultAccepted {
		t.Fatalf("result was lost between execution and outbox: %+v", records)
	}
	if _, err := restarted.Start(context.Background(), a); !errors.Is(err, ErrAlreadyTracked) || docker.runs != 1 {
		t.Fatal("pending completed result was rerun")
	}
	if err := restarted.journal.Accept(a.JobID, a.DispatchToken); err != nil {
		t.Fatal(err)
	}
	restarted.journal.Close()
	again := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if !again.journal.Records()[0].ResultAccepted {
		t.Fatal("result acceptance was not durable")
	}
	if _, err := again.Start(context.Background(), a); !errors.Is(err, ErrAlreadyTracked) || docker.runs != 1 {
		t.Fatal("acknowledged execution was repeated after restart")
	}
}

func TestJournalRestoredNativeAndUnavailableDockerRemainExcluded(t *testing.T) {
	for _, runtime := range []string{"ollama", "docker-custom"} {
		t.Run(runtime, func(t *testing.T) {
			cache, docker := t.TempDir(), &journalDocker{unavailable: true}
			first := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
			a := types.JobAssignment{JobID: "unresolved", DispatchToken: "A", Runtime: runtime}
			if _, err := first.journal.Admit(a, "cpu"); err != nil {
				t.Fatal(err)
			}
			if err := first.journal.Running(a.JobID, a.DispatchToken); err != nil {
				t.Fatal(err)
			}
			if runtime == "docker-custom" {
				if _, err := first.journal.AssignDockerDaemon(a.JobID, a.DispatchToken, "fake-daemon"); err != nil {
					t.Fatal(err)
				}
			}
			first.journal.Close()
			gate := &sync.RWMutex{}
			restarted := newJournalExecutor(t, cache, "gpu", gate, docker)
			if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err == nil {
				t.Fatal("unverifiable backend stop was acknowledged")
			}
			if _, err := restarted.Start(context.Background(), types.JobAssignment{
				JobID: "other", DispatchToken: "B", Runtime: "docker-custom",
			}); err == nil {
				t.Fatal("unresolved restored execution released its GPU")
			}
			sibling := newJournalExecutor(t, cache, "sibling", gate, docker)
			if _, err := sibling.Start(context.Background(), types.JobAssignment{
				JobID: "native", DispatchToken: "C", Runtime: "ollama",
			}); err == nil {
				t.Fatal("unresolved restored work allowed machine-wide native execution")
			}
			if runtime == "ollama" && !MachineRecoveryBlocked(gate) {
				t.Fatal("native recovery did not fence sibling GPUs")
			}
		})
	}
}

func TestJournalWriteCorruptionPermissionsAndOwnershipFailClosed(t *testing.T) {
	for _, failure := range []string{"write", "corrupt", "changed-proof", "permissions", "ownership", "missing-identity", "missing-record", "missing-catalog"} {
		t.Run(failure, func(t *testing.T) {
			cache, docker := t.TempDir(), &journalDocker{}
			e := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
			a := types.JobAssignment{JobID: "job", DispatchToken: "A", Runtime: "docker-custom"}
			if failure == "write" {
				if err := setJournalDirectoryWritable(e.journal.dir, false); err != nil {
					t.Fatal(err)
				}
				defer setJournalDirectoryWritable(e.journal.dir, true)
				if _, err := e.Start(context.Background(), a); err == nil || docker.runs != 0 {
					t.Fatal("runtime started without a durable journal")
				}
				return
			}
			record, err := e.journal.Admit(a, "cpu")
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(e.journal.dir, attemptKey(a.JobID, a.DispatchToken)+".json")
			switch failure {
			case "corrupt":
				err = os.WriteFile(path, []byte("{"), 0o600)
			case "changed-proof":
				var data []byte
				data, err = os.ReadFile(path)
				if err == nil {
					data = bytes.Replace(data, []byte(`"stop_verified":false`), []byte(`"stop_verified":true`), 1)
					err = os.WriteFile(path, data, 0o600)
				}
			case "permissions":
				err = makeJournalFilePublic(path)
			case "ownership":
				record.Resource.Name = "unrelated-work"
				err = atomicCheckedJSON(e.journal.dir, filepath.Base(path), record)
			case "missing-identity":
				err = os.Remove(filepath.Join(e.journal.dir, "identity.json"))
			case "missing-record":
				err = os.Remove(path)
			case "missing-catalog":
				err = os.Remove(filepath.Join(e.journal.dir, "catalog.json"))
			}
			if err != nil {
				t.Fatal(err)
			}
			e.journal.Close()
			_, err = NewExecutorWithOptions(ExecutorOptions{
				CacheDir: cache, GPUID: "gpu", Runtime: docker, OwnedTeardown: docker, ExecutionGate: &sync.RWMutex{},
			})
			if err == nil || docker.runs != 0 || docker.removed != 0 {
				t.Fatalf("unsafe recovery admitted or removed resources: %v", err)
			}
		})
	}
}

func TestJournalRejectsConcurrentOwnerAndUnknownCancellationProof(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	e := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if _, err := OpenAttemptJournal(cache, "gpu"); err == nil || !strings.Contains(err.Error(), "another process") {
		t.Fatalf("live journal ownership was not exclusive: %v", err)
	}
	if err := e.CancelAttempt("unknown", "old"); err == nil {
		t.Fatal("unknown old execution was invented as stopped")
	}
	e.journal.Close()
	restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if err := restarted.CancelAttempt("unknown", "old"); err == nil {
		t.Fatal("recovered unknown cancellation invented stop proof")
	}
	if _, err := restarted.Start(context.Background(), types.JobAssignment{
		JobID: "unknown", DispatchToken: "old", Runtime: "docker-custom",
	}); !errors.Is(err, ErrAlreadyTracked) {
		t.Fatalf("durable cancellation tombstone allowed a late assignment: %v", err)
	}
}

func TestJournalSiblingFirstReconcilesOrphanResourcesBeforeAdmission(t *testing.T) {
	for _, runtime := range []string{"docker-custom", "ollama"} {
		t.Run(runtime, func(t *testing.T) {
			cache, docker := t.TempDir(), &journalDocker{resources: make(map[string]dockermgr.OwnedResource)}
			old := newJournalExecutor(t, cache, "gpu-old", &sync.RWMutex{}, docker)
			a := types.JobAssignment{JobID: "orphan", DispatchToken: "A", Runtime: runtime}
			record, err := old.journal.Admit(a, "cpu")
			if err != nil {
				t.Fatal(err)
			}
			if err := old.journal.Running(a.JobID, a.DispatchToken); err != nil {
				t.Fatal(err)
			}
			if runtime == "docker-custom" {
				record, err = old.journal.AssignDockerDaemon(a.JobID, a.DispatchToken, "fake-daemon")
				if err != nil {
					t.Fatal(err)
				}
				docker.resources[record.Resource.Name] = record.Resource
			}
			old.journal.Close()
			gate := &sync.RWMutex{}
			sibling := newJournalExecutor(t, cache, "gpu-new", gate, docker)
			assignment := types.JobAssignment{JobID: "new-job", DispatchToken: "B", Runtime: "ollama"}
			if runtime == "ollama" {
				if _, err := sibling.Start(context.Background(), assignment); err == nil || !MachineRecoveryBlocked(gate) {
					t.Fatal("first sibling admitted work ahead of unresolved orphan native execution")
				}
			} else {
				if docker.removed != 1 || len(docker.resources) != 0 {
					t.Fatal("sibling admission did not reconcile orphan Docker ownership")
				}
				if result := sibling.Execute(context.Background(), assignment); !result.Success {
					t.Fatalf("verified orphan removal did not release machine admission: %+v", result)
				}
			}
		})
	}
}

func TestJournalKernelMachineLeaseCoordinatesIndependentMemoryGates(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	first := newJournalExecutor(t, cache, "gpu-first", &sync.RWMutex{}, docker)
	workspace := types.JobAssignment{JobID: "workspace", DispatchToken: "A", Runtime: "workspace"}
	if result := first.Execute(context.Background(), workspace); !result.Success {
		t.Fatal(result.Error)
	}

	second := newJournalExecutor(t, cache, "gpu-second", &sync.RWMutex{}, docker)
	native := types.JobAssignment{JobID: "native", DispatchToken: "B", Runtime: "ollama"}
	if _, err := second.Start(context.Background(), native); err == nil {
		t.Fatal("kernel ownership lock allowed native execution alongside another GPU's workspace")
	}
	if err := first.CancelAttempt(workspace.JobID, workspace.DispatchToken); err != nil {
		t.Fatal(err)
	}
	if result := second.Execute(context.Background(), native); !result.Success {
		t.Fatalf("verified resource stop did not release kernel admission: %+v", result)
	}
}

func TestJournalCompletionWriteFailureRetainsExclusionAndRecovers(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	e := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	docker.beforeRun = func(types.JobAssignment) error { return setJournalDirectoryWritable(e.journal.dir, false) }
	a := types.JobAssignment{JobID: "write-failure", DispatchToken: "A", Runtime: "docker-custom"}
	result := e.Execute(context.Background(), a)
	if result.Success || !strings.Contains(result.Error, "journal") {
		t.Fatalf("completion without durable proof was reported successful: %+v", result)
	}
	if err := e.CancelAttempt(a.JobID, a.DispatchToken); err == nil {
		t.Fatal("journal write failure returned positive cancellation proof")
	}
	if _, err := e.Start(context.Background(), types.JobAssignment{JobID: "next", DispatchToken: "B", Runtime: "ollama"}); err == nil {
		t.Fatal("journal write failure silently admitted more work")
	}
	docker.beforeRun = nil
	if err := setJournalDirectoryWritable(e.journal.dir, true); err != nil {
		t.Fatal(err)
	}
	crashJournalExecutor(e)
	restarted := newJournalExecutor(t, cache, "gpu", &sync.RWMutex{}, docker)
	if err := restarted.CancelAttempt(a.JobID, a.DispatchToken); err != nil {
		t.Fatalf("verified absence could not recover the interrupted completion checkpoint: %v", err)
	}
}
