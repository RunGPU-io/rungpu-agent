package job

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

var recoveryGates sync.Map
var restoredLeaseMu sync.Mutex
var restoredLeases = make(map[string]func())

func MachineRecoveryBlocked(gate *sync.RWMutex) bool {
	if gate == nil {
		return false
	}
	_, blocked := recoveryGates.Load(gate)
	return blocked
}

func (e *Executor) restoreLease(record AttemptRecord) (func(), error) {
	key := e.journal.dir + "/" + attemptKey(record.JobID, record.DispatchToken)
	restoredLeaseMu.Lock()
	defer restoredLeaseMu.Unlock()
	if e.executionGate != nil && (record.Runtime == "ollama" || record.LegacyUnverified) {
		recoveryGates.Store(e.executionGate, true)
	}
	release := func() {
		restoredLeaseMu.Lock()
		defer restoredLeaseMu.Unlock()
		if unlock := restoredLeases[key]; unlock != nil {
			delete(restoredLeases, key)
			unlock()
		}
	}
	if restoredLeases[key] != nil {
		return release, nil
	}
	var gateRelease func()
	if e.executionGate != nil {
		if dockerRuntimeName(record.Runtime) && e.executionGate.TryRLock() {
			gateRelease = e.executionGate.RUnlock
		} else if !dockerRuntimeName(record.Runtime) && e.executionGate.TryLock() {
			gateRelease = e.executionGate.Unlock
		} else {
			recoveryGates.Store(e.executionGate, true)
			return nil, fmt.Errorf("machine recovery conflicts with an existing execution lease")
		}
	}
	fileRelease, err := e.journal.machineLease(record.Runtime)
	if err != nil {
		if gateRelease != nil {
			gateRelease()
		}
		return nil, err
	}
	restoredLeases[key] = func() {
		fileRelease()
		if gateRelease != nil {
			gateRelease()
		}
	}
	return release, nil
}

func (e *Executor) restoreOrphanSiblingJournals() error {
	root := filepath.Dir(e.journal.dir)
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.Name() == ".machine.lock" || entry.Name() == ".identity.lock" || entry.Name() == "machine.json" ||
			strings.HasPrefix(entry.Name(), ".stage-") || entry.Name() == filepath.Base(e.journal.dir) {
			continue
		}
		if !entry.IsDir() {
			return fmt.Errorf("unexpected entry in execution journal root")
		}
		var identity journalIdentity
		if err := readCheckedJSON(filepath.Join(root, entry.Name(), "identity.json"), &identity); err != nil {
			return err
		}
		if entry.Name() != attemptKey(identity.GPUID, "") {
			return fmt.Errorf("sibling journal GPU identity mismatch")
		}
		journal, err := OpenAttemptJournal(filepath.Dir(root), identity.GPUID)
		if errors.Is(err, ErrJournalOwned) {
			continue
		}
		if err != nil {
			return err
		}
		peer := &Executor{
			gpuID: identity.GPUID, journal: journal, ownedTeardown: e.ownedTeardown,
			executionGate: e.executionGate, workspaces: make(map[string]bool),
		}
		err = peer.restoreJournal()
		journal.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func (e *Executor) restoreJournal() error {
	e.executions = make(map[string]*execution)
	for _, record := range e.journal.Records() {
		if record.LegacyUnverified {
			if err := e.TrackUnverifiedLegacy(record); err != nil {
				return err
			}
			continue
		}
		if record.Phase == "unknown" || record.StopVerified {
			continue
		}

		if record.KnownUnstarted || (dockerRuntimeName(record.Runtime) && record.DockerDaemonID == "") {
			if err := e.journal.recoverUnstarted(record); err != nil {
				return err
			}
			continue
		}
		state := &execution{
			dispatchToken: record.DispatchToken, runtime: record.Runtime, resource: record.Resource,
			cancel: func() {}, done: make(chan struct{}), stopUnverified: true, restored: true,
			cancelRequested: record.CancelRequested,
		}
		close(state.done)
		var err error
		state.release, err = e.restoreLease(record)
		if err != nil {
			return err
		}
		if dockerRuntimeName(record.Runtime) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			var err error
			running := false
			if record.Runtime == "workspace" {
				running, err = e.ownedTeardown.InspectOwned(ctx, record.Resource)
			}
			if err == nil && (!running || record.CancelRequested || record.Result == nil || !record.Result.Success) {
				err = e.ownedTeardown.RemoveOwnedAndVerify(ctx, record.Resource)
				running = false
			}
			cancel()
			if err == nil && !running {
				result := record.Result
				if result == nil || (record.Runtime == "workspace" && !record.ResultAccepted && result.Success) {
					result = &types.JobResult{Type: "job_result", JobID: record.JobID, DispatchToken: record.DispatchToken,
						GPUID: e.gpuID, Error: "agent restarted; prior owned Docker execution was stopped"}
				}
				if err := e.journal.Complete(record.JobID, record.DispatchToken, result, true); err != nil {
					return err
				}
				e.releaseExecution(state)
				continue
			}
			if err := e.journal.BlockReplay(record.JobID, record.DispatchToken, err != nil); err != nil {
				return err
			}
			if err != nil {
				fmt.Printf("[recovery] GPU %s: owned Docker stop is unverified; admission remains reserved: %v\n", e.gpuID, err)
			}
		} else if err := e.journal.BlockReplay(record.JobID, record.DispatchToken, true); err != nil {
			return err
		} else {
			fmt.Printf("[recovery] GPU %s: native execution cannot be verified stopped; the machine remains reserved\n", e.gpuID)
		}
		if e.executions[record.JobID] != nil {
			return fmt.Errorf("multiple unresolved journal attempts for job %s", record.JobID)
		}
		e.executions[record.JobID] = state
		if record.Runtime == "workspace" {
			e.workspaces[record.JobID] = true
		}
	}
	return nil
}

func (e *Executor) TrackUnverifiedLegacy(record AttemptRecord) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !record.LegacyUnverified || record.GPUID != e.gpuID {
		return nil
	}
	if e.executions[record.JobID] == nil {
		state := &execution{
			dispatchToken: record.DispatchToken, runtime: "legacy-unverified",
			cancel: func() {}, done: make(chan struct{}), stopUnverified: true, restored: true,
		}
		close(state.done)
		var err error
		state.release, err = e.restoreLease(record)
		if err != nil {
			return err
		}
		e.executions[record.JobID] = state
	}
	return nil
}
