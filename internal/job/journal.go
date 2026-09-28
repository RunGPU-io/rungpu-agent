package job

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/durablefs"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const maxJournalAttempts = 10000

var ErrJournalOwned = errors.New("execution journal is already owned by another process")

type AttemptRecord struct {
	Version          int                     `json:"version"`
	GPUID            string                  `json:"gpu_id"`
	JobID            string                  `json:"job_id"`
	DispatchToken    string                  `json:"dispatch_token"`
	Backend          string                  `json:"backend"`
	DockerDaemonID   string                  `json:"docker_daemon_id,omitempty"`
	Runtime          string                  `json:"runtime"`
	Phase            string                  `json:"phase"`
	KnownUnstarted   bool                    `json:"known_unstarted"`
	CancelRequested  bool                    `json:"cancel_requested"`
	StopVerified     bool                    `json:"stop_verified"`
	Resource         dockermgr.OwnedResource `json:"resource"`
	Result           *types.JobResult        `json:"result,omitempty"`
	ResultAccepted   bool                    `json:"result_accepted"`
	LegacyUnverified bool                    `json:"legacy_unverified,omitempty"`
	ReplayBlocked    bool                    `json:"replay_blocked,omitempty"`
}

type journalIdentity struct {
	Version     int    `json:"version"`
	GPUID       string `json:"gpu_id"`
	Owner       string `json:"owner"`
	Initialized bool   `json:"initialized"`
}

type journalCatalog struct {
	Version int      `json:"version"`
	GPUID   string   `json:"gpu_id"`
	Owner   string   `json:"owner"`
	Keys    []string `json:"keys"`
}

type checkedJournalJSON struct {
	Payload json.RawMessage `json:"payload"`
	SHA256  string          `json:"sha256"`
}

type AttemptJournal struct {
	mu        sync.Mutex
	dir       string
	identity  journalIdentity
	records   map[string]AttemptRecord
	recovered map[string]bool
	fault     error
	lockFile  *os.File
	catalog   map[string]bool
}

func attemptKey(id, token string) string {
	sum := sha256.Sum256([]byte(id + "\x00" + token))
	return hex.EncodeToString(sum[:])
}

func readJournalJSON(path string, value interface{}) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if err := validatePrivateJournalFile(path, info); err != nil {
		return err
	}
	if info.Size() > 16*1024*1024 {
		return fmt.Errorf("journal file has unsafe type, permissions, or size: %s", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, value); err != nil {
		return fmt.Errorf("corrupt journal %s: %w", path, err)
	}
	return nil
}

func atomicJournalJSON(dir, name string, value interface{}) error {
	if err := privateDirectory(dir); err != nil {
		return err
	}
	path := filepath.Join(dir, name)
	if info, err := os.Lstat(path); err == nil {
		if err := validatePrivateJournalFile(path, info); err != nil {
			return err
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	file, err := createJournalStage(dir)
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return durablefs.Replace(file.Name(), path)
}

func atomicCheckedJSON(dir, name string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return atomicJournalJSON(dir, name, checkedJournalJSON{
		Payload: data, SHA256: fmt.Sprintf("%x", sha256.Sum256(data)),
	})
}

func readCheckedJSON(path string, value interface{}) error {
	var checked checkedJournalJSON
	if err := readJournalJSON(path, &checked); err != nil {
		return err
	}
	if len(checked.Payload) == 0 || checked.SHA256 != fmt.Sprintf("%x", sha256.Sum256(checked.Payload)) {
		return fmt.Errorf("journal integrity check failed: %s", path)
	}
	if err := json.Unmarshal(checked.Payload, value); err != nil {
		return fmt.Errorf("corrupt journal payload: %w", err)
	}
	return nil
}

func (j *AttemptJournal) saveCatalog() error {
	keys := make([]string, 0, len(j.catalog))
	for key := range j.catalog {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return atomicCheckedJSON(j.dir, "catalog.json", journalCatalog{
		Version: 1, GPUID: j.identity.GPUID, Owner: j.identity.Owner, Keys: keys,
	})
}

func OpenAttemptJournal(cacheDir, gpuID string) (*AttemptJournal, error) {
	return openAttemptJournal(cacheDir, gpuID, "")
}

func openAttemptJournal(cacheDir, gpuID, expectedRoot string) (*AttemptJournal, error) {
	rootIdentity, releaseRoot, err := openJournalRoot(cacheDir, expectedRoot)
	if err != nil {
		return nil, err
	}
	defer releaseRoot()
	root := filepath.Join(cacheDir, "executions")
	if err := privateDirectory(root); err != nil {
		return nil, err
	}
	j := &AttemptJournal{
		dir:     filepath.Join(root, attemptKey(gpuID, "")),
		records: make(map[string]AttemptRecord), recovered: make(map[string]bool), catalog: make(map[string]bool),
	}
	expectedOwner := rootIdentity.Scopes[attemptKey(gpuID, "")]
	if expectedOwner != "" {
		if _, err := os.Lstat(filepath.Join(j.dir, "identity.json")); err != nil {
			return nil, fmt.Errorf("registered GPU execution journal is missing: %w", err)
		}
	}
	if err := privateDirectory(j.dir); err != nil {
		return nil, err
	}
	j.lockFile, err = openJournalLockFile(filepath.Join(j.dir, ".lock"))
	if err != nil {
		return nil, err
	}
	opened := false
	defer func() {
		if !opened {
			j.lockFile.Close()
		}
	}()
	lockInfo, err := os.Lstat(filepath.Join(j.dir, ".lock"))
	openedInfo, openedErr := j.lockFile.Stat()
	if err != nil || openedErr != nil || validatePrivateJournalFile(filepath.Join(j.dir, ".lock"), lockInfo) != nil ||
		!os.SameFile(lockInfo, openedInfo) {
		return nil, fmt.Errorf("journal lock has unsafe type or permissions")
	}
	if err := lockJournalFile(j.lockFile); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrJournalOwned, err)
	}
	identityPath := filepath.Join(j.dir, "identity.json")
	if err := readCheckedJSON(identityPath, &j.identity); os.IsNotExist(err) {
		entries, scanErr := os.ReadDir(j.dir)
		if scanErr != nil {
			return nil, fmt.Errorf("journal identity missing from nonempty journal")
		}
		for _, entry := range entries {
			if entry.Name() != ".lock" && !strings.HasPrefix(entry.Name(), ".stage-") {
				return nil, fmt.Errorf("journal identity missing from nonempty journal")
			}
		}
		var owner [32]byte
		if _, err := rand.Read(owner[:]); err != nil {
			return nil, err
		}
		j.identity = journalIdentity{Version: 1, GPUID: gpuID, Owner: hex.EncodeToString(owner[:])}
		if err := atomicCheckedJSON(j.dir, "identity.json", j.identity); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	if j.identity.Version != 1 || j.identity.GPUID != gpuID || len(j.identity.Owner) != 64 ||
		(expectedOwner != "" && expectedOwner != j.identity.Owner) {
		return nil, fmt.Errorf("journal identity does not match this GPU")
	}
	if _, err := hex.DecodeString(j.identity.Owner); err != nil {
		return nil, fmt.Errorf("corrupt journal owner")
	}
	entries, err := os.ReadDir(j.dir)
	if err != nil {
		return nil, err
	}
	if !j.identity.Initialized {
		for _, entry := range entries {
			if entry.Name() != "identity.json" && entry.Name() != "catalog.json" &&
				entry.Name() != ".lock" && !strings.HasPrefix(entry.Name(), ".stage-") {
				return nil, fmt.Errorf("uninitialized journal contains execution records")
			}
		}
		if err := j.saveCatalog(); err != nil {
			return nil, err
		}
		j.identity.Initialized = true
		if err := atomicCheckedJSON(j.dir, "identity.json", j.identity); err != nil {
			return nil, err
		}
	}
	var catalog journalCatalog
	if err := readCheckedJSON(filepath.Join(j.dir, "catalog.json"), &catalog); err != nil {
		return nil, err
	}
	if catalog.Version != 1 || catalog.GPUID != gpuID || catalog.Owner != j.identity.Owner || len(catalog.Keys) > maxJournalAttempts {
		return nil, fmt.Errorf("journal catalog identity or capacity mismatch")
	}
	for _, key := range catalog.Keys {
		if decoded, err := hex.DecodeString(key); err != nil || len(decoded) != 32 || j.catalog[key] {
			return nil, fmt.Errorf("invalid journal catalog key")
		}
		j.catalog[key] = true
	}
	for _, entry := range entries {
		if entry.Name() == "identity.json" || entry.Name() == "catalog.json" || entry.Name() == ".lock" || strings.HasPrefix(entry.Name(), ".stage-") {
			continue
		}
		var record AttemptRecord
		if err := readCheckedJSON(filepath.Join(j.dir, entry.Name()), &record); err != nil {
			return nil, err
		}
		if err := j.validate(record); err != nil {
			return nil, err
		}
		key := attemptKey(record.JobID, record.DispatchToken)
		if entry.Name() != key+".json" || len(j.records) >= maxJournalAttempts {
			return nil, fmt.Errorf("invalid journal filename or journal capacity exceeded")
		}
		j.records[key], j.recovered[key] = record, true
	}
	for key := range j.catalog {
		if _, exists := j.records[key]; !exists {
			return nil, fmt.Errorf("journal catalog references a missing execution record")
		}
	}
	catalogChanged := false
	for key, record := range j.records {
		if !j.catalog[key] {
			if (record.Phase != "unstarted" && record.Phase != "unknown") || record.StopVerified || record.ResultAccepted {
				return nil, fmt.Errorf("executed attempt is missing from journal catalog")
			}
			j.catalog[key], catalogChanged = true, true
		}
	}
	if catalogChanged {
		if err := j.saveCatalog(); err != nil {
			return nil, err
		}
	}
	if expectedOwner == "" {
		if len(j.records) != 0 {
			return nil, fmt.Errorf("unregistered GPU journal contains execution records")
		}
		rootIdentity.Scopes[attemptKey(gpuID, "")] = j.identity.Owner
		if err := atomicCheckedJSON(root, "machine.json", rootIdentity); err != nil {
			return nil, err
		}
	}
	opened = true
	return j, nil
}

func (j *AttemptJournal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.fault = fmt.Errorf("execution journal is closed")
	return j.lockFile.Close()
}

func (j *AttemptJournal) machineLease(runtime string) (func(), error) {
	path := filepath.Join(filepath.Dir(j.dir), ".machine.lock")
	file, err := openJournalLockFile(path)
	if err != nil {
		return nil, err
	}
	info, statErr := os.Lstat(path)
	opened, openErr := file.Stat()
	if statErr != nil || openErr != nil || validatePrivateJournalFile(path, info) != nil ||
		!os.SameFile(info, opened) {
		file.Close()
		return nil, fmt.Errorf("machine execution lock has unsafe ownership")
	}
	if err := lockMachineFile(file, dockerRuntimeName(runtime)); err != nil {
		file.Close()
		return nil, fmt.Errorf("machine execution ownership is busy: %w", err)
	}
	return func() { _ = file.Close() }, nil
}

func dockerRuntimeName(runtime string) bool {
	return runtime == "docker-custom" || runtime == "comfyui-batch" || runtime == "workspace"
}

func (j *AttemptJournal) resource(record AttemptRecord) dockermgr.OwnedResource {
	if !dockerRuntimeName(record.Runtime) {
		return dockermgr.OwnedResource{}
	}
	identity := j.identity.Owner + "\x00" + record.GPUID + "\x00" + record.Backend + "\x00" + record.Runtime + "\x00" + record.DockerDaemonID
	key := attemptKey(record.JobID, record.DispatchToken)
	return dockermgr.OwnedResource{
		Name:     "tokenize-attempt-" + attemptKey(identity, key),
		DaemonID: record.DockerDaemonID,
		Labels: map[string]string{
			"ai.rungpu.owner": j.identity.Owner, "ai.rungpu.gpu": record.GPUID,
			"ai.rungpu.job": record.JobID, "ai.rungpu.attempt": key,
			"ai.rungpu.backend": record.Backend, "ai.rungpu.runtime": record.Runtime,
		},
	}
}

func (j *AttemptJournal) validate(record AttemptRecord) error {
	if record.Version != 1 || record.GPUID != j.identity.GPUID || !types.ValidJobID(record.JobID) ||
		(record.Runtime != "" && record.Runtime != "ollama" && !dockerRuntimeName(record.Runtime)) {
		return fmt.Errorf("invalid journal attempt identity")
	}
	switch record.Phase {
	case "unknown", "unstarted", "running", "finished":
	default:
		return fmt.Errorf("invalid journal execution phase")
	}
	if (record.Phase == "running" && (record.Runtime == "" || record.KnownUnstarted)) ||
		(record.Phase == "unknown" && (record.KnownUnstarted || record.StopVerified)) ||
		(record.LegacyUnverified && (record.Runtime != "" || record.KnownUnstarted || record.StopVerified)) ||
		(record.StopVerified && record.ReplayBlocked) ||
		!reflect.DeepEqual(record.Resource, j.resource(record)) {
		return fmt.Errorf("inconsistent journal execution ownership")
	}
	if record.Result != nil && (record.Result.JobID != record.JobID ||
		record.Result.DispatchToken != record.DispatchToken || record.Result.GPUID != record.GPUID) {
		return fmt.Errorf("journal result does not match execution identity")
	}
	return nil
}

func (j *AttemptJournal) store(record AttemptRecord) error {
	if j.fault != nil {
		return j.fault
	}
	err := j.validate(record)
	if err == nil {
		err = atomicCheckedJSON(j.dir, attemptKey(record.JobID, record.DispatchToken)+".json", record)
	}
	key := attemptKey(record.JobID, record.DispatchToken)
	if err == nil && !j.catalog[key] {
		j.catalog[key] = true
		err = j.saveCatalog()
	}
	if err != nil {
		j.fault = fmt.Errorf("execution journal is fenced: %w", err)
		return j.fault
	}
	j.records[key] = record
	return nil
}

func (j *AttemptJournal) fresh(id, token string) (AttemptRecord, error) {
	if j.fault != nil {
		return AttemptRecord{}, j.fault
	}
	if !types.ValidJobID(id) || len(j.records) >= maxJournalAttempts {
		return AttemptRecord{}, fmt.Errorf("invalid attempt ID or execution journal capacity reached")
	}
	return AttemptRecord{Version: 1, GPUID: j.identity.GPUID, JobID: id, DispatchToken: token}, nil
}

func (j *AttemptJournal) Err() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.fault
}

func (j *AttemptJournal) Records() []AttemptRecord {
	j.mu.Lock()
	defer j.mu.Unlock()
	records := make([]AttemptRecord, 0, len(j.records))
	for _, record := range j.records {
		records = append(records, record)
	}
	return records
}

func (j *AttemptJournal) Find(id, token string) (AttemptRecord, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	return record, exists
}

func (j *AttemptJournal) RememberLegacyResult(result types.JobResult) (AttemptRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if record, exists := j.records[attemptKey(result.JobID, result.DispatchToken)]; exists {
		return record, nil
	}
	record, err := j.fresh(result.JobID, result.DispatchToken)
	if err != nil {
		return record, err
	}
	record.Phase, record.LegacyUnverified, record.Result, record.ReplayBlocked = "unknown", true, &result, true
	return record, j.store(record)
}

func (j *AttemptJournal) Admit(a types.JobAssignment, backend string) (AttemptRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if a.Runtime != "ollama" && !dockerRuntimeName(a.Runtime) {
		return AttemptRecord{}, fmt.Errorf("unsupported execution runtime %q", a.Runtime)
	}
	key := attemptKey(a.JobID, a.DispatchToken)
	record, exists := j.records[key]
	if exists && (j.recovered[key] || record.CancelRequested || !record.KnownUnstarted || record.Result != nil) {
		return record, ErrAlreadyTracked
	}
	if !exists {
		var err error
		record, err = j.fresh(a.JobID, a.DispatchToken)
		if err != nil {
			return record, err
		}
	}
	record.Runtime, record.Backend = a.Runtime, backend
	record.DockerDaemonID = a.ResourceDaemonID
	record.Phase, record.KnownUnstarted = "unstarted", true
	record.Resource = j.resource(record)
	return record, j.store(record)
}

func (j *AttemptJournal) Running(id, token string) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	if !exists || !record.KnownUnstarted || record.CancelRequested {
		return fmt.Errorf("attempt cannot enter runtime")
	}
	record.Phase, record.KnownUnstarted = "running", false
	return j.store(record)
}

func (j *AttemptJournal) Complete(id, token string, result *types.JobResult, stopped bool) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	if !exists {
		return fmt.Errorf("execution has no durable journal record")
	}
	record.Phase, record.StopVerified = "finished", stopped
	if result != nil {
		record.Result = result
	}
	healthyWorkspace := record.Runtime == "workspace" && record.Result != nil && record.Result.Success
	record.ReplayBlocked = !stopped && !healthyWorkspace && !record.KnownUnstarted
	return j.store(record)
}

func (j *AttemptJournal) RememberCancellation(id, token string, knownUnstarted bool) (AttemptRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	if !exists {
		var err error
		record, err = j.fresh(id, token)
		if err != nil {
			return record, err
		}
		record.Phase = "unknown"
	}
	if knownUnstarted && !record.KnownUnstarted {
		return record, fmt.Errorf("attempt has no durable never-started proof")
	}
	record.CancelRequested = true
	if dockerRuntimeName(record.Runtime) && record.DockerDaemonID == "" && record.Phase == "running" {
		record.Phase, record.KnownUnstarted = "unstarted", true
	}
	return record, j.store(record)
}

func (j *AttemptJournal) RememberResult(result types.JobResult) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(result.JobID, result.DispatchToken)]
	if exists && record.Result != nil {
		return j.fault
	}
	if !exists {
		var err error
		record, err = j.fresh(result.JobID, result.DispatchToken)
		if err != nil {
			return err
		}
		record.Phase, record.KnownUnstarted = "unstarted", true
	}
	record.Result = &result
	if record.Phase == "unknown" && !record.StopVerified {
		record.ReplayBlocked = true
	}
	return j.store(record)
}

func (j *AttemptJournal) BlockReplay(id, token string, blocked bool) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	if !exists {
		return fmt.Errorf("attempt record missing during recovery")
	}
	if record.ReplayBlocked == blocked {
		return j.fault
	}
	record.ReplayBlocked = blocked
	return j.store(record)
}

func (j *AttemptJournal) Accept(id, token string) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	if !exists {
		return j.fault
	}
	record.ResultAccepted = true
	return j.store(record)
}
