package job

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type Executor struct {
	runtime       Runtime
	gpuID         string
	backend       string
	cacheDir      string
	jobTimeout    time.Duration
	OnProgress    func(types.JobProgress)
	executionGate *sync.RWMutex
	journal       *AttemptJournal
	ownedTeardown dockermgr.OwnedResourceRemover

	mu         sync.Mutex
	inflight   map[string]context.CancelFunc
	workspaces map[string]bool
	teardown   interface {
		RemoveAndVerify(context.Context, string) error
	}
	executions map[string]*execution
}

type execution struct {
	dispatchToken   string
	cancel          context.CancelFunc
	done            chan struct{}
	runtime         string
	cancelRequested bool
	stopConfirmed   bool
	stopUnverified  bool
	release         func()
	resource        dockermgr.OwnedResource
	restored        bool
}

var ErrAlreadyTracked = errors.New("job is already tracked")

type ExecutorOptions struct {
	CacheDir           string
	MaxCacheGB         int
	GPUID              string
	Backend            string
	GPUDevice          string
	JobTimeout         time.Duration
	MaxCustomFileBytes int64
	Policy             dockermgr.SecurityPolicy
	HFToken            string
	ExecutionGate      *sync.RWMutex
	Runtime            Runtime
	OwnedTeardown      dockermgr.OwnedResourceRemover
	ExpectedJournalID  string
}

func NewExecutor(cacheDir string, maxCacheGB int, gpuID, backend string) (*Executor, error) {
	return NewExecutorWithOptions(ExecutorOptions{
		CacheDir: cacheDir, MaxCacheGB: maxCacheGB, GPUID: gpuID, Backend: backend,
	})
}

func NewExecutorWithOptions(o ExecutorOptions) (*Executor, error) {
	if o.JobTimeout <= 0 {
		o.JobTimeout = 60 * time.Minute
	}
	rt := o.Runtime
	if rt == nil {
		var err error
		rt, err = NewRuntimeOpts(o.Backend, o.CacheDir, o.MaxCacheGB, RuntimeOptions{
			GPUDevice: o.GPUDevice, JobTimeout: o.JobTimeout, Policy: o.Policy,
		})
		if err != nil {
			return nil, err
		}
	}
	SetMaxDownloadBytes(o.MaxCustomFileBytes)
	SetHFToken(o.HFToken)
	e := &Executor{
		runtime:       rt,
		gpuID:         o.GPUID,
		backend:       o.Backend,
		cacheDir:      o.CacheDir,
		jobTimeout:    o.JobTimeout,
		inflight:      map[string]context.CancelFunc{},
		workspaces:    map[string]bool{},
		teardown:      dockermgr.New(),
		executionGate: o.ExecutionGate,
		ownedTeardown: o.OwnedTeardown,
	}
	if e.ownedTeardown == nil {
		e.ownedTeardown = dockermgr.New()
	}
	var err error
	e.journal, err = openAttemptJournal(o.CacheDir, o.GPUID, o.ExpectedJournalID)
	if err != nil {
		return nil, fmt.Errorf("open execution journal: %w", err)
	}
	if err := e.restoreJournal(); err != nil {
		e.journal.Close()
		return nil, err
	}
	if err := e.restoreOrphanSiblingJournals(); err != nil {
		e.journal.Close()
		return nil, err
	}
	return e, nil
}

func (e *Executor) Journal() *AttemptJournal { return e.journal }

func (e *Executor) Backend() string { return e.backend }
func (e *Executor) Runtime() string { return e.runtime.Name() }

func (e *Executor) IsTracked(jobID string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.executions[jobID] != nil
}

func (e *Executor) TrackedDispatchToken(jobID string) (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	state := e.executions[jobID]
	if state == nil {
		return "", false
	}
	return state.dispatchToken, true
}

func (e *Executor) ReservedJobs() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	count := 0
	for id, state := range e.executions {
		if state.release != nil || state.stopUnverified || e.workspaces[id] || e.inflight[id] != nil {
			count++
		}
	}
	return count
}

func (e *Executor) WorkspaceIDs() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	ids := make([]string, 0, len(e.workspaces))
	for id := range e.workspaces {
		ids = append(ids, id)
	}
	return ids
}

func (e *Executor) Execute(ctx context.Context, a types.JobAssignment) types.JobResult {
	results, err := e.Start(ctx, a)
	if err != nil {
		return types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken, GPUID: e.gpuID, Error: err.Error()}
	}
	return <-results
}

func (e *Executor) Start(ctx context.Context, a types.JobAssignment) (<-chan types.JobResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.executions == nil {
		e.executions = make(map[string]*execution)
	}
	if e.journal != nil {
		if err := e.journal.Err(); err != nil {
			return nil, err
		}
		if a.Runtime != "ollama" && !dockerRuntimeName(a.Runtime) {
			return nil, fmt.Errorf("unsupported execution runtime %q", a.Runtime)
		}
	}
	if MachineRecoveryBlocked(e.executionGate) {
		return nil, fmt.Errorf("machine is reserved by unresolved execution recovery")
	}
	if _, exists := e.executions[a.JobID]; exists {
		return nil, fmt.Errorf("%w: %s", ErrAlreadyTracked, a.JobID)
	}
	var release func()
	for _, state := range e.executions {
		if state.stopUnverified || state.release != nil {
			return nil, fmt.Errorf("GPU is already running a job or its stop is unverified")
		}
	}
	if e.executionGate != nil {
		for _, state := range e.executions {
			if state.release != nil {
				return nil, fmt.Errorf("GPU is already running a job or its stop is unverified")
			}
		}
		switch a.Runtime {
		case "ollama":
			if !e.executionGate.TryLock() {
				return nil, fmt.Errorf("machine is busy: native Ollama requires all sibling GPUs idle")
			}
			release = e.executionGate.Unlock
		case "docker-custom", "comfyui-batch", "workspace":
			if !e.executionGate.TryRLock() {
				return nil, fmt.Errorf("machine is busy with native Ollama or maintenance")
			}
			release = e.executionGate.RUnlock
		default:
			return nil, fmt.Errorf("unsupported execution runtime %q", a.Runtime)
		}
	}
	if e.journal != nil {
		machineRelease, err := e.journal.machineLease(a.Runtime)
		if err != nil {
			if release != nil {
				release()
			}
			return nil, err
		}
		gateRelease := release
		release = func() {
			machineRelease()
			if gateRelease != nil {
				gateRelease()
			}
		}
	}
	jobCtx, cancel := context.WithCancel(ctx)
	state := &execution{dispatchToken: a.DispatchToken, cancel: cancel, done: make(chan struct{}), runtime: a.Runtime, release: release}
	if e.journal != nil {
		record, err := e.journal.Admit(a, e.backend)
		if err != nil {
			cancel()
			if release != nil {
				release()
			}
			return nil, err
		}
		state.resource = record.Resource
		a.ResourceName, a.ResourceLabels = record.Resource.Name, record.Resource.Labels
	}
	e.executions[a.JobID] = state
	if e.inflight == nil {
		e.inflight = make(map[string]context.CancelFunc)
	}
	e.inflight[a.JobID] = cancel
	results := make(chan types.JobResult, 1)
	go func() {
		defer cancel()
		var result types.JobResult
		var enterErr error
		if e.journal != nil {
			enterErr = e.journal.Running(a.JobID, a.DispatchToken)
		}
		if enterErr != nil {
			result = types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken, GPUID: e.gpuID, Error: enterErr.Error()}
		} else {
			result = e.execute(jobCtx, a)
		}
		e.mu.Lock()
		delete(e.inflight, a.JobID)
		stopped := !e.workspaces[a.JobID] && !state.cancelRequested && !state.stopUnverified
		if e.journal != nil {
			if err := e.journal.Complete(a.JobID, a.DispatchToken, &result, stopped); err != nil {
				state.stopUnverified = true
				result.Success, result.Error = false, err.Error()
				stopped = false
			}
		}
		if stopped {
			if e.executionGate != nil || e.journal != nil {
				state.stopConfirmed = true
			}
			e.releaseExecution(state)
		}
		close(state.done)
		e.mu.Unlock()
		results <- result
	}()
	return results, nil
}

func (e *Executor) execute(ctx context.Context, a types.JobAssignment) types.JobResult {
	start := time.Now()
	res := types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken, GPUID: e.gpuID}

	workspace := a.Runtime == "workspace"
	if !workspace {
		jobTimeout := jobTimeoutForAssignment(a, e.jobTimeout)
		var timeoutCancel context.CancelFunc
		ctx, timeoutCancel = context.WithTimeout(ctx, jobTimeout)
		defer timeoutCancel()
	}
	if err := ctx.Err(); err != nil {
		res.Error = err.Error()
		return res
	}

	stagingDir := e.cacheDir + "/staging/" + a.JobID
	if a.WorkflowJSON != "" {
		if err := StageInlineWorkflow(a.WorkflowJSON, a.WorkflowSHA256, stagingDir); err != nil {
			res.Success = false
			res.Error = "inline workflow staging failed: " + err.Error()
			res.DurationMS = elapsedMilliseconds(start)
			return res
		}
	}
	if len(a.CustomFiles) > 0 {
		e.progress(a, "downloading_files", 0, "Downloading custom files...")
		if err := DownloadCustomFilesCached(ctx, a.CustomFiles, stagingDir, e.cacheDir+"/assets", func(stage string, pct float64, msg string) {
			e.progress(a, stage, pct, msg)
		}); err != nil {
			res.Success = false
			res.Error = "custom file download failed: " + err.Error()
			res.DurationMS = elapsedMilliseconds(start)
			return res
		}
	}

	e.progress(a, "pulling_image", 0, "Preparing model...")
	if err := e.runtime.Prepare(ctx, a); err != nil {
		res.Success = false
		res.Error = "model preparation failed: " + err.Error()
		res.DurationMS = elapsedMilliseconds(start)
		return res
	}
	if e.journal != nil && dockerRuntimeName(a.Runtime) {
		daemon, err := e.ownedTeardown.DaemonID(ctx)
		if err != nil {
			res.Error = "Docker backend identity failed: " + err.Error()
			return res
		}
		e.mu.Lock()
		record, err := e.journal.AssignDockerDaemon(a.JobID, a.DispatchToken, daemon)
		if err != nil {
			e.mu.Unlock()
			res.Error = err.Error()
			return res
		}
		e.executions[a.JobID].resource = record.Resource
		e.mu.Unlock()
		a.ResourceName, a.ResourceLabels, a.ResourceDaemonID = record.Resource.Name, record.Resource.Labels, record.DockerDaemonID
	}

	e.progress(a, "running", 0.5, "Running inference...")
	out, err := e.runtime.Run(ctx, a)
	if workspace && err == nil {

		e.mu.Lock()
		if e.workspaces == nil {
			e.workspaces = make(map[string]bool)
		}
		e.workspaces[a.JobID] = true
		e.mu.Unlock()
	}
	if e.executionGate != nil || e.journal != nil {
		if a.Runtime == "ollama" && err != nil {
			e.markStopUnverified(a.JobID)
			err = fmt.Errorf("%w; native backend stop cannot be verified; machine remains reserved", err)
		} else if a.Runtime != "ollama" && (!workspace || err != nil) {
			stopCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			stopErr := e.removeContainers(stopCtx, a.JobID)
			cancel()
			if stopErr != nil {
				e.markStopUnverified(a.JobID)
				err = errors.Join(err, fmt.Errorf("job teardown could not be verified: %w", stopErr))
			}
		}
	}
	res.DurationMS = elapsedMilliseconds(start)
	if err != nil {
		res.Success = false
		res.Error = err.Error()
		return res
	}

	if a.UploadURL != "" {
		outputPath, ok := out["output_file"].(string)
		if !ok || strings.TrimSpace(outputPath) == "" {
			res.Error = "output upload failed: required output_file is missing or empty"
			return res
		}
		if info, statErr := os.Stat(outputPath); statErr == nil && info.Size() == 0 {
			res.Error = "output upload failed: required output_file is empty"
			return res
		}
		e.progress(a, "uploading", 0.9, "Uploading output...")
		if uploadErr := UploadOutput(ctx, outputPath, a.UploadURL); uploadErr != nil {
			res.Error = "output upload failed: " + uploadErr.Error()
			return res
		}
		out["uploaded"] = true
		if info, statErr := os.Stat(outputPath); statErr == nil {
			out["output_size_bytes"] = info.Size()
			out["output_content_type"] = outputContentType(outputPath)
		}
	}

	e.progress(a, "completed", 1.0, "Done")

	res.Success = true
	if out != nil {
		if e.backend != "" {
			out["backend"] = e.backend
		}
		res.Result = out
	} else {
		res.Result = map[string]interface{}{"status": "completed"}
		if e.backend != "" {
			res.Result["backend"] = e.backend
		}
	}
	return res
}

func elapsedMilliseconds(start time.Time) int64 {
	elapsed := time.Since(start)
	return (elapsed.Nanoseconds() + int64(time.Millisecond) - 1) / int64(time.Millisecond)
}

func (e *Executor) progress(a types.JobAssignment, stage string, pct float64, msg string) {
	if e.OnProgress != nil {
		e.OnProgress(types.JobProgress{
			Type:          "job_progress",
			JobID:         a.JobID,
			DispatchToken: a.DispatchToken,
			GPUID:         e.gpuID,
			Stage:         stage,
			Progress:      pct,
			Message:       msg,
		})
	}
}

func (e *Executor) CleanupModels(force bool) error {
	return e.runtime.Cleanup(force)
}

func (e *Executor) ForgetExecution(jobID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	state := e.executions[jobID]
	if state == nil || state.release != nil || state.stopUnverified || e.workspaces[jobID] || (state.cancelRequested && !state.stopConfirmed) {
		return
	}
	select {
	case <-state.done:
		delete(e.executions, jobID)
	default:
	}
}

func (e *Executor) releaseExecution(state *execution) {
	if state.release != nil {
		state.release()
		state.release = nil
	}
}

func (e *Executor) markStopUnverified(jobID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.executions[jobID].stopUnverified = true
}

func (e *Executor) removeContainers(ctx context.Context, jobID string) error {
	e.mu.Lock()
	if state := e.executions[jobID]; state != nil && state.resource.Name != "" {
		owner, remover := state.resource, e.ownedTeardown
		e.mu.Unlock()
		return remover.RemoveOwnedAndVerify(ctx, owner)
	}
	if e.teardown == nil {
		e.teardown = dockermgr.New()
	}
	teardown := e.teardown
	e.mu.Unlock()
	var errs []error
	for _, name := range []string{CustomContainerName(jobID), WorkspaceContainerName(jobID)} {
		if err := teardown.RemoveAndVerify(ctx, name); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (e *Executor) Cancel(jobID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return e.CancelContext(ctx, jobID)
}

func (e *Executor) CancelAttempt(jobID, dispatchToken string) error {
	if dispatchToken == "" {
		return fmt.Errorf("dispatch token is required for attempt-scoped cancellation")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return e.cancelContext(ctx, jobID, &dispatchToken)
}

func (e *Executor) CancelContext(ctx context.Context, jobID string) error {
	return e.cancelContext(ctx, jobID, nil)
}

func (e *Executor) cancelContext(ctx context.Context, jobID string, dispatchToken *string) error {
	e.mu.Lock()
	state := e.executions[jobID]
	knownUnstarted := false
	if state == nil {
		e.mu.Unlock()
		if e.journal != nil {
			token := ""
			if dispatchToken != nil {
				token = *dispatchToken
			}
			record, err := e.journal.RememberCancellation(jobID, token, false)
			if err != nil {
				return err
			}
			if record.StopVerified || record.KnownUnstarted {
				return nil
			}
		}
		return fmt.Errorf("job %s is not tracked; execution stop cannot be confirmed", jobID)
	}
	if dispatchToken != nil && state.dispatchToken != *dispatchToken {
		e.mu.Unlock()
		return fmt.Errorf("job %s is tracked under a different dispatch token; requested attempt stop cannot be confirmed", jobID)
	}
	if e.journal != nil {
		record, err := e.journal.RememberCancellation(jobID, state.dispatchToken, false)
		if err != nil {
			e.mu.Unlock()
			return err
		}
		knownUnstarted = record.KnownUnstarted
	}
	if state.stopConfirmed {
		e.mu.Unlock()
		return nil
	}
	state.cancelRequested = true
	state.cancel()
	e.mu.Unlock()

	dockerRuntime := state.runtime == "docker-custom" || state.runtime == "comfyui-batch" || state.runtime == "workspace"
	var teardownErr error
	remove := func() error {
		return e.removeContainers(ctx, jobID)
	}
	if dockerRuntime && !knownUnstarted {
		teardownErr = remove()
	}
	select {
	case <-state.done:
	case <-ctx.Done():
		return errors.Join(fmt.Errorf("waiting for job execution to stop: %w", ctx.Err()), teardownErr)
	}
	if !dockerRuntime && !knownUnstarted {
		return fmt.Errorf("runtime %q cannot confirm backend execution stopped", state.runtime)
	}

	if !knownUnstarted {
		if err := errors.Join(teardownErr, remove()); err != nil {
			return fmt.Errorf("job teardown could not be verified: %w", err)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	e.mu.Lock()
	if e.journal != nil {
		if err := e.journal.Complete(jobID, state.dispatchToken, nil, true); err != nil {
			state.stopUnverified = true
			e.mu.Unlock()
			return err
		}
	}
	state.stopConfirmed = true
	state.stopUnverified = false
	delete(e.workspaces, jobID)
	e.releaseExecution(state)
	e.mu.Unlock()
	return nil
}

func (e *Executor) StopAll(ctx context.Context) error {
	e.mu.Lock()
	ids := make(map[string]bool, len(e.executions))
	for id, state := range e.executions {
		if e.inflight[id] != nil || e.workspaces[id] || state.stopUnverified || (state.cancelRequested && !state.stopConfirmed) {
			ids[id] = true
		}
	}
	for id := range e.workspaces {
		ids[id] = true
	}
	e.mu.Unlock()

	var errs []error
	for id := range ids {
		if err := e.CancelContext(ctx, id); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func jobTimeoutForAssignment(a types.JobAssignment, cap time.Duration) time.Duration {
	if cap <= 0 {
		cap = 60 * time.Minute
	}
	timeout := cap
	if a.Parameters == nil {
		return timeout
	}
	if requested := durationFromParam(a.Parameters["timeout_seconds"], time.Second); requested > 0 && requested < timeout {
		timeout = requested
	}
	if requested := durationFromParam(a.Parameters["timeout_minutes"], time.Minute); requested > 0 && requested < timeout {
		timeout = requested
	}
	return timeout
}

func durationFromParam(value interface{}, unit time.Duration) time.Duration {
	switch v := value.(type) {
	case float64:
		if v > 0 {
			return time.Duration(v) * unit
		}
	case int:
		if v > 0 {
			return time.Duration(v) * unit
		}
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err == nil && n > 0 {
			return time.Duration(n) * unit
		}
	}
	return 0
}
