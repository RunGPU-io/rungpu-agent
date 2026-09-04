package job

import (
	"context"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type Executor struct {
	runtime    Runtime
	gpuID      string
	backend    string
	cacheDir   string
	jobTimeout time.Duration
	OnProgress func(types.JobProgress)

	mu         sync.Mutex
	inflight   map[string]context.CancelFunc
	workspaces map[string]bool
	teardown   *dockermgr.Manager
}

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
	rt, err := NewRuntimeOpts(o.Backend, o.CacheDir, o.MaxCacheGB, RuntimeOptions{
		GPUDevice:  o.GPUDevice,
		JobTimeout: o.JobTimeout,
		Policy:     o.Policy,
	})
	if err != nil {
		return nil, err
	}
	SetMaxDownloadBytes(o.MaxCustomFileBytes)
	SetHFToken(o.HFToken)
	return &Executor{
		runtime:    rt,
		gpuID:      o.GPUID,
		backend:    o.Backend,
		cacheDir:   o.CacheDir,
		jobTimeout: o.JobTimeout,
		inflight:   map[string]context.CancelFunc{},
		workspaces: map[string]bool{},
		teardown:   dockermgr.New(),
	}, nil
}

func (e *Executor) Backend() string { return e.backend }
func (e *Executor) Runtime() string { return e.runtime.Name() }

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
	start := time.Now()
	res := types.JobResult{Type: "job_result", JobID: a.JobID, GPUID: e.gpuID}

	jobCtx, cancel := context.WithCancel(ctx)
	e.trackInflight(a.JobID, cancel)
	workspace := a.Runtime == "workspace"
	if !workspace {
		jobTimeout := jobTimeoutForAssignment(a, e.jobTimeout)
		var timeoutCancel context.CancelFunc
		jobCtx, timeoutCancel = context.WithTimeout(jobCtx, jobTimeout)
		defer timeoutCancel()
	}
	defer func() {
		e.untrackInflight(a.JobID)

		if workspace && res.Success {
			e.markWorkspace(a.JobID)
		}
		cancel()
	}()
	ctx = jobCtx

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
		e.progress(a.JobID, "downloading_files", 0, "Downloading custom files...")
		if err := DownloadCustomFilesCached(ctx, a.CustomFiles, stagingDir, e.cacheDir+"/assets", func(stage string, pct float64, msg string) {
			e.progress(a.JobID, stage, pct, msg)
		}); err != nil {
			res.Success = false
			res.Error = "custom file download failed: " + err.Error()
			res.DurationMS = elapsedMilliseconds(start)
			return res
		}
	}

	e.progress(a.JobID, "pulling_image", 0, "Preparing model...")
	if err := e.runtime.Prepare(ctx, a); err != nil {
		res.Success = false
		res.Error = "model preparation failed: " + err.Error()
		res.DurationMS = elapsedMilliseconds(start)
		return res
	}

	e.progress(a.JobID, "running", 0.5, "Running inference...")
	out, err := e.runtime.Run(ctx, a)
	res.DurationMS = elapsedMilliseconds(start)
	if err != nil {
		res.Success = false
		res.Error = err.Error()
		return res
	}

	if a.UploadURL != "" {
		e.progress(a.JobID, "uploading", 0.9, "Uploading output...")

		if outputPath, ok := out["output_file"].(string); ok && outputPath != "" {
			if uploadErr := UploadOutput(ctx, outputPath, a.UploadURL); uploadErr != nil {
				res.Success = false
				res.Error = "output upload failed: " + uploadErr.Error()
				return res
			} else {
				out["uploaded"] = true
				if info, statErr := os.Stat(outputPath); statErr == nil {
					out["output_size_bytes"] = info.Size()
					out["output_content_type"] = outputContentType(outputPath)
				}
			}
		}
	}

	e.progress(a.JobID, "completed", 1.0, "Done")

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

func (e *Executor) progress(jobID, stage string, pct float64, msg string) {
	if e.OnProgress != nil {
		e.OnProgress(types.JobProgress{
			Type:     "job_progress",
			JobID:    jobID,
			GPUID:    e.gpuID,
			Stage:    stage,
			Progress: pct,
			Message:  msg,
		})
	}
}

func (e *Executor) CleanupModels(force bool) error {
	return e.runtime.Cleanup(force)
}

func (e *Executor) trackInflight(jobID string, cancel context.CancelFunc) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.inflight == nil {
		e.inflight = map[string]context.CancelFunc{}
	}
	e.inflight[jobID] = cancel
}

func (e *Executor) untrackInflight(jobID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.inflight, jobID)
}

func (e *Executor) markWorkspace(jobID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.workspaces == nil {
		e.workspaces = map[string]bool{}
	}
	e.workspaces[jobID] = true
}

func (e *Executor) Cancel(jobID string) {
	e.cancelWithContext(context.Background(), jobID)
}

func (e *Executor) cancelWithContext(ctx context.Context, jobID string) {
	e.mu.Lock()
	if cancel, ok := e.inflight[jobID]; ok {
		cancel()
		delete(e.inflight, jobID)
	}
	delete(e.workspaces, jobID)
	e.mu.Unlock()

	if e.teardown == nil {
		e.teardown = dockermgr.New()
	}
	for _, name := range []string{CustomContainerName(jobID), WorkspaceContainerName(jobID)} {
		if ctx.Err() != nil {
			return
		}
		_ = e.teardown.Stop(ctx, name)
		_ = e.teardown.Remove(ctx, name)
	}
}

func (e *Executor) StopAll(ctx context.Context) {
	e.mu.Lock()
	ids := make(map[string]bool, len(e.inflight)+len(e.workspaces))
	for id := range e.inflight {
		ids[id] = true
	}
	for id := range e.workspaces {
		ids[id] = true
	}
	e.mu.Unlock()

	for id := range ids {
		e.cancelWithContext(ctx, id)
	}
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
