package job

import (
	"context"
	"os"
	"sync"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

// Executor orchestrates a single job through the full pipeline:
//  1. Resolve Docker image
//  2. Pull image (cached)
//  3. Download custom files (LoRAs, workflows)
//  4. Run inference
//  5. Upload outputs
//  6. Return result
//
// Progress is reported via the optional OnProgress callback so the pool client
// can relay it to the coordinator (and thus to the user's browser).
type Executor struct {
	runtime    Runtime
	gpuID      string
	backend    string
	cacheDir   string
	jobTimeout time.Duration
	OnProgress func(types.JobProgress) // set by pool client

	// Teardown/tracking: `inflight` holds cancel funcs for jobs currently
	// executing; `workspaces` holds job ids whose detached workspace container
	// is still running (so it can be stopped on cancel/shutdown). `teardown`
	// stops containers by their deterministic name.
	mu         sync.Mutex
	inflight   map[string]context.CancelFunc
	workspaces map[string]bool
	teardown   *dockermgr.Manager
}

// ExecutorOptions configures a new Executor.
type ExecutorOptions struct {
	CacheDir           string
	MaxCacheGB         int
	GPUID              string
	Backend            string
	GPUDevice          string                   // scope containers to a GPU (see types.Config)
	JobTimeout         time.Duration            // batch job cap; 0 → 60m
	MaxCustomFileBytes int64                    // per-file download cap; 0 → unlimited
	Policy             dockermgr.SecurityPolicy // container sandbox policy
	HFToken            string                   // HuggingFace token for gated downloads
}

// NewExecutor builds an executor with default execution options.
func NewExecutor(cacheDir string, maxCacheGB int, gpuID, backend string) (*Executor, error) {
	return NewExecutorWithOptions(ExecutorOptions{
		CacheDir: cacheDir, MaxCacheGB: maxCacheGB, GPUID: gpuID, Backend: backend,
	})
}

// NewExecutorWithOptions builds an executor whose Runtime matches the backend,
// honoring GPU scoping, job timeout, and download caps.
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

// Execute runs a single job end-to-end and returns a JobResult ready to send.
func (e *Executor) Execute(ctx context.Context, a types.JobAssignment) types.JobResult {
	start := time.Now()
	res := types.JobResult{Type: "job_result", JobID: a.JobID, GPUID: e.gpuID}

	// Wrap in a cancelable context registered by job id so a job_cancel message
	// (or shutdown) can interrupt this job.
	jobCtx, cancel := context.WithCancel(ctx)
	e.trackInflight(a.JobID, cancel)
	workspace := isWorkspaceJob(a)
	if !workspace {
		jobTimeout := e.jobTimeout
		if jobTimeout <= 0 {
			jobTimeout = 60 * time.Minute
		}
		var timeoutCancel context.CancelFunc
		jobCtx, timeoutCancel = context.WithTimeout(jobCtx, jobTimeout)
		defer timeoutCancel()
	}
	defer func() {
		e.untrackInflight(a.JobID)
		// Keep workspace containers registered on success so they can be torn
		// down later; a failed/aborted workspace leaves nothing running.
		if workspace && res.Success {
			e.markWorkspace(a.JobID)
		}
		cancel()
	}()
	ctx = jobCtx

	// ── Stage 1: Download verified custom files ────────────────────────
	if len(a.CustomFiles) > 0 {
		e.progress(a.JobID, "downloading_files", 0, "Downloading custom files...")
		stagingDir := e.cacheDir + "/staging/" + a.JobID
		if err := DownloadCustomFilesCached(ctx, a.CustomFiles, stagingDir, e.cacheDir+"/assets", func(stage string, pct float64, msg string) {
			e.progress(a.JobID, stage, pct, msg)
		}); err != nil {
			res.Success = false
			res.Error = "custom file download failed: " + err.Error()
			res.DurationMS = elapsedMilliseconds(start)
			return res
		}
	}

	// ── Stage 2: Prepare (pull image / download model) ──────────────────
	e.progress(a.JobID, "pulling_image", 0, "Preparing model...")
	if err := e.runtime.Prepare(ctx, a); err != nil {
		res.Success = false
		res.Error = "model preparation failed: " + err.Error()
		res.DurationMS = elapsedMilliseconds(start)
		return res
	}

	// ── Stage 3: Run inference ──────────────────────────────────────────
	e.progress(a.JobID, "running", 0.5, "Running inference...")
	out, err := e.runtime.Run(ctx, a)
	res.DurationMS = elapsedMilliseconds(start)
	if err != nil {
		res.Success = false
		res.Error = err.Error()
		return res
	}

	// ── Stage 4: Upload output (if upload URL provided) ─────────────────
	if a.UploadURL != "" {
		e.progress(a.JobID, "uploading", 0.9, "Uploading output...")
		// Look for output file path in the result
		if outputPath, ok := out["output_file"].(string); ok && outputPath != "" {
			if uploadErr := UploadOutput(ctx, outputPath, a.UploadURL); uploadErr != nil {
				// Don't fail the job — just note the upload error
				out["upload_error"] = uploadErr.Error()
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
		res.Result = out
	} else {
		res.Result = map[string]interface{}{"status": "completed"}
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

// CleanupModels delegates cache cleanup to the active runtime.
func (e *Executor) CleanupModels(force bool) error {
	return e.runtime.Cleanup(force)
}

// ── Job tracking / cancellation ──────────────────────────────────────────────

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

// Cancel stops a running job and tears down any container it started
// (batch or workspace). Idempotent and safe to call for unknown job ids.
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

	// Stop+remove by deterministic name. Only one of these exists per job; the
	// other is a no-op. Uses Background so teardown isn't tied to any request.
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

// StopAll cancels every tracked job and tears down its containers. Called on
// agent shutdown so no detached workspace container is left holding the GPU.
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
