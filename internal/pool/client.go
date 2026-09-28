package pool

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	stdlog "log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/config"
	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/gpu"
	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
	"github.com/gorilla/websocket"
)

func log(format string, args ...interface{}) {
	stdlog.Printf("[agent] "+format, args...)
}

func shortJobRef(jobID string) string {
	if len(jobID) > 12 {
		return jobID[:12]
	}
	return jobID
}

const (
	writeWait                 = 10 * time.Second
	pongWait                  = 60 * time.Second
	pingPeriod                = 50 * time.Second
	registrationWait          = 30 * time.Second
	maxDeferredAssignments    = 64
	maxCancellationTombstones = 1024
)

var machineExecutionGate sync.RWMutex

type cancellationTombstone struct {
	knownUnstarted bool
	stopVerified   bool
}

type Client struct {
	cfg         *types.Config
	monitor     *gpu.Monitor
	executor    jobExecutor
	gpuID       string
	backend     string
	deviceIndex int

	baseCtx context.Context

	outbox chan interface{}

	resultsMu    sync.Mutex
	results      map[string]*pendingResult
	resultsReady chan struct{}
	cancelling   sync.Map

	assignmentsMu               sync.Mutex
	activeAssignment            string
	activeDispatchToken         string
	deferredAssignments         map[string]types.JobAssignment
	settledAttempts             []string
	cancellationTombstones      map[string]*cancellationTombstone
	cancellationAdmissionFenced bool
	executionJournal            *job.AttemptJournal
	activeJobs                  int64
	cleanupRunning              int32
	jobSlot                     chan struct{}
	outboxDir                   string
	maintenanceGate             *sync.RWMutex
	configPath                  string
}

type jobExecutor interface {
	Start(context.Context, types.JobAssignment) (<-chan types.JobResult, error)
	IsTracked(string) bool
	TrackedDispatchToken(string) (string, bool)
	Cancel(string) error
	CancelAttempt(string, string) error
	ForgetExecution(string)
	StopAll(context.Context) error
	Runtime() string
	WorkspaceIDs() []string
	PreviewCleanup([]string) (job.CleanupPreview, error)
	ExecuteCleanup(context.Context, []string) (job.CleanupPreview, error)
}

func (c *Client) SetConfigPath(path string) {
	c.configPath = path
}

func (c *Client) earningNow() config.EarningStatus {
	if c.configPath != "" {
		if loaded, err := config.Load(c.configPath); err == nil {
			return config.EvaluateEarning(loaded, time.Now())
		}
	}
	return config.EvaluateEarning(c.cfg, time.Now())
}

func NewClient(cfg *types.Config) (*Client, error) {
	return NewClientForGPU(cfg, 0)
}

func NewClients(cfg *types.Config) ([]*Client, error) {
	monitor := gpu.NewMonitor()
	if monitor.Backend() == "cpu" && !cfg.AllowCPUServing {
		return nil, fmt.Errorf("no GPU accelerator detected; set allow_cpu_serving: true to intentionally serve jobs on CPU")
	}
	detected := monitor.GPUs()
	count := len(detected)
	if count == 0 {
		count = 1
	}
	clients := make([]*Client, 0, count)
	maintenanceGate := &machineExecutionGate
	for position := 0; position < count; position++ {
		deviceIndex := position
		if position < len(detected) {
			deviceIndex = detected[position].Index
		}
		client, err := newClientForGPU(cfg, deviceIndex, maintenanceGate)
		if err != nil {
			return nil, err
		}
		clients = append(clients, client)
	}
	return clients, nil
}

func NewClientForGPU(cfg *types.Config, deviceIndex int) (*Client, error) {
	return newClientForGPU(cfg, deviceIndex, &machineExecutionGate)
}

func newClientForGPU(cfg *types.Config, deviceIndex int, maintenanceGate *sync.RWMutex) (*Client, error) {
	gpuID := deterministicGPUID(cfg.MachineID, deviceIndex)
	if cfg.MachineID == "" && deviceIndex < len(cfg.GPUIDs) {
		gpuID = cfg.GPUIDs[deviceIndex]
	}
	monitor := gpu.NewMonitor()
	backend := monitor.Backend()
	executor, err := job.NewExecutorWithOptions(job.ExecutorOptions{
		CacheDir:           cfg.ModelCacheDir,
		MaxCacheGB:         cfg.MaxModelCacheGB,
		GPUID:              gpuID,
		Backend:            backend,
		GPUDevice:          strconv.Itoa(deviceIndex),
		JobTimeout:         time.Duration(cfg.JobTimeoutMinutes) * time.Minute,
		MaxCustomFileBytes: int64(cfg.MaxCustomFileGB) * 1024 * 1024 * 1024,
		Policy:             dockermgr.PolicyFromConfig(cfg.Security),
		HFToken:            cfg.Security.HFToken,
		ExecutionGate:      maintenanceGate,
		ExpectedJournalID:  cfg.ExecutionJournalID,
	})
	if err != nil {
		return nil, err
	}

	return clientWithExecutor(cfg, gpuID, backend, deviceIndex, monitor, maintenanceGate, executor)
}

func clientWithExecutor(cfg *types.Config, gpuID, backend string, deviceIndex int, monitor *gpu.Monitor, maintenanceGate *sync.RWMutex, executor *job.Executor) (*Client, error) {
	initialized := false
	defer func() {
		if !initialized && executor.Journal() != nil {
			_ = executor.Journal().Close()
		}
	}()
	if !types.ValidJobID(gpuID) {
		return nil, fmt.Errorf("invalid GPU identifier")
	}
	c := &Client{
		cfg:              cfg,
		monitor:          monitor,
		executor:         executor,
		gpuID:            gpuID,
		backend:          backend,
		deviceIndex:      deviceIndex,
		outbox:           make(chan interface{}, 64),
		resultsReady:     make(chan struct{}, 1),
		jobSlot:          make(chan struct{}, 1),
		outboxDir:        filepath.Join(cfg.ModelCacheDir, "outbox", gpuID),
		maintenanceGate:  maintenanceGate,
		executionJournal: executor.Journal(),
	}
	if err := os.MkdirAll(c.outboxDir, 0o700); err != nil {
		return nil, fmt.Errorf("create result outbox: %w", err)
	}
	c.loadPendingResults()
	if c.executionJournal != nil {
		for _, pending := range c.results {
			if _, exists := c.executionJournal.Find(pending.result.JobID, pending.result.DispatchToken); !exists {
				record, err := c.executionJournal.RememberLegacyResult(pending.result)
				if err != nil {
					return nil, err
				}
				if err := executor.TrackUnverifiedLegacy(record); err != nil {
					return nil, err
				}
			}
		}
		for _, record := range c.executionJournal.Records() {
			if record.CancelRequested {
				if len(c.cancellationTombstones) >= maxCancellationTombstones {
					c.cancellationAdmissionFenced = true
				} else {
					if c.cancellationTombstones == nil {
						c.cancellationTombstones = make(map[string]*cancellationTombstone)
					}
					c.cancellationTombstones[attemptResultKey(record.JobID, record.DispatchToken)] = &cancellationTombstone{
						knownUnstarted: record.KnownUnstarted, stopVerified: record.StopVerified,
					}
				}
			}
			if record.Result != nil {

				for key, pending := range c.results {
					if pending.result.JobID == record.JobID && pending.result.DispatchToken == record.DispatchToken {
						if record.ResultAccepted {
							if err := os.Remove(c.resultPath(key)); err != nil && !os.IsNotExist(err) {
								return nil, fmt.Errorf("recover accepted result cleanup: %w", err)
							}
						}
						delete(c.results, key)
					}
				}
				if !record.ResultAccepted {
					if pending := c.results[record.JobID]; pending != nil && pending.result.DispatchToken != record.DispatchToken {
						c.queueAttemptRejection(*record.Result)
					} else {
						c.queueResult(*record.Result)
					}
				}
			}
		}
		if err := c.executionJournal.Err(); err != nil {
			return nil, err
		}
		if len(c.cancellationTombstones) >= maxCancellationTombstones {
			c.cancellationAdmissionFenced = true
		}
	}

	executor.OnProgress = func(p types.JobProgress) {
		log("job %s: stage=%s progress=%.0f%%", shortJobRef(p.JobID), p.Stage, p.Progress*100)
		c.enqueue(p)
	}

	initialized = true
	return c, nil
}

func deterministicGPUID(machineID string, deviceIndex int) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", machineID, deviceIndex)))
	b := sum[:16]
	b[6] = (b[6] & 0x0f) | 0x50
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func (c *Client) enqueue(msg interface{}) {
	select {
	case c.outbox <- msg:
	default:
	}
}

func RunCustomAssetCleanup(ctx context.Context, cfg *types.Config, clients []*Client) {
	if cfg.CleanupIntervalHours <= 0 ||
		(cfg.CustomAssetTTLDays < 0 && cfg.MaxCustomAssetCacheGB < 0) {
		return
	}
	interval := time.Duration(cfg.CleanupIntervalHours) * time.Hour
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	prune := func() {
		files, bytes, skipped, err := pruneCustomAssetsIfIdle(cfg, clients, time.Now())
		if skipped {
			return
		}
		if err != nil {
			log("custom asset cleanup failed: %v", err)
		} else if files > 0 {
			log("custom asset cleanup evicted %d expired/LRU cache file(s), reclaiming %d bytes", files, bytes)
		}
	}
	prune()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			prune()
		}
	}
}

func pruneCustomAssetsIfIdle(cfg *types.Config, clients []*Client, now time.Time) (files int, bytes int64, skipped bool, err error) {
	for _, client := range clients {
		if job.MachineRecoveryBlocked(client.maintenanceGate) {
			return 0, 0, true, nil
		}
	}
	if len(clients) > 0 && clients[0].maintenanceGate != nil {
		if !clients[0].maintenanceGate.TryLock() {
			return 0, 0, true, nil
		}
		defer clients[0].maintenanceGate.Unlock()
	}
	workspaceIDs := map[string]bool{}
	for _, client := range clients {
		if atomic.LoadInt64(&client.activeJobs) > 0 || atomic.LoadInt32(&client.cleanupRunning) > 0 {
			return 0, 0, true, nil
		}
		if client.executor != nil {
			for _, id := range client.executor.WorkspaceIDs() {
				workspaceIDs[id] = true
			}
		}
	}
	if err := job.PruneBatchStaging(cfg.ModelCacheDir, workspaceIDs); err != nil {
		return 0, 0, false, err
	}
	maxBytes := int64(cfg.MaxCustomAssetCacheGB) * 1024 * 1024 * 1024
	if cfg.MaxCustomAssetCacheGB < 0 {
		maxBytes = 0
	}
	files, bytes, err = job.PruneCustomAssets(
		cfg.ModelCacheDir,
		time.Duration(cfg.CustomAssetTTLDays)*24*time.Hour,
		maxBytes,
		now,
	)
	return files, bytes, false, err
}

func (c *Client) Run(ctx context.Context) error {
	c.baseCtx = ctx

	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := c.executor.StopAll(cleanupCtx); err != nil {
			log("shutdown teardown could not be confirmed: %v", err)
		} else if c.executionJournal != nil {
			if err := c.executionJournal.Close(); err != nil {
				log("close execution journal: %v", err)
			}
		}
	}()

	backoff := 2 * time.Second
	const maxBackoff = 30 * time.Second

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err := c.waitUntilEarning(ctx); err != nil {
			return err
		}

		sessionCtx, stopWatch := c.watchEarning(ctx)
		err := c.session(sessionCtx)
		stopWatch()
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if sessionCtx.Err() != nil && ctx.Err() == nil && !c.earningNow().Open {
			log("%s", c.earningNow().Summary)
			continue
		}
		if err != nil {
			log("connection error: %v (reconnecting in %s)", err, backoff)
		} else {
			log("disconnected; reconnecting in %s", backoff)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		if backoff < maxBackoff {
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}
}

func (c *Client) waitUntilEarning(ctx context.Context) error {
	for {
		status := c.earningNow()
		if status.Open {
			return nil
		}
		log("%s — keep the app or start command running", status.Summary)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
}

func (c *Client) watchEarning(parent context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if !c.earningNow().Open {
					cancel()
					return
				}
			}
		}
	}()
	return ctx, cancel
}

func (c *Client) session(parent context.Context) error {
	endpoint, err := c.endpoint()
	if err != nil {
		return err
	}

	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+c.cfg.APIKey)
	conn, _, err := websocket.DefaultDialer.DialContext(parent, endpoint, headers)
	if err != nil {
		return err
	}
	defer conn.Close()
	stopClose := context.AfterFunc(parent, func() { _ = conn.Close() })
	defer stopClose()

	log("connected to pool as %s", c.gpuID)

	_ = conn.SetWriteDeadline(time.Now().Add(writeWait))
	if err := conn.WriteJSON(c.registerMessage()); err != nil {
		return err
	}
	return c.serveConnection(parent, conn, registrationWait)
}

func (c *Client) serveConnection(parent context.Context, conn *websocket.Conn, registrationTimeout time.Duration) error {
	ctx, cancel := context.WithCancel(parent)
	writerDone := make(chan struct{})
	defer func() {
		cancel()
		_ = conn.Close()
		<-writerDone
	}()

	sendCh := make(chan interface{}, 16)
	registered := make(chan struct{})
	go func() {
		defer close(writerDone)
		c.writer(ctx, conn, sendCh, registered)
	}()

	go c.heartbeatLoop(ctx, sendCh)

	registrationDeadline := time.Now().Add(registrationTimeout)
	ready := false
	refreshDeadline := func() error {
		deadline := time.Now().Add(pongWait)
		if !ready && registrationDeadline.Before(deadline) {
			deadline = registrationDeadline
		}
		return conn.SetReadDeadline(deadline)
	}
	_ = refreshDeadline()
	conn.SetPongHandler(func(string) error {
		return refreshDeadline()
	})
	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			cancel()
			if parent.Err() != nil {
				return parent.Err()
			}
			if timeout, ok := err.(net.Error); !ready && ok && timeout.Timeout() {
				return fmt.Errorf("registration acknowledgement timeout: %w", err)
			}
			return err
		}
		if !ready {
			if err := parent.Err(); err != nil {
				return err
			}
			if !time.Now().Before(registrationDeadline) {
				return fmt.Errorf("registration acknowledgement timeout")
			}
			var ack struct {
				Type    string `json:"type"`
				Success bool   `json:"success"`
				Error   string `json:"error"`
			}
			if json.Unmarshal(data, &ack) != nil || ack.Type != "gpu_register_ack" {
				continue
			}
			if !ack.Success {
				return fmt.Errorf("registration rejected: %s", ack.Error)
			}
			ready = true
			close(registered)
		}

		_ = refreshDeadline()
		c.dispatch(ctx, sendCh, data)
	}
}

func (c *Client) writer(ctx context.Context, conn *websocket.Conn, sendCh <-chan interface{}, registered <-chan struct{}) {

	defer conn.Close()
	ping := time.NewTicker(pingPeriod)
	defer ping.Stop()
	retry := time.NewTicker(resultRetryTick)
	defer retry.Stop()

	var outbox <-chan interface{}
	var resultsReady <-chan struct{}
	var retryResults <-chan time.Time

	write := func(msg interface{}) bool {
		if ctx.Err() != nil {
			return false
		}
		_ = conn.SetWriteDeadline(time.Now().Add(writeWait))
		if err := conn.WriteJSON(msg); err != nil {
			log("write error: %v", err)
			return false
		}
		return true
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-registered:
			registered = nil
			outbox = c.outbox
			resultsReady = c.resultsReady
			retryResults = retry.C
			c.loadPendingResults()
			if !c.sendPendingResults(ctx, time.Now(), write) {
				return
			}
		case msg := <-sendCh:
			if !write(msg) {
				return
			}
		case msg := <-outbox:
			if !write(msg) {
				c.enqueue(msg)
				return
			}
		case <-resultsReady:
			if !c.sendPendingResults(ctx, time.Now(), write) {
				return
			}
		case now := <-retryResults:
			if !c.sendPendingResults(ctx, now, write) {
				return
			}
		case <-ping.C:
			_ = conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (c *Client) heartbeatLoop(ctx context.Context, sendCh chan<- interface{}) {
	interval := time.Duration(c.cfg.HeartbeatIntervalSecs) * time.Second
	if interval <= 0 {
		interval = 30 * time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.trySend(ctx, sendCh, c.heartbeatMessage())
		}
	}
}

func (c *Client) dispatch(ctx context.Context, sendCh chan<- interface{}, data []byte) {
	var env types.Envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return
	}

	switch env.Type {
	case "job_assignment":
		var a types.JobAssignment
		decodeErr := json.Unmarshal(data, &a)

		if !types.ValidJobID(a.JobID) {
			log("rejected job_assignment: invalid job_id")
			return
		}
		c.assignmentsMu.Lock()
		defer c.assignmentsMu.Unlock()

		if c.executionJournal != nil {
			if record, exists := c.executionJournal.Find(a.JobID, a.DispatchToken); exists && record.Result != nil {
				return
			}
		}
		attemptKey := attemptResultKey(a.JobID, a.DispatchToken)
		for _, settled := range c.settledAttempts {
			if a.DispatchToken != "" && settled == attemptKey {
				return
			}
		}
		c.resultsMu.Lock()
		pending := c.results[a.JobID]
		rejected := a.DispatchToken != "" && c.results[attemptKey] != nil
		c.resultsMu.Unlock()
		if rejected {
			return
		}
		if c.cancellationTombstones[attemptKey] != nil {
			if (pending != nil && pending.result.DispatchToken == a.DispatchToken) ||
				c.tracksAttempt(a.JobID, a.DispatchToken) {
				return
			}
			c.rejectDeferredAssignment(a, fmt.Errorf("dispatch attempt was cancelled before admission"))
			return
		}
		if c.cancellationAdmissionFenced {
			if c.tracksAttempt(a.JobID, a.DispatchToken) {
				return
			}
			if a.DispatchToken != "" {
				if pending == nil || pending.result.DispatchToken != a.DispatchToken {
					c.rejectDeferredAssignment(a, fmt.Errorf("agent admission is fenced: cancellation tombstone capacity reached"))
				}
			} else if pending == nil && c.activeAssignment != a.JobID && (c.executor == nil || !c.executor.IsTracked(a.JobID)) {
				c.queueResult(types.JobResult{Type: "job_result", JobID: a.JobID, GPUID: c.gpuID,
					Error: "agent admission is fenced: cancellation tombstone capacity reached"})
			}
			return
		}
		if deferred, exists := c.deferredAssignments[a.JobID]; exists {
			if a.DispatchToken == "" || a.DispatchToken == deferred.DispatchToken ||
				(pending != nil && a.DispatchToken == pending.result.DispatchToken) {
				return
			}
			c.rejectDeferredAssignment(a, fmt.Errorf("GPU already has a deferred retry for this job"))
			return
		}
		if pending != nil {
			if a.DispatchToken == "" || pending.result.DispatchToken == "" || a.DispatchToken == pending.result.DispatchToken {
				return
			}
			if c.assignmentCancelling(a) {
				c.rejectDeferredAssignment(a, fmt.Errorf("job cancellation is in progress"))
			} else if decodeErr != nil {
				c.rejectDeferredAssignment(a, fmt.Errorf("invalid job_assignment: %w", decodeErr))
			} else if err := a.Validate(); err != nil {
				c.rejectDeferredAssignment(a, err)
			} else if len(c.deferredAssignments) >= maxDeferredAssignments {
				c.rejectDeferredAssignment(a, fmt.Errorf("deferred assignment queue is full"))
			} else {
				if c.executionJournal != nil {
					if _, err := c.executionJournal.Admit(a, c.backend); err != nil {
						c.rejectDeferredAssignment(a, err)
						return
					}
				}
				if c.deferredAssignments == nil {
					c.deferredAssignments = make(map[string]types.JobAssignment)
				}
				c.deferredAssignments[a.JobID] = a
			}
			return
		}
		if c.activeAssignment == a.JobID || (c.executor != nil && c.executor.IsTracked(a.JobID)) {
			return
		}
		reject := func(err error) {
			log("rejected job_assignment: %v", err)
			c.queueResult(types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken, GPUID: c.gpuID, Error: err.Error()})
		}
		if decodeErr != nil {
			reject(fmt.Errorf("invalid job_assignment: %w", decodeErr))
			return
		}
		if err := a.Validate(); err != nil {
			reject(err)
			return
		}
		if c.assignmentCancelling(a) {
			reject(fmt.Errorf("job cancellation is in progress"))
			return
		}
		c.startAssignmentLocked(a, false)
	case "job_cancel", "job_stop", "stop_job":
		var jc types.JobControl
		if err := json.Unmarshal(data, &jc); err != nil || !types.ValidJobID(jc.JobID) {
			return
		}
		c.cancelJob(jc)
	case "job_result_ack":
		var ack struct {
			JobID         string `json:"job_id"`
			DispatchToken string `json:"dispatch_token"`
			Success       bool   `json:"success"`
		}
		if json.Unmarshal(data, &ack) == nil && types.ValidJobID(ack.JobID) && ack.Success {
			c.assignmentsMu.Lock()
			defer c.assignmentsMu.Unlock()
			if accepted, canonical, token := c.ackResult(ack.JobID, ack.DispatchToken); accepted {
				if token != "" {
					if len(c.settledAttempts) == maxDeferredAssignments {
						c.settledAttempts = c.settledAttempts[1:]
					}
					c.settledAttempts = append(c.settledAttempts, attemptResultKey(ack.JobID, token))
				}
				if canonical && c.executor != nil {
					c.executor.ForgetExecution(ack.JobID)
				}
				c.drainDeferredAssignmentsLocked()
			}
		}
	case "asset_cleanup":
		var request types.AssetCleanupRequest
		if json.Unmarshal(data, &request) != nil || request.RequestID == "" {
			return
		}
		if !atomic.CompareAndSwapInt32(&c.cleanupRunning, 0, 1) {
			c.trySend(ctx, sendCh, types.AssetCleanupResult{
				Type: "asset_cleanup_result", RequestID: request.RequestID,
				Phase: request.Phase, Success: false, Error: "cleanup already running",
			})
			return
		}
		go func() {
			defer atomic.StoreInt32(&c.cleanupRunning, 0)
			result := types.AssetCleanupResult{
				Type: "asset_cleanup_result", RequestID: request.RequestID,
				Phase: request.Phase, Success: true,
			}
			var preview job.CleanupPreview
			var err error
			if request.Phase == "preview" {
				preview, err = c.executor.PreviewCleanup(request.Categories)
			} else if request.Phase == "execute" {
				cleanupCtx, cancel := context.WithTimeout(c.baseCtx, 30*time.Minute)
				defer cancel()
				if job.MachineRecoveryBlocked(c.maintenanceGate) {
					err = fmt.Errorf("cleanup blocked by unresolved execution recovery")
				} else if c.maintenanceGate != nil {
					if c.maintenanceGate.TryLock() {
						defer c.maintenanceGate.Unlock()
					} else {
						err = fmt.Errorf("cleanup blocked while machine jobs are active or their stop is unverified")
					}
				}
				if err == nil {
					preview, err = c.executor.ExecuteCleanup(cleanupCtx, request.Categories)
				}
			} else {
				err = fmt.Errorf("unsupported cleanup phase")
			}
			if err != nil {
				result.Success = false
				result.Error = err.Error()
			} else {
				result.TotalBytes = preview.TotalBytes
				result.ActiveJobs = preview.ActiveJobs
				result.Categories = map[string]interface{}{}
				for name, category := range preview.Categories {
					result.Categories[name] = category
				}
			}
			select {
			case <-c.baseCtx.Done():
			case c.outbox <- result:
			}
		}()
	case "gpu_register_ack", "gpu_heartbeat_ack", "pool_metrics":

	default:

	}
}

func (c *Client) tracksAttempt(jobID, token string) bool {
	if c.activeAssignment == jobID && c.activeDispatchToken == token {
		return true
	}
	if c.executor == nil {
		return false
	}
	trackedToken, tracked := c.executor.TrackedDispatchToken(jobID)
	return tracked && trackedToken == token
}

func (c *Client) assignmentCancelling(a types.JobAssignment) bool {
	if _, pending := c.cancelling.Load(a.JobID); pending {
		return true
	}
	_, pending := c.cancelling.Load(attemptResultKey(a.JobID, a.DispatchToken))
	return a.DispatchToken != "" && pending
}

func (c *Client) cancellationTombstoneLocked(jobID, token string) (*cancellationTombstone, error) {
	key := attemptResultKey(jobID, token)
	if tombstone := c.cancellationTombstones[key]; tombstone != nil {
		if c.executionJournal != nil {
			record, err := c.executionJournal.RememberCancellation(jobID, token, false)
			if err != nil {
				c.cancellationAdmissionFenced = true
				return nil, err
			}
			tombstone.knownUnstarted, tombstone.stopVerified = record.KnownUnstarted, record.StopVerified
		}
		return tombstone, nil
	}
	if len(c.cancellationTombstones) >= maxCancellationTombstones {
		c.cancellationAdmissionFenced = true
		return nil, fmt.Errorf("cancellation tombstone capacity reached; agent admission is fenced")
	}
	if c.cancellationTombstones == nil {
		c.cancellationTombstones = make(map[string]*cancellationTombstone)
	}
	tombstone := &cancellationTombstone{}
	if c.executionJournal != nil {
		record, err := c.executionJournal.RememberCancellation(jobID, token, false)
		if err != nil {
			c.cancellationAdmissionFenced = true
			return nil, err
		}
		tombstone.knownUnstarted, tombstone.stopVerified = record.KnownUnstarted, record.StopVerified
	}
	c.cancellationTombstones[key] = tombstone
	return tombstone, nil
}

func (c *Client) cancelJob(control types.JobControl) {
	c.assignmentsMu.Lock()
	key := control.JobID
	var tombstone *cancellationTombstone
	if control.DispatchToken != "" {
		var err error
		tombstone, err = c.cancellationTombstoneLocked(control.JobID, control.DispatchToken)
		if err != nil {
			c.enqueue(types.JobCancelAck{Type: "job_cancel_ack", JobID: control.JobID,
				DispatchToken: control.DispatchToken, Error: err.Error()})
			c.assignmentsMu.Unlock()
			return
		}
		key = attemptResultKey(control.JobID, control.DispatchToken)
	}
	if _, pending := c.cancelling.LoadOrStore(key, struct{}{}); pending {
		c.assignmentsMu.Unlock()
		return
	}
	deferred, wasDeferred := c.deferredAssignments[control.JobID]
	wasDeferred = wasDeferred && (control.DispatchToken == "" || control.DispatchToken == deferred.DispatchToken)
	if wasDeferred {
		delete(c.deferredAssignments, control.JobID)
		if tombstone == nil && deferred.DispatchToken != "" {
			tombstone, _ = c.cancellationTombstoneLocked(control.JobID, deferred.DispatchToken)
		}
		if tombstone != nil {
			tombstone.knownUnstarted = true
		}
		c.rejectDeferredAssignment(deferred, fmt.Errorf("job cancelled before deferred execution"))
	}
	matchingAttempt := control.DispatchToken != "" && c.tracksAttempt(control.JobID, control.DispatchToken)
	legacyDeferredOnly := control.DispatchToken == "" && wasDeferred && c.activeAssignment != control.JobID &&
		(c.executor == nil || !c.executor.IsTracked(control.JobID))
	if control.DispatchToken != "" && !matchingAttempt && !tombstone.knownUnstarted && !tombstone.stopVerified {
		c.cancelling.Delete(key)
		c.enqueue(types.JobCancelAck{Type: "job_cancel_ack", JobID: control.JobID, DispatchToken: control.DispatchToken,
			Error: "dispatch attempt is not tracked; execution stop cannot be confirmed"})
		c.assignmentsMu.Unlock()
		return
	}
	c.assignmentsMu.Unlock()
	go func() {
		var err error
		switch {
		case matchingAttempt:
			err = c.executor.CancelAttempt(control.JobID, control.DispatchToken)
		case control.DispatchToken == "" && !legacyDeferredOnly:
			err = c.executor.Cancel(control.JobID)
		}
		c.assignmentsMu.Lock()
		if matchingAttempt && err == nil {
			tombstone.stopVerified = true
		}
		c.cancelling.Delete(key)
		c.resultsMu.Lock()
		pending := c.results[control.JobID] != nil
		c.resultsMu.Unlock()
		if err == nil && !pending && c.executor != nil &&
			(control.DispatchToken == "" || (matchingAttempt && c.tracksAttempt(control.JobID, control.DispatchToken))) {
			c.executor.ForgetExecution(control.JobID)
		}
		c.drainDeferredAssignmentsLocked()
		c.assignmentsMu.Unlock()
		ack := types.JobCancelAck{Type: "job_cancel_ack", JobID: control.JobID,
			DispatchToken: control.DispatchToken, Success: err == nil}
		if err != nil {
			ack.Error = err.Error()
		}
		c.enqueue(ack)
	}()
}

func (c *Client) rejectDeferredAssignment(a types.JobAssignment, err error) {
	c.queueAttemptRejection(types.JobResult{
		Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken,
		GPUID: c.gpuID, Error: err.Error(),
	})
}

func (c *Client) startAssignmentLocked(a types.JobAssignment, deferred bool) {
	if c.cancellationAdmissionFenced || (a.DispatchToken != "" &&
		c.cancellationTombstones[attemptResultKey(a.JobID, a.DispatchToken)] != nil) {
		result := types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken,
			GPUID: c.gpuID, Error: "dispatch attempt cannot start: cancellation admission fence"}
		c.queueResult(result)
		return
	}
	select {
	case c.jobSlot <- struct{}{}:
		results, err := c.executor.Start(c.baseCtx, a)
		if err != nil {
			<-c.jobSlot
			log("could not start job %s: %v", shortJobRef(a.JobID), err)
			if deferred || !errors.Is(err, job.ErrAlreadyTracked) {
				c.queueResult(types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken, GPUID: c.gpuID, Error: err.Error()})
			}
			return
		}
		c.activeAssignment = a.JobID
		c.activeDispatchToken = a.DispatchToken
		go c.runJob(a, results)
	default:
		c.queueResult(types.JobResult{Type: "job_result", JobID: a.JobID, DispatchToken: a.DispatchToken, GPUID: c.gpuID, Error: "GPU is already running a job"})
	}
}

func (c *Client) drainDeferredAssignmentsLocked() {
	if c.activeAssignment != "" || len(c.jobSlot) != 0 || c.jobSlot == nil {
		return
	}
	for id, a := range c.deferredAssignments {
		c.resultsMu.Lock()
		pending := c.results[id] != nil
		c.resultsMu.Unlock()
		if pending || c.assignmentCancelling(a) {
			continue
		}
		delete(c.deferredAssignments, id)
		c.startAssignmentLocked(a, true)
		if c.activeAssignment != "" {
			return
		}
	}
}

func (c *Client) runJob(a types.JobAssignment, results <-chan types.JobResult) {
	atomic.AddInt64(&c.activeJobs, 1)
	defer atomic.AddInt64(&c.activeJobs, -1)

	jobRef := shortJobRef(a.JobID)
	source := a.Source
	if source == "" {
		source = "RunGPU customer"
	}
	log("received job %s from %s: model=%q runtime=%s custom_assets=%d workspace=%t", jobRef, source, a.ModelName, c.executor.Runtime(), len(a.CustomFiles), a.Workspace)

	result := <-results
	result.DispatchToken = a.DispatchToken
	if result.Success {
		log("job %s completed in %dms", jobRef, result.DurationMS)
	} else {
		log("job %s failed after %dms", jobRef, result.DurationMS)
	}
	c.assignmentsMu.Lock()
	defer c.assignmentsMu.Unlock()
	c.queueResult(result)
	c.activeAssignment = ""
	c.activeDispatchToken = ""
	<-c.jobSlot
	c.drainDeferredAssignmentsLocked()
}

func (c *Client) registerMessage() types.RegisterMessage {
	gpus := c.monitor.GPUs()
	var name, driver string
	var vramGB float64
	for _, detected := range gpus {
		if detected.Index != c.deviceIndex {
			continue
		}
		name = detected.Name
		driver = detected.DriverVersion
		vramGB = float64(detected.MemoryMB) / 1024.0
		break
	}
	return types.RegisterMessage{
		Type:                "gpu_register",
		GPUID:               c.gpuID,
		MachineID:           c.cfg.MachineID,
		DeviceIndex:         c.deviceIndex,
		DetectedDeviceCount: len(gpus),
		GPUType:             name,
		Backend:             c.backend,
		VRAMGB:              vramGB,
		PricePerMinute:      c.cfg.PricePerMinute,
		ModelsCached:        c.cachedModels(),
		DriverVersion:       driver,
		Capabilities:        gpu.RuntimeCapabilities(),
		OllamaModels:        gpu.OllamaModels(),
	}
}

func (c *Client) heartbeatMessage() types.HeartbeatMessage {
	var availGB float64
	for _, m := range c.monitor.CollectMetrics() {
		if m.GPUIndex != c.deviceIndex {
			continue
		}
		if m.MemoryTotalMB > 0 {
			availGB = float64(m.MemoryTotalMB-m.MemoryUsedMB) / 1024.0
		}
		break
	}
	currentJobs := int(atomic.LoadInt64(&c.activeJobs))
	if executor, ok := c.executor.(*job.Executor); ok {
		currentJobs = max(currentJobs, executor.ReservedJobs())
	}
	if job.MachineRecoveryBlocked(c.maintenanceGate) {
		availGB = 0
	}
	return types.HeartbeatMessage{
		Type:            "gpu_heartbeat",
		GPUID:           c.gpuID,
		AvailableVRAMGB: availGB,
		CurrentJobs:     currentJobs,
		ModelsCached:    c.cachedModels(),
		OllamaModels:    gpu.OllamaModels(),
	}
}

func (c *Client) cachedModels() []string {
	entries, err := os.ReadDir(c.cfg.ModelCacheDir)
	if err != nil {
		return []string{}
	}
	models := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			models = append(models, e.Name())
		}
	}
	return models
}

func (c *Client) endpoint() (string, error) {
	u, err := url.Parse(c.cfg.PoolURL)
	if err != nil {
		return "", err
	}
	switch strings.ToLower(u.Scheme) {
	case "https", "wss":
		u.Scheme = "wss"
	case "http", "ws":
		host := u.Hostname()
		ip := net.ParseIP(host)
		if host != "localhost" && (ip == nil || !ip.IsLoopback()) {
			return "", fmt.Errorf("insecure pool URL %q: use HTTPS/WSS except for localhost", c.cfg.PoolURL)
		}
		u.Scheme = "ws"
	default:
		return "", fmt.Errorf("unsupported pool URL scheme %q", u.Scheme)
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/agent"
	q := u.Query()
	q.Set("gpu_id", c.gpuID)
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func (c *Client) trySend(ctx context.Context, sendCh chan<- interface{}, msg interface{}) {
	select {
	case <-ctx.Done():
	case sendCh <- msg:
	}
}
