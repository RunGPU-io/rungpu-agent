// Package pool implements the agent's connection to the coordinator over raw
// WebSocket: outbound dial (NAT/firewall friendly), register, heartbeat, and a
// job loop that executes assignments and reports results. Reconnects with
// backoff on disconnect.
package pool

import (
	"context"
	"crypto/sha256"
	"encoding/json"
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

	"github.com/gorilla/websocket"
	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/gpu"
	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func log(format string, args ...interface{}) {
	stdlog.Printf("[agent] "+format, args...)
}

// WebSocket keepalive tuning. The agent pings periodically; if no pong (or any
// other frame) arrives within pongWait the read fails and we reconnect — so a
// half-open TCP connection is detected in ~1 minute instead of hanging.
const (
	writeWait  = 10 * time.Second
	pongWait   = 60 * time.Second
	pingPeriod = 50 * time.Second // must be < pongWait
)

type Client struct {
	cfg         *types.Config
	monitor     *gpu.Monitor
	executor    *job.Executor
	gpuID       string
	backend     string
	deviceIndex int

	// baseCtx is the process-lifetime context; jobs run under it (not the
	// per-connection session ctx) so a brief reconnect doesn't abort them.
	baseCtx context.Context
	// outbox carries job results/progress and is drained by whichever session
	// is currently connected, so results survive a reconnect.
	outbox chan interface{}

	activeJobs      int64 // atomic
	cleanupRunning  int32
	jobSlot         chan struct{}
	outboxDir       string
	maintenanceGate *sync.RWMutex
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
	maintenanceGate := &sync.RWMutex{}
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
	return newClientForGPU(cfg, deviceIndex, &sync.RWMutex{})
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
	})
	if err != nil {
		return nil, err
	}

	c := &Client{
		cfg:             cfg,
		monitor:         monitor,
		executor:        executor,
		gpuID:           gpuID,
		backend:         backend,
		deviceIndex:     deviceIndex,
		outbox:          make(chan interface{}, 64),
		jobSlot:         make(chan struct{}, 1),
		outboxDir:       filepath.Join(cfg.ModelCacheDir, "outbox", gpuID),
		maintenanceGate: maintenanceGate,
	}
	if err := os.MkdirAll(c.outboxDir, 0o700); err != nil {
		return nil, fmt.Errorf("create result outbox: %w", err)
	}
	c.loadPendingResults()

	// Relay job progress to the coordinator (best-effort — dropped if the outbox
	// is full, so slow delivery never blocks inference).
	executor.OnProgress = func(p types.JobProgress) {
		log("job %s: [%s] %.0f%% — %s", p.JobID, p.Stage, p.Progress*100, p.Message)
		c.enqueue(p)
	}

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

// enqueue queues a message for delivery, dropping it if the outbox is full.
func (c *Client) enqueue(msg interface{}) {
	select {
	case c.outbox <- msg:
	default:
	}
}

// RunCustomAssetCleanup removes expired machine-level custom assets only while
// all GPU clients are idle. A single loop must be used for clients sharing a
// cache directory.
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
	if len(clients) > 0 && clients[0].maintenanceGate != nil {
		clients[0].maintenanceGate.Lock()
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

// Run connects and serves until ctx is cancelled, reconnecting with backoff.
func (c *Client) Run(ctx context.Context) error {
	c.baseCtx = ctx
	// On shutdown, tear down any containers/workspaces still running so we don't
	// leave detached containers holding the GPU.
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		c.executor.StopAll(cleanupCtx)
	}()

	backoff := 2 * time.Second
	const maxBackoff = 30 * time.Second

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		err := c.session(ctx)
		if ctx.Err() != nil {
			return ctx.Err()
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

// session runs one full connection lifecycle.
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

	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	log("connected to pool as %s", c.gpuID)

	// Session-scoped channel for register/heartbeat (these belong to this
	// connection). Job results/progress go through the persistent c.outbox.
	sendCh := make(chan interface{}, 16)
	go c.writer(ctx, conn, sendCh)

	// Register immediately.
	c.trySend(ctx, sendCh, c.registerMessage())

	// Heartbeat loop.
	go c.heartbeatLoop(ctx, sendCh)

	// Read loop (blocks until disconnect/error). A read deadline plus pong
	// handler detects a dead peer even when no application frames arrive.
	_ = conn.SetReadDeadline(time.Now().Add(pongWait))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(pongWait))
	})
	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			cancel()
			return err
		}
		// Any inbound frame proves liveness — extend the deadline.
		_ = conn.SetReadDeadline(time.Now().Add(pongWait))
		c.dispatch(ctx, sendCh, data)
	}
}

func (c *Client) writer(ctx context.Context, conn *websocket.Conn, sendCh <-chan interface{}) {
	ping := time.NewTicker(pingPeriod)
	defer ping.Stop()

	write := func(msg interface{}) bool {
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
		case msg := <-sendCh:
			if !write(msg) {
				return
			}
		case msg := <-c.outbox:
			if !write(msg) {
				// Delivery failed (disconnect). Requeue so the next session
				// delivers the result rather than losing it, then exit.
				c.enqueue(msg)
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

// dispatch routes an inbound message.
func (c *Client) dispatch(ctx context.Context, sendCh chan<- interface{}, data []byte) {
	var env types.Envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return
	}

	switch env.Type {
	case "job_assignment":
		var a types.JobAssignment
		if err := json.Unmarshal(data, &a); err != nil {
			log("bad job_assignment: %v", err)
			return
		}
		hasMaintenanceLease := c.maintenanceGate != nil && c.maintenanceGate.TryRLock()
		if c.maintenanceGate != nil && !hasMaintenanceLease {
			c.enqueue(types.JobResult{Type: "job_result", JobID: a.JobID, GPUID: c.gpuID, Error: "GPU is in maintenance"})
			return
		}
		select {
		case c.jobSlot <- struct{}{}:
			go c.runJob(a, hasMaintenanceLease)
		default:
			if hasMaintenanceLease {
				c.maintenanceGate.RUnlock()
			}
			c.enqueue(types.JobResult{Type: "job_result", JobID: a.JobID, GPUID: c.gpuID, Error: "GPU is already running a job"})
		}
	case "job_cancel", "job_stop", "stop_job":
		var jc types.JobControl
		if err := json.Unmarshal(data, &jc); err != nil || jc.JobID == "" {
			return
		}
		log("cancelling job %s", jc.JobID)
		go c.executor.Cancel(jc.JobID)
	case "job_result_ack":
		var ack struct {
			JobID string `json:"job_id"`
		}
		if json.Unmarshal(data, &ack) == nil && ack.JobID != "" {
			_ = os.Remove(c.resultPath(ack.JobID))
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
				if c.maintenanceGate != nil {
					c.maintenanceGate.Lock()
					defer c.maintenanceGate.Unlock()
				}
				preview, err = c.executor.ExecuteCleanup(cleanupCtx, request.Categories)
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
		// informational; ignore
	default:
		// unknown; ignore
	}
}

func (c *Client) runJob(a types.JobAssignment, hasMaintenanceLease bool) {
	if hasMaintenanceLease {
		defer c.maintenanceGate.RUnlock()
	}
	defer func() { <-c.jobSlot }()
	atomic.AddInt64(&c.activeJobs, 1)
	defer atomic.AddInt64(&c.activeJobs, -1)

	log("starting job %s (%s)", a.JobID, a.ModelName)
	// Run under the process-lifetime context so a reconnect doesn't abort the
	// job. Cancellation still happens via the executor (job_cancel / shutdown).
	result := c.executor.Execute(c.baseCtx, a)
	if result.Success {
		log("job %s completed in %dms", a.JobID, result.DurationMS)
	} else {
		log("job %s failed: %s", a.JobID, result.Error)
	}
	// Deliver via the persistent outbox (blocking, so the result is never
	// dropped — it waits for a live session if we're momentarily disconnected).
	if err := c.persistResult(result); err != nil {
		log("could not persist result for job %s: %v", a.JobID, err)
	}
	select {
	case <-c.baseCtx.Done():
	case c.outbox <- result:
	}
}

func (c *Client) resultPath(jobID string) string {
	return filepath.Join(c.outboxDir, filepath.Base(jobID)+".json")
}

func (c *Client) persistResult(result types.JobResult) error {
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}
	path := c.resultPath(result.JobID)
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, data, 0o600); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func (c *Client) loadPendingResults() {
	entries, err := os.ReadDir(c.outboxDir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(c.outboxDir, entry.Name()))
		if readErr != nil {
			continue
		}
		var result types.JobResult
		if json.Unmarshal(data, &result) != nil || result.JobID == "" {
			continue
		}
		c.enqueue(result)
	}
}

// ── Message builders ──────────────────────────────────────────────────────────

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
		break // primary GPU
	}
	return types.HeartbeatMessage{
		Type:            "gpu_heartbeat",
		GPUID:           c.gpuID,
		AvailableVRAMGB: availGB,
		CurrentJobs:     int(atomic.LoadInt64(&c.activeJobs)),
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

// endpoint converts the configured pool URL into a ws(s):// agent endpoint
// with a non-secret GPU identity query parameter. Authentication is sent in
// the Authorization header during the WebSocket upgrade.
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

// trySend pushes a message unless the context is done.
func (c *Client) trySend(ctx context.Context, sendCh chan<- interface{}, msg interface{}) {
	select {
	case <-ctx.Done():
	case sendCh <- msg:
	}
}
