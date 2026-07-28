// Package pool implements the agent's connection to the coordinator over raw
// WebSocket: outbound dial (NAT/firewall friendly), register, heartbeat, and a
// job loop that executes assignments and reports results. Reconnects with
// backoff on disconnect.
package pool

import (
	"context"
	"encoding/json"
	stdlog "log"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/RunGPU-io/gpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/gpu-agent/internal/gpu"
	"github.com/RunGPU-io/gpu-agent/internal/job"
	"github.com/RunGPU-io/gpu-agent/internal/types"
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
	cfg      *types.Config
	monitor  *gpu.Monitor
	executor *job.Executor
	gpuID    string
	backend  string

	// baseCtx is the process-lifetime context; jobs run under it (not the
	// per-connection session ctx) so a brief reconnect doesn't abort them.
	baseCtx context.Context
	// outbox carries job results/progress and is drained by whichever session
	// is currently connected, so results survive a reconnect.
	outbox chan interface{}

	activeJobs int64 // atomic
}

func NewClient(cfg *types.Config) (*Client, error) {
	gpuID := "gpu-0"
	if len(cfg.GPUIDs) > 0 {
		gpuID = cfg.GPUIDs[0]
	}
	monitor := gpu.NewMonitor()
	backend := monitor.Backend()
	executor, err := job.NewExecutorWithOptions(job.ExecutorOptions{
		CacheDir:           cfg.ModelCacheDir,
		MaxCacheGB:         cfg.MaxModelCacheGB,
		GPUID:              gpuID,
		Backend:            backend,
		GPUDevice:          cfg.GPUDevice,
		JobTimeout:         time.Duration(cfg.JobTimeoutMinutes) * time.Minute,
		MaxCustomFileBytes: int64(cfg.MaxCustomFileGB) * 1024 * 1024 * 1024,
		Policy:             dockermgr.PolicyFromConfig(cfg.Security),
		HFToken:            cfg.Security.HFToken,
	})
	if err != nil {
		return nil, err
	}

	c := &Client{
		cfg:      cfg,
		monitor:  monitor,
		executor: executor,
		gpuID:    gpuID,
		backend:  backend,
		outbox:   make(chan interface{}, 64),
	}

	// Relay job progress to the coordinator (best-effort — dropped if the outbox
	// is full, so slow delivery never blocks inference).
	executor.OnProgress = func(p types.JobProgress) {
		log("job %s: [%s] %.0f%% — %s", p.JobID, p.Stage, p.Progress*100, p.Message)
		c.enqueue(p)
	}

	return c, nil
}

// enqueue queues a message for delivery, dropping it if the outbox is full.
func (c *Client) enqueue(msg interface{}) {
	select {
	case c.outbox <- msg:
	default:
	}
}

// Run connects and serves until ctx is cancelled, reconnecting with backoff.
func (c *Client) Run(ctx context.Context) error {
	c.baseCtx = ctx
	// On shutdown, tear down any containers/workspaces still running so we don't
	// leave detached containers holding the GPU.
	defer c.executor.StopAll(context.Background())

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

	conn, _, err := websocket.DefaultDialer.DialContext(parent, endpoint, nil)
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
		go c.runJob(a)
	case "job_cancel", "job_stop", "stop_job":
		var jc types.JobControl
		if err := json.Unmarshal(data, &jc); err != nil || jc.JobID == "" {
			return
		}
		log("cancelling job %s", jc.JobID)
		go c.executor.Cancel(jc.JobID)
	case "gpu_register_ack", "gpu_heartbeat_ack", "job_result_ack", "pool_metrics":
		// informational; ignore
	default:
		// unknown; ignore
	}
}

func (c *Client) runJob(a types.JobAssignment) {
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
	select {
	case <-c.baseCtx.Done():
	case c.outbox <- result:
	}
}

// ── Message builders ──────────────────────────────────────────────────────────

func (c *Client) registerMessage() types.RegisterMessage {
	gpus := c.monitor.GPUs()
	var name, driver string
	var vramGB float64
	if len(gpus) > 0 {
		name = gpus[0].Name
		driver = gpus[0].DriverVersion
		vramGB = float64(gpus[0].MemoryMB) / 1024.0
	}
	host, _ := os.Hostname()
	sysInfo := gpu.DetectSystem()

	return types.RegisterMessage{
		Type:           "gpu_register",
		GPUID:          c.gpuID,
		Hostname:       host,
		GPUType:        name,
		Backend:        c.backend,
		VRAMGB:         vramGB,
		PricePerMinute: c.cfg.PricePerMinute,
		ModelsCached:   c.cachedModels(),
		DriverVersion:  driver,
		CPUModel:       sysInfo.CPUModel,
		CPUCores:       sysInfo.CPUCores,
		RAMTotalGB:     sysInfo.RAMTotalGB,
		OSInfo:         sysInfo.OSInfo,
		Capabilities:   gpu.RuntimeCapabilities(),
		OllamaModels:   gpu.OllamaModels(),
	}
}

func (c *Client) heartbeatMessage() types.HeartbeatMessage {
	var availGB float64
	for _, m := range c.monitor.CollectMetrics() {
		if m.MemoryTotalMB > 0 {
			availGB = float64(m.MemoryTotalMB-m.MemoryUsedMB) / 1024.0
		}
		break // primary GPU
	}
	ramUsed, ramTotal := gpu.RAMUsage()

	return types.HeartbeatMessage{
		Type:            "gpu_heartbeat",
		GPUID:           c.gpuID,
		AvailableVRAMGB: availGB,
		CurrentJobs:     int(atomic.LoadInt64(&c.activeJobs)),
		ModelsCached:    c.cachedModels(),
		RAMUsedGB:       ramUsed,
		RAMTotalGB:      ramTotal,
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
// with auth query params.
func (c *Client) endpoint() (string, error) {
	u, err := url.Parse(c.cfg.PoolURL)
	if err != nil {
		return "", err
	}
	switch strings.ToLower(u.Scheme) {
	case "https", "wss":
		u.Scheme = "wss"
	default:
		u.Scheme = "ws"
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/agent"
	q := u.Query()
	q.Set("api_key", c.cfg.APIKey)
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
