package pool

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
	"github.com/gorilla/websocket"
)

type registrationPeer struct {
	conn   *websocket.Conn
	frames chan types.JobResult
	done   chan error
	cancel context.CancelFunc
}

func connectRegistrationPeer(t *testing.T, c *Client, timeout time.Duration) *registrationPeer {
	t.Helper()
	accepted := make(chan *websocket.Conn, 1)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err == nil {
			accepted <- conn
		}
	}))
	t.Cleanup(server.Close)
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	peer := <-accepted
	if err := conn.WriteJSON(map[string]interface{}{"type": "gpu_register", "gpu_id": "test-gpu"}); err != nil {
		t.Fatal(err)
	}
	var registration types.Envelope
	if err := peer.ReadJSON(&registration); err != nil || registration.Type != "gpu_register" {
		t.Fatalf("registration: %+v, %v", registration, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	p := &registrationPeer{conn: peer, frames: make(chan types.JobResult, 256), done: make(chan error, 1), cancel: cancel}
	go func() { p.done <- c.serveConnection(ctx, conn, timeout) }()
	readDone := make(chan struct{})
	go func() {
		defer close(readDone)
		defer close(p.frames)
		for {
			var frame types.JobResult
			if err := peer.ReadJSON(&frame); err != nil {
				return
			}
			select {
			case p.frames <- frame:
			case <-ctx.Done():
				return
			}
		}
	}()
	t.Cleanup(func() {
		cancel()
		_ = peer.Close()
		_ = conn.Close()
		select {
		case <-p.done:
		case <-time.After(3 * time.Second):
			t.Error("connection did not stop")
		}
		select {
		case <-readDone:
		case <-time.After(3 * time.Second):
			t.Error("peer reader did not stop")
		}
	})
	return p
}

func (p *registrationPeer) send(t *testing.T, data string) {
	t.Helper()
	if err := p.conn.WriteMessage(websocket.TextMessage, []byte(data)); err != nil {
		t.Fatal(err)
	}
}

func (p *registrationPeer) quiet(t *testing.T) {
	t.Helper()
	select {
	case frame, ok := <-p.frames:
		t.Fatalf("traffic before positive registration ACK: %+v (open=%t)", frame, ok)
	case <-time.After(100 * time.Millisecond):
	}
}

func (p *registrationPeer) stopped(t *testing.T) error {
	t.Helper()
	select {
	case err := <-p.done:

		p.done <- err
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("connection did not stop")
		return nil
	}
}

func registrationClient(t *testing.T) *Client {
	t.Helper()
	c := resultTestClient(t)

	c.cfg = &types.Config{HeartbeatIntervalSecs: 3600}
	return c
}

func TestRegistrationDelayedAckGatesDiskBacklogAndBestEffort(t *testing.T) {
	c := registrationClient(t)
	const count = 2*resultBatchSize + 7
	for i := 0; i < count; i++ {
		if err := c.persistResult(types.JobResult{Type: "job_result", JobID: fmt.Sprintf("disk-%d", i)}); err != nil {
			t.Fatal(err)
		}
	}
	c.enqueue(types.JobProgress{Type: "job_progress", JobID: "progress"})
	c.enqueue(types.JobResult{Type: "job_result", JobID: "best-effort"})
	p := connectRegistrationPeer(t, c, 5*time.Second)
	p.quiet(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "during-registration"})

	p.send(t, `{"type":"gpu_heartbeat_ack","success":true}`)
	p.send(t, `{"type":"job_result_ack","job_id":"during-registration","success":true}`)
	p.quiet(t)
	if _, err := os.Stat(c.resultPath("during-registration")); err != nil {
		t.Fatalf("premature result ACK removed the result: %v", err)
	}
	p.send(t, `{"type":"gpu_register_ack","success":true,"gpu_id":"test-gpu","message":"GPU registered successfully"}`)
	results, progress := make(map[string]bool), 0
	deadline := time.After(3 * time.Second)
	for len(results) < count+2 || progress < 1 {
		select {
		case frame, ok := <-p.frames:
			if !ok {
				t.Fatal("connection closed during replay")
			}
			switch frame.Type {
			case "job_result":
				if results[frame.JobID] {
					t.Fatalf("duplicate result before backlog drained: %s", frame.JobID)
				}
				results[frame.JobID] = true
				p.send(t, fmt.Sprintf(`{"type":"job_result_ack","job_id":%q,"success":true}`, frame.JobID))
			case "job_progress":
				progress++
			default:
				t.Fatalf("unexpected frame: %+v", frame)
			}
		case <-deadline:
			t.Fatalf("replay stalled: results=%d progress=%d", len(results), progress)
		}
	}
}

func TestRegistrationFailureRetainsResultsForReconnect(t *testing.T) {
	for _, failure := range []string{"rejected", "missing-success", "timeout-with-traffic", "disconnect", "cancel"} {
		t.Run(failure, func(t *testing.T) {
			c := registrationClient(t)
			if err := c.persistResult(types.JobResult{Type: "job_result", JobID: "reconnect"}); err != nil {
				t.Fatal(err)
			}
			c.enqueue(types.JobProgress{Type: "job_progress", JobID: "reconnect"})
			wait := 5 * time.Second
			if failure == "timeout-with-traffic" {
				wait = 200 * time.Millisecond
			}
			p := connectRegistrationPeer(t, c, wait)
			switch failure {
			case "rejected":
				p.send(t, `{"type":"gpu_register_ack","success":false,"error":"registration unavailable"}`)
			case "missing-success":
				p.send(t, `{"type":"gpu_register_ack"}`)
			case "timeout-with-traffic":
				p.send(t, `{"type":"gpu_register_ack","success":"true"}`)
				trafficDone := make(chan struct{})
				go func() {
					defer close(trafficDone)
					ticker := time.NewTicker(10 * time.Millisecond)
					defer ticker.Stop()
					for range ticker.C {
						if err := p.conn.WriteMessage(websocket.PongMessage, nil); err != nil {
							return
						}
						if err := p.conn.WriteJSON(map[string]interface{}{"type": "gpu_heartbeat_ack", "success": true}); err != nil {
							return
						}
					}
				}()
				defer func() { _ = p.conn.Close(); <-trafficDone }()
			case "disconnect":
				_ = p.conn.Close()
			case "cancel":
				p.cancel()
			}
			err := p.stopped(t)
			if err == nil {
				t.Fatal("registration failure returned success")
			}
			if failure == "timeout-with-traffic" && !strings.Contains(err.Error(), "registration acknowledgement timeout") {
				t.Fatalf("unexpected timeout error: %v", err)
			}
			if (failure == "rejected" || failure == "missing-success") && !strings.Contains(err.Error(), "registration rejected") {
				t.Fatalf("unexpected rejection error: %v", err)
			}
			if failure == "cancel" && err != context.Canceled {
				t.Fatalf("unexpected cancellation error: %v", err)
			}
			for frame := range p.frames {
				t.Fatalf("registration failure sent queued traffic: %+v", frame)
			}
			if _, err := os.Stat(c.resultPath("reconnect")); err != nil {
				t.Fatalf("registration failure lost disk result: %v", err)
			}
			next := connectRegistrationPeer(t, c, 5*time.Second)
			next.quiet(t)
			next.send(t, `{"type":"gpu_register_ack","success":true}`)
			replayed := make(map[string]bool)
			for i := 0; i < 2; i++ {
				select {
				case frame, ok := <-next.frames:
					if !ok || frame.JobID != "reconnect" || replayed[frame.Type] ||
						(frame.Type != "job_result" && frame.Type != "job_progress") {
						t.Fatalf("unexpected replay: %+v (open=%t)", frame, ok)
					}
					replayed[frame.Type] = true
				case <-time.After(3 * time.Second):
					t.Fatal("reconnect did not replay queued traffic")
				}
			}
			next.send(t, `{"type":"job_result_ack","job_id":"reconnect","success":true}`)
			deadline := time.Now().Add(3 * time.Second)
			for {
				if _, err := os.Stat(c.resultPath("reconnect")); os.IsNotExist(err) {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("positive result ACK did not remove durable result")
				}
				time.Sleep(time.Millisecond)
			}
		})
	}
}

func TestRegistrationReconnectRequiresNewAck(t *testing.T) {
	c := registrationClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "unacknowledged"})
	first := connectRegistrationPeer(t, c, 5*time.Second)
	first.send(t, `{"type":"gpu_register_ack","success":true}`)
	select {
	case frame := <-first.frames:
		if frame.Type != "job_result" || frame.JobID != "unacknowledged" {
			t.Fatalf("unexpected first result: %+v", frame)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("first connection did not send result")
	}
	_ = first.conn.Close()
	_ = first.stopped(t)
	if _, err := os.Stat(c.resultPath("unacknowledged")); err != nil {
		t.Fatalf("disconnect lost unacknowledged result: %v", err)
	}
	c.enqueue(types.JobProgress{Type: "job_progress", JobID: "unacknowledged"})
	next := connectRegistrationPeer(t, c, 5*time.Second)
	next.quiet(t)
	next.send(t, `{"type":"gpu_register_ack","success":true}`)
	seen := make(map[string]bool)
	for len(seen) < 2 {
		select {
		case frame, ok := <-next.frames:
			if !ok || frame.JobID != "unacknowledged" ||
				(frame.Type != "job_result" && frame.Type != "job_progress") {
				t.Fatalf("unexpected replay: %+v (open=%t)", frame, ok)
			}
			seen[frame.Type] = true
		case <-time.After(3 * time.Second):
			t.Fatal("new registration did not enable replay")
		}
	}
}

func TestRegistrationWriterKeepsHeartbeatTransport(t *testing.T) {
	upgrader := websocket.Upgrader{}
	accepted := make(chan *websocket.Conn, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err == nil {
			accepted <- conn
		}
	}))
	defer server.Close()
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	peer := <-accepted
	defer peer.Close()
	c := registrationClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "pending"})
	c.enqueue(types.JobProgress{Type: "job_progress"})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	defer func() { cancel(); <-done }()
	sendCh := make(chan interface{}, 1)
	go func() {
		defer close(done)
		c.writer(ctx, conn, sendCh, make(chan struct{}))
	}()
	sendCh <- types.HeartbeatMessage{Type: "gpu_heartbeat", GPUID: "test-gpu"}
	_ = peer.SetReadDeadline(time.Now().Add(3 * time.Second))
	_, data, err := peer.ReadMessage()
	if err != nil {
		t.Fatal(err)
	}
	var frame types.Envelope
	if err := json.Unmarshal(data, &frame); err != nil || frame.Type != "gpu_heartbeat" {
		t.Fatalf("expected heartbeat while registration pending, got %s (%v)", data, err)
	}
}
