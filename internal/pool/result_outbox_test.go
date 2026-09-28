package pool

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
	"github.com/gorilla/websocket"
)

func resultTestClient(t *testing.T) *Client {
	t.Helper()
	return &Client{
		outboxDir:    t.TempDir(),
		outbox:       make(chan interface{}, 64),
		resultsReady: make(chan struct{}, 1),
	}
}

func acceptResult(c *Client, jobID string) {
	c.dispatch(context.Background(), nil, []byte(fmt.Sprintf(
		`{"type":"job_result_ack","job_id":%q,"success":true}`, jobID)))
}

func TestResultOutboxTokenAckFencing(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "fenced", DispatchToken: "attempt-current", Success: true})
	restarted := &Client{outboxDir: c.outboxDir}
	restarted.loadPendingResults()
	forgotten := 0
	restarted.executor = &cancellationExecutor{forget: func(id string) {
		if id != "fenced" {
			t.Errorf("forgot wrong execution: %q", id)
		}
		forgotten++
	}}
	original := restarted.results["fenced"]
	for _, fields := range []string{
		`"success":true`,
		`"success":true,"dispatch_token":""`,
		`"success":true,"dispatch_token":"attempt-old"`,
		`"success":true,"dispatch_token":42`,
		`"success":true,"dispatch_token":null`,
		`"success":false,"dispatch_token":"attempt-current"`,
	} {
		restarted.dispatch(context.Background(), nil, []byte(`{"type":"job_result_ack","job_id":"fenced",`+fields+`}`))
		if restarted.results["fenced"] != original || forgotten != 0 {
			t.Fatalf("nonmatching ACK discarded result or tracking: %s", fields)
		}
		if _, err := os.Stat(restarted.resultPath("fenced")); err != nil {
			t.Fatalf("nonmatching ACK deleted durable result: %v", err)
		}
	}
	sends := 0
	now := time.Now()
	for _, at := range []time.Time{now, now.Add(resultRetryMin)} {
		restarted.sendPendingResults(context.Background(), at, func(msg interface{}) bool {
			result := msg.(types.JobResult)
			if result.DispatchToken != "attempt-current" {
				t.Fatalf("retry lost token: %+v", result)
			}
			sends++
			return true
		})
	}
	if sends != 2 {
		t.Fatalf("unacknowledged result retried %d times, want 2", sends)
	}
	ack := []byte(`{"type":"job_result_ack","job_id":"fenced","success":true,"dispatch_token":"attempt-current"}`)
	restarted.dispatch(context.Background(), nil, ack)
	restarted.dispatch(context.Background(), nil, ack)
	if len(restarted.results) != 0 || forgotten != 1 {
		t.Fatalf("matching ACK did not remove exactly one result: forgotten=%d", forgotten)
	}
	if _, err := os.Stat(restarted.resultPath("fenced")); !os.IsNotExist(err) {
		t.Fatalf("matching ACK left durable result: %v", err)
	}
}

func TestResultOutboxLateTokenAckAndReplayPreserveCurrentAttempt(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "redispatched", DispatchToken: "old", Success: false})
	oldACK := []byte(`{"type":"job_result_ack","job_id":"redispatched","success":true,"dispatch_token":"old"}`)
	c.dispatch(context.Background(), nil, oldACK)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "redispatched", DispatchToken: "new", Success: true})
	original := c.results["redispatched"]
	for _, token := range []string{"old", "", "new"} {
		c.queueResult(types.JobResult{Type: "job_result", JobID: "redispatched", DispatchToken: token, Error: "replay"})
	}
	for _, data := range []string{
		`{"type":"job_assignment","job_id":"redispatched","dispatch_token":"old","model_name":"test","runtime":"ollama"}`,
		`{"type":"job_assignment","job_id":"redispatched","dispatch_token":"old"}`,
	} {

		c.dispatch(context.Background(), nil, []byte(data))
	}
	c.dispatch(context.Background(), nil, oldACK)
	acceptResult(c, "redispatched")
	if c.results["redispatched"] != original || !original.result.Success {
		t.Fatal("late replay or stale ACK replaced the current attempt")
	}
	restarted := &Client{outboxDir: c.outboxDir}
	restarted.loadPendingResults()
	if pending := restarted.results["redispatched"]; pending == nil ||
		pending.result.DispatchToken != "new" || !pending.result.Success {
		t.Fatalf("replay replaced current result on disk: %+v", pending)
	}
}

func TestResultOutboxLegacyTokenAckCompatibility(t *testing.T) {
	for _, token := range []string{"", "coordinator-token"} {
		t.Run("ack="+token, func(t *testing.T) {
			c := resultTestClient(t)
			c.queueResult(types.JobResult{Type: "job_result", JobID: "legacy", Success: true})
			original := c.results["legacy"]
			c.queueResult(types.JobResult{Type: "job_result", JobID: "legacy", DispatchToken: "new", Error: "replay"})
			if c.results["legacy"] != original {
				t.Fatal("new token replaced an unacknowledged legacy result")
			}

			c = &Client{outboxDir: c.outboxDir}
			c.loadPendingResults()
			c.dispatch(context.Background(), nil, []byte(fmt.Sprintf(
				`{"type":"job_result_ack","job_id":"legacy","success":true,"dispatch_token":%q}`, token)))
			if len(c.results) != 0 {
				t.Fatal("legacy result required an exact ACK token")
			}
			if _, err := os.Stat(c.resultPath("legacy")); !os.IsNotExist(err) {
				t.Fatalf("legacy ACK left durable result: %v", err)
			}
		})
	}
}

func TestResultOutboxRetriesUntilPositiveAck(t *testing.T) {
	for _, negativeAck := range []bool{false, true} {
		t.Run(fmt.Sprintf("negative_ack=%t", negativeAck), func(t *testing.T) {
			c := resultTestClient(t)
			c.queueResult(types.JobResult{Type: "job_result", JobID: "retry", Success: true})
			now := time.Now()
			writes := 0
			write := func(msg interface{}) bool {
				writes++
				if msg.(types.JobResult).JobID != "retry" {
					t.Fatal("wrong result")
				}
				if _, err := os.Stat(c.resultPath("retry")); err != nil {
					t.Fatalf("result not persisted before send: %v", err)
				}
				if negativeAck {
					c.dispatch(context.Background(), nil, []byte(
						`{"type":"job_result_ack","job_id":"retry","success":false}`))
				}
				return true
			}
			for i, delay := range []time.Duration{
				2 * time.Second, 4 * time.Second, 8 * time.Second,
				16 * time.Second, 30 * time.Second, 30 * time.Second,
			} {
				if !c.sendPendingResults(context.Background(), now, write) {
					t.Fatal("send failed")
				}
				if writes != i+1 {
					t.Fatalf("writes = %d, want %d", writes, i+1)
				}
				c.sendPendingResults(context.Background(), now.Add(delay-time.Nanosecond), write)
				if writes != i+1 {
					t.Fatal("result retried before backoff elapsed")
				}
				now = now.Add(delay)
			}
			acceptResult(c, "retry")
			c.sendPendingResults(context.Background(), now.Add(time.Hour), write)
			c.loadPendingResults()
			c.sendPendingResults(context.Background(), now.Add(time.Hour), write)
			if writes != 6 {
				t.Fatal("accepted result retransmitted")
			}
		})
	}
}

func TestResultOutboxRecoversLargeBacklogInBoundedBatches(t *testing.T) {
	c := resultTestClient(t)
	const count = 2*resultBatchSize + 7
	for i := 0; i < count; i++ {
		if err := c.persistResult(types.JobResult{
			Type: "job_result", JobID: fmt.Sprintf("job-%03d", i), Success: true,
		}); err != nil {
			t.Fatal(err)
		}
	}
	for name, data := range map[string]string{
		"invalid.json": "{", "empty.json": `{}`, "interrupted.json.tmp": `{"job_id":"partial"}`,
	} {
		if err := os.WriteFile(filepath.Join(c.outboxDir, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	loaded := make(chan struct{})
	go func() {
		c.loadPendingResults()
		close(loaded)
	}()
	select {
	case <-loaded:
	case <-time.After(3 * time.Second):
		t.Fatal("recovery blocked on a full channel")
	}
	if len(c.results) != count || len(c.outbox) != 0 {
		t.Fatalf("recovered %d results and queued %d transient messages", len(c.results), len(c.outbox))
	}
	seen := make(map[string]bool)
	now := time.Now()
	for batch, want := range []int{resultBatchSize, resultBatchSize, 7} {
		writes := 0
		c.sendPendingResults(context.Background(), now, func(msg interface{}) bool {
			result := msg.(types.JobResult)
			if seen[result.JobID] {
				t.Fatalf("result %s repeated before backlog drained", result.JobID)
			}
			seen[result.JobID] = true
			writes++

			return true
		})
		if writes != want {
			t.Fatalf("batch %d writes = %d, want %d", batch, writes, want)
		}
	}
	if len(seen) != count {
		t.Fatalf("delivered %d results, want %d", len(seen), count)
	}
	for jobID := range seen {
		acceptResult(c, jobID)
	}
	c.loadPendingResults()
	if len(c.results) != 0 {
		t.Fatal("accepted backlog recovered again")
	}
}

func TestResultOutboxBacklogDoesNotStarveBehindRetries(t *testing.T) {
	c := resultTestClient(t)
	for i := 0; i < resultBatchSize+1; i++ {
		c.queueResult(types.JobResult{Type: "job_result", JobID: fmt.Sprintf("job-%03d", i)})
	}
	now := time.Now()
	c.sendPendingResults(context.Background(), now, func(interface{}) bool { return true })
	first := ""
	c.sendPendingResults(context.Background(), now.Add(resultRetryMin), func(msg interface{}) bool {
		if first == "" {
			first = msg.(types.JobResult).JobID
		}
		return true
	})
	if first != fmt.Sprintf("job-%03d", resultBatchSize) {
		t.Fatalf("unsent result starved behind retries: first = %q", first)
	}
}

func TestResultOutboxWriteFailureAndReconnectRecovery(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "in-memory"})
	now := time.Now()
	if c.sendPendingResults(context.Background(), now, func(interface{}) bool { return false }) {
		t.Fatal("write failure not reported")
	}
	if _, err := os.Stat(c.resultPath("in-memory")); err != nil {
		t.Fatalf("failed write lost durable result: %v", err)
	}

	if err := c.persistResult(types.JobResult{Type: "job_result", JobID: "disk-only"}); err != nil {
		t.Fatal(err)
	}
	c.loadPendingResults()
	seen := make(map[string]bool)
	c.sendPendingResults(context.Background(), now, func(msg interface{}) bool {
		id := msg.(types.JobResult).JobID
		seen[id] = true
		acceptResult(c, id)
		return true
	})
	if !seen["in-memory"] || !seen["disk-only"] || len(c.results) != 0 {
		t.Fatalf("reconnect did not recover and acknowledge all results: %v", seen)
	}
}

func TestResultOutboxRetriesPersistenceFailure(t *testing.T) {
	c := resultTestClient(t)
	c.outboxDir = filepath.Join(c.outboxDir, "unavailable")
	c.queueResult(types.JobResult{Type: "job_result", JobID: "disk-failure"})
	if len(c.results) != 1 {
		t.Fatal("persistence failure dropped in-memory result")
	}
	if err := os.Mkdir(c.outboxDir, 0o700); err != nil {
		t.Fatal(err)
	}
	c.loadPendingResults()
	c.sendPendingResults(context.Background(), time.Now(), func(interface{}) bool { return true })
	if _, err := os.Stat(c.resultPath("disk-failure")); err != nil {
		t.Fatalf("result not persisted after disk recovered: %v", err)
	}
}

func TestResultOutboxAckDeletionFailureRetainsResult(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "delete-failure", Success: true})
	path := c.resultPath("delete-failure")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}

	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	blocker := filepath.Join(path, "blocker")
	if err := os.WriteFile(blocker, []byte("blocked"), 0o600); err != nil {
		t.Fatal(err)
	}
	acceptResult(c, "delete-failure")
	c.loadPendingResults()
	if len(c.results) != 1 {
		t.Fatal("failed ACK cleanup discarded the pending result")
	}
	if err := os.Remove(blocker); err != nil {
		t.Fatal(err)
	}
	acceptResult(c, "delete-failure")
	c.loadPendingResults()
	if len(c.results) != 0 {
		t.Fatal("acknowledged result returned after cleanup recovered")
	}
}

func TestResultOutboxReconnectKeepsNewerMemoryResult(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "replacement", Success: false})
	temporary := c.resultPath("replacement") + ".tmp"
	if err := os.Mkdir(temporary, 0o700); err != nil {
		t.Fatal(err)
	}
	c.queueResult(types.JobResult{Type: "job_result", JobID: "replacement", Success: true})
	c.loadPendingResults()
	if !c.results["replacement"].result.Success {
		t.Fatal("older disk result replaced the newer memory result")
	}
	if err := os.Remove(temporary); err != nil {
		t.Fatal(err)
	}
	writes := 0
	c.sendPendingResults(context.Background(), time.Now(), func(msg interface{}) bool {
		writes++
		if !msg.(types.JobResult).Success {
			t.Fatal("sent superseded result")
		}
		return true
	})
	restarted := &Client{outboxDir: c.outboxDir}
	restarted.loadPendingResults()
	if writes != 1 || !restarted.results["replacement"].result.Success {
		t.Fatal("newer result was not durably recovered")
	}
}

func TestResultOutboxConcurrentQueueAndAck(t *testing.T) {
	c := resultTestClient(t)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			c.queueResult(types.JobResult{Type: "job_result", JobID: fmt.Sprintf("job-%d", i)})
		}
	}()
	write := func(msg interface{}) bool {
		acceptResult(c, msg.(types.JobResult).JobID)
		return true
	}
	for i := 0; i < 100; i++ {
		c.sendPendingResults(context.Background(), time.Now(), write)
	}
	<-done
	for i := 0; i < 2; i++ {
		c.sendPendingResults(context.Background(), time.Now(), write)
	}
	if len(c.results) != 0 {
		t.Fatalf("%d results remain unacknowledged", len(c.results))
	}
}

func TestResultOutboxQueuesWhileDisconnectedAndCancelled(t *testing.T) {
	c := resultTestClient(t)
	for i := 0; i < cap(c.outbox); i++ {
		c.outbox <- types.JobProgress{}
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < resultBatchSize+1; i++ {
			c.queueResult(types.JobResult{Type: "job_result", JobID: fmt.Sprintf("job-%d", i)})
		}
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("result producer blocked on disconnected/full outbox")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if c.sendPendingResults(ctx, time.Now(), func(interface{}) bool {
		t.Fatal("cancelled writer sent a result")
		return true
	}) {
		t.Fatal("cancelled send was not stopped")
	}
	recovered := &Client{outboxDir: c.outboxDir}
	recovered.loadPendingResults()
	if len(recovered.results) != resultBatchSize+1 {
		t.Fatal("shutdown lost pending results")
	}
}

func TestResultOutboxWriterRetriesAndReconnects(t *testing.T) {
	upgrader := websocket.Upgrader{}
	peers := make(chan *websocket.Conn, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err == nil {
			peers <- conn
		}
	}))
	defer server.Close()
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "socket", Success: true})

	connect := func() (*websocket.Conn, func()) {
		t.Helper()
		conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http"), nil)
		if err != nil {
			t.Fatal(err)
		}
		peer := <-peers
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		registered := make(chan struct{})
		close(registered)
		go func() {
			defer close(done)
			c.writer(ctx, conn, nil, registered)
		}()
		readDone := make(chan struct{})
		go func() {
			defer close(readDone)
			for {
				_, data, err := conn.ReadMessage()
				if err != nil {
					return
				}
				c.dispatch(ctx, nil, data)
			}
		}()
		return peer, func() {
			cancel()
			defer peer.Close()
			defer conn.Close()
			select {
			case <-done:
			case <-time.After(3 * time.Second):
				t.Fatal("writer did not stop on cancellation")
			}
			select {
			case <-readDone:
			case <-time.After(3 * time.Second):
				t.Fatal("writer exit did not unblock the session reader")
			}
		}
	}
	readResult := func(peer *websocket.Conn) {
		t.Helper()
		_ = peer.SetReadDeadline(time.Now().Add(5 * time.Second))
		var result types.JobResult
		if err := peer.ReadJSON(&result); err != nil {
			t.Fatalf("read result: %v", err)
		}
		if result.JobID != "socket" || !result.Success {
			t.Fatalf("unexpected result: %+v", result)
		}
		if _, err := os.Stat(c.resultPath("socket")); err != nil {
			t.Fatalf("unacknowledged socket write lost durable result: %v", err)
		}
	}
	peer, stop := connect()
	readResult(peer)

	readResult(peer)
	stop()
	peer, stop = connect()
	defer stop()
	readResult(peer)
	if err := peer.WriteJSON(map[string]interface{}{
		"type": "job_result_ack", "job_id": "socket", "success": true,
	}); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		c.resultsMu.Lock()
		pending := len(c.results)
		c.resultsMu.Unlock()
		if pending == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("socket ACK did not clear pending result")
		}
		time.Sleep(time.Millisecond)
	}
	if _, err := os.Stat(c.resultPath("socket")); !os.IsNotExist(err) {
		t.Fatalf("positive socket ACK did not remove result: %v", err)
	}
}
