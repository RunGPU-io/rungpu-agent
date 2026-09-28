package pool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type retryExecution struct {
	assignment types.JobAssignment
	results    chan types.JobResult
	finished   bool
}

type retryExecutor struct {
	*job.Executor
	mu      sync.Mutex
	tracked map[string]*retryExecution
	starts  chan types.JobAssignment
}

func newRetryClient(t *testing.T) (*Client, *retryExecutor) {
	t.Helper()
	c := resultTestClient(t)
	c.baseCtx = context.Background()
	c.jobSlot = make(chan struct{}, 1)
	e := &retryExecutor{
		Executor: &job.Executor{}, tracked: make(map[string]*retryExecution),
		starts: make(chan types.JobAssignment, 8),
	}
	c.executor = e
	return c, e
}

func (e *retryExecutor) Runtime() string { return "retry-fixture" }
func (e *retryExecutor) IsTracked(id string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.tracked[id] != nil
}
func (e *retryExecutor) TrackedDispatchToken(id string) (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if state := e.tracked[id]; state != nil {
		return state.assignment.DispatchToken, true
	}
	return "", false
}
func (e *retryExecutor) Start(_ context.Context, a types.JobAssignment) (<-chan types.JobResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.tracked[a.JobID] != nil {
		return nil, job.ErrAlreadyTracked
	}
	state := &retryExecution{assignment: a, results: make(chan types.JobResult, 1)}
	e.tracked[a.JobID] = state
	e.starts <- a
	return state.results, nil
}
func (e *retryExecutor) ForgetExecution(id string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if state := e.tracked[id]; state != nil && state.finished {
		delete(e.tracked, id)
	}
}
func (e *retryExecutor) finish(id string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	state := e.tracked[id]
	state.finished = true
	state.results <- types.JobResult{Type: "job_result", JobID: id, Success: true, DispatchToken: state.assignment.DispatchToken}
}

func dispatchRetry(c *Client, id, token string) {
	data, _ := json.Marshal(types.JobAssignment{
		Type: "job_assignment", JobID: id, DispatchToken: token, ModelName: "fixture", Runtime: "ollama",
	})
	c.dispatch(context.Background(), nil, data)
}

func ackRetry(c *Client, id, token string) {
	data, _ := json.Marshal(map[string]interface{}{
		"type": "job_result_ack", "job_id": id, "dispatch_token": token, "success": true,
	})
	c.dispatch(context.Background(), nil, data)
}

func waitRetryResult(t *testing.T, c *Client, id, token string) types.JobResult {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		c.assignmentsMu.Lock()
		c.resultsMu.Lock()
		pending := c.results[id]
		ready := pending != nil && pending.result.DispatchToken == token
		c.resultsMu.Unlock()
		c.assignmentsMu.Unlock()
		if ready {
			return pending.result
		}
		if time.Now().After(deadline) {
			t.Fatalf("missing result for %s/%s", id, token)
		}
		time.Sleep(time.Millisecond)
	}
}

func expectRetryStart(t *testing.T, e *retryExecutor, id, token string) {
	t.Helper()
	select {
	case a := <-e.starts:
		if a.JobID != id || a.DispatchToken != token {
			t.Fatalf("wrong execution started: %+v", a)
		}
	case <-time.After(time.Second):
		t.Fatalf("missing execution for %s/%s", id, token)
	}
}

func TestDeferredAssignmentWaitsForMatchingAckAcrossReconnect(t *testing.T) {
	c, e := newRetryClient(t)
	dispatchRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "A")
	e.finish("retry")

	select {
	case <-c.resultsReady:
	case <-time.After(time.Second):
		t.Fatal("completion did not signal the result writer")
	}
	dispatchRetry(c, "retry", "B")
	dispatchRetry(c, "retry", "B")
	dispatchRetry(c, "retry", "")
	c.loadPendingResults()
	ackRetry(c, "retry", "stale")
	ackRetry(c, "retry", "")
	if len(e.starts) != 0 || len(c.deferredAssignments) != 1 {
		t.Fatal("retry ran before predecessor ACK or duplicates replaced deferred work")
	}
	ackRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "B")
	dispatchRetry(c, "retry", "A")
	dispatchRetry(c, "retry", "B")
	e.finish("retry")
	result := waitRetryResult(t, c, "retry", "B")
	if !result.Success || len(e.starts) != 0 || len(c.deferredAssignments) != 0 {
		t.Fatal("retry did not complete exactly once")
	}
	ackRetry(c, "retry", "A")
	waitRetryResult(t, c, "retry", "B")
	if !e.IsTracked("retry") {
		t.Fatal("stale predecessor ACK forgot retry tracking")
	}
	ackRetry(c, "retry", "B")
	if e.IsTracked("retry") {
		t.Fatal("matching retry ACK did not release tracking")
	}
}

func TestDeferredAssignmentAckAdmissionRace(t *testing.T) {
	for i := 0; i < 20; i++ {
		c, e := newRetryClient(t)
		dispatchRetry(c, "retry", "A")
		expectRetryStart(t, e, "retry", "A")
		e.finish("retry")
		waitRetryResult(t, c, "retry", "A")
		var wg sync.WaitGroup
		for _, action := range []func(){
			func() { dispatchRetry(c, "retry", "B") },
			func() { ackRetry(c, "retry", "A") },
		} {
			wg.Add(1)
			go func(f func()) { defer wg.Done(); f() }(action)
		}
		wg.Wait()
		expectRetryStart(t, e, "retry", "B")
		e.finish("retry")
		waitRetryResult(t, c, "retry", "B")
		if len(e.starts) != 0 {
			t.Fatal("racing ACK caused duplicate execution")
		}
	}
}

func TestDeferredAssignmentDrainsAfterAnotherJobReleasesSlot(t *testing.T) {
	c, e := newRetryClient(t)
	dispatchRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "A")
	e.finish("retry")
	waitRetryResult(t, c, "retry", "A")
	dispatchRetry(c, "other", "other-attempt")
	expectRetryStart(t, e, "other", "other-attempt")
	dispatchRetry(c, "retry", "B")
	ackRetry(c, "retry", "A")
	if len(e.starts) != 0 || len(c.deferredAssignments) != 1 {
		t.Fatal("retry ignored the occupied GPU slot")
	}
	e.finish("other")
	expectRetryStart(t, e, "retry", "B")
	e.finish("retry")
	if !waitRetryResult(t, c, "retry", "B").Success {
		t.Fatal("slot-release drain incorrectly produced a busy rejection")
	}
}

func TestDeferredAssignmentLegacyAndSameTokenRemainDuplicates(t *testing.T) {
	for _, tokens := range [][2]string{{"A", "A"}, {"A", ""}, {"", "B"}, {"", ""}} {
		c := resultTestClient(t)
		c.queueResult(types.JobResult{Type: "job_result", JobID: "legacy", DispatchToken: tokens[0], Success: true})
		original := c.results["legacy"]
		dispatchRetry(c, "legacy", tokens[1])
		if len(c.deferredAssignments) != 0 || c.results["legacy"] != original || len(c.results) != 1 {
			t.Fatalf("legacy/same-token replay was admitted: %v", tokens)
		}
	}
}

func TestDeferredAssignmentInvalidRetryDoesNotReplacePriorResult(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "retry", DispatchToken: "A", Success: true})
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"retry","dispatch_token":"B"}`))
	rejected := c.results[attemptResultKey("retry", "B")]
	if len(c.deferredAssignments) != 0 || rejected == nil || !rejected.durable ||
		!strings.Contains(rejected.result.Error, "model_name") || !c.results["retry"].result.Success {
		t.Fatal("invalid retry replaced its predecessor or vanished")
	}
}

func TestDeferredAssignmentUnreleasedTrackingReturnsFencedFailure(t *testing.T) {
	c := resultTestClient(t)
	c.baseCtx = context.Background()
	c.jobSlot = make(chan struct{}, 1)
	c.executor = &cancellationExecutor{
		tracked: func(string) bool { return true },
		start: func(context.Context, types.JobAssignment) (<-chan types.JobResult, error) {
			return nil, job.ErrAlreadyTracked
		},
	}
	c.queueResult(types.JobResult{Type: "job_result", JobID: "workspace", DispatchToken: "A", Success: true})
	dispatchRetry(c, "workspace", "B")
	ackRetry(c, "workspace", "A")
	result := waitRetryResult(t, c, "workspace", "B")
	if result.Success || !strings.Contains(result.Error, job.ErrAlreadyTracked.Error()) ||
		len(c.jobSlot) != 0 || len(c.deferredAssignments) != 0 {
		t.Fatalf("unreleased tracking silently dropped or started the retry: %+v", result)
	}
}

func TestDeferredAssignmentBoundedOverflowHasIndependentDurableRejection(t *testing.T) {
	c := resultTestClient(t)
	for i := 0; i <= maxDeferredAssignments; i++ {
		id := fmt.Sprintf("retry-%d", i)
		c.queueResult(types.JobResult{Type: "job_result", JobID: id, DispatchToken: "A", Success: true})
		dispatchRetry(c, id, "B")
	}
	if len(c.deferredAssignments) != maxDeferredAssignments {
		t.Fatalf("deferred queue size = %d", len(c.deferredAssignments))
	}
	id := fmt.Sprintf("retry-%d", maxDeferredAssignments)
	key := attemptResultKey(id, "B")
	if rejection := c.results[key]; rejection == nil || !rejection.durable ||
		!strings.Contains(rejection.result.Error, "queue is full") || rejection.result.DispatchToken != "B" {
		t.Fatalf("overflow silently dropped assignment: %+v", rejection)
	}

	dispatchRetry(c, "retry-0", "C")
	if c.deferredAssignments["retry-0"].DispatchToken != "B" ||
		c.results[attemptResultKey("retry-0", "C")] == nil {
		t.Fatal("conflicting deferred token replaced work instead of being rejected")
	}
	restarted := &Client{outboxDir: c.outboxDir}
	restarted.loadPendingResults()
	forgotten := 0
	restarted.executor = &cancellationExecutor{forget: func(string) { forgotten++ }}
	seen := map[string]bool{}
	restarted.sendPendingResults(context.Background(), time.Now(), func(msg interface{}) bool {
		result := msg.(types.JobResult)
		if result.JobID == id {
			seen[result.DispatchToken] = true
		}
		return true
	})
	restarted.sendPendingResults(context.Background(), time.Now(), func(msg interface{}) bool {
		result := msg.(types.JobResult)
		if result.JobID == id {
			seen[result.DispatchToken] = true
		}
		return true
	})
	if !seen["A"] || !seen["B"] {
		t.Fatalf("outbox recovery lost predecessor or overflow rejection: %v", seen)
	}
	ackRetry(restarted, id, "B")
	if restarted.results[id] == nil || restarted.results[key] != nil || forgotten != 0 {
		t.Fatal("rejection ACK removed the predecessor result")
	}
}

func TestDeferredAssignmentCancellationPreventsExecutionDuringAckRace(t *testing.T) {
	c, e := newRetryClient(t)
	dispatchRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "A")
	e.finish("retry")
	waitRetryResult(t, c, "retry", "A")
	dispatchRetry(c, "retry", "B")
	cancelStarted, release := make(chan struct{}), make(chan struct{})
	c.executor = &cancellationExecutor{
		Executor: &job.Executor{}, tracked: e.IsTracked, forget: e.ForgetExecution, start: e.Start,
		cancel: func(string) error {
			close(cancelStarted)
			<-release
			return nil
		},
	}
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_cancel","job_id":"retry"}`))
	select {
	case <-cancelStarted:
	case <-time.After(time.Second):
		t.Fatal("cancellation verification did not start")
	}
	dispatchRetry(c, "retry", "C")
	ackRetry(c, "retry", "A")
	dispatchRetry(c, "retry", "B")
	if len(e.starts) != 0 {
		t.Fatal("cancelled deferred attempt started while ACK raced teardown")
	}
	close(release)
	select {
	case msg := <-c.outbox:
		if ack, ok := msg.(types.JobCancelAck); !ok || !ack.Success {
			t.Fatalf("unexpected cancellation result: %+v", msg)
		}
	case <-time.After(time.Second):
		t.Fatal("missing verified cancellation ACK")
	}
	c.assignmentsMu.Lock()
	defer c.assignmentsMu.Unlock()
	if len(c.deferredAssignments) != 0 || len(e.starts) != 0 {
		t.Fatal("cancellation left runnable deferred work")
	}
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	pending := c.results[attemptResultKey("retry", "B")]
	if pending == nil || !pending.durable || !strings.Contains(pending.result.Error, "cancelled") {
		t.Fatal("deferred cancellation lost its correlated durable result")
	}
}

func TestDeferredAssignmentCancellationWithoutTrackedPredecessor(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "retry", DispatchToken: "A", Success: true})
	dispatchRetry(c, "retry", "B")
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_cancel","job_id":"retry"}`))
	select {
	case msg := <-c.outbox:
		if ack, ok := msg.(types.JobCancelAck); !ok || !ack.Success {
			t.Fatalf("never-started retry was not safely cancelled: %+v", msg)
		}
	case <-time.After(time.Second):
		t.Fatal("missing deferred-only cancellation ACK")
	}
	ackRetry(c, "retry", "A")
	if len(c.deferredAssignments) != 0 {
		t.Fatal("cancelled deferred-only retry remained runnable")
	}
}
