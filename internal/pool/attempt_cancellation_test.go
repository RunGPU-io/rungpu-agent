package pool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func cancelRetry(c *Client, id, token string) {
	data, _ := json.Marshal(types.JobControl{Type: "job_cancel", JobID: id, DispatchToken: token})
	c.dispatch(context.Background(), nil, data)
}

func readAttemptCancelAck(t *testing.T, c *Client, token string, success bool) types.JobCancelAck {
	t.Helper()
	select {
	case msg := <-c.outbox:
		ack, ok := msg.(types.JobCancelAck)
		if !ok || ack.DispatchToken != token || ack.Success != success || (!success && ack.Error == "") {
			t.Fatalf("unexpected attempt cancellation ACK: %+v", msg)
		}
		return ack
	case <-time.After(time.Second):
		t.Fatal("missing attempt cancellation ACK")
		return types.JobCancelAck{}
	}
}

func TestAttemptCancellationOvertakesDelayedRetryWithoutStoppingPredecessor(t *testing.T) {
	c, e := newRetryClient(t)
	dispatchRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "A")
	e.finish("retry")
	waitRetryResult(t, c, "retry", "A")
	var stopCalls atomic.Int32
	c.executor = &cancellationExecutor{
		start: e.Start, tracked: e.IsTracked, trackedToken: e.TrackedDispatchToken, forget: e.ForgetExecution,
		cancel:        func(string) error { stopCalls.Add(1); return nil },
		cancelAttempt: func(string, string) error { stopCalls.Add(1); return nil },
	}

	cancelRetry(c, "retry", "B")
	readAttemptCancelAck(t, c, "B", false)
	if stopCalls.Load() != 0 || !e.IsTracked("retry") {
		t.Fatal("stop of predecessor A was used as proof for unknown B")
	}
	ackRetry(c, "retry", "A")
	dispatchRetry(c, "retry", "B")
	if len(e.starts) != 0 || len(c.deferredAssignments) != 0 {
		t.Fatal("late cancelled assignment B executed after ACK A")
	}
	if pending := c.results[attemptResultKey("retry", "B")]; pending == nil ||
		pending.result.Success || !strings.Contains(pending.result.Error, "cancelled") {
		t.Fatalf("late B was not durably rejected: %+v", pending)
	}
	ackRetry(c, "retry", "B")
	c.settledAttempts = nil
	dispatchRetry(c, "retry", "B")
	if len(e.starts) != 0 {
		t.Fatal("cancellation tombstone disappeared after result acknowledgement")
	}

	cancelRetry(c, "retry", "B")
	readAttemptCancelAck(t, c, "B", false)
}

func TestAttemptCancellationKnownDeferredRetryNeedsNoPredecessorStop(t *testing.T) {
	c, e := newRetryClient(t)
	dispatchRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "A")
	e.finish("retry")
	waitRetryResult(t, c, "retry", "A")
	dispatchRetry(c, "retry", "B")
	var stopCalls atomic.Int32
	c.executor = &cancellationExecutor{
		start: e.Start, tracked: e.IsTracked, trackedToken: e.TrackedDispatchToken, forget: e.ForgetExecution,
		cancel:        func(string) error { stopCalls.Add(1); return nil },
		cancelAttempt: func(string, string) error { stopCalls.Add(1); return nil },
	}
	cancelRetry(c, "retry", "B")
	readAttemptCancelAck(t, c, "B", true)
	if stopCalls.Load() != 0 || !e.IsTracked("retry") {
		t.Fatal("known-unstarted B incorrectly cancelled predecessor A")
	}
	ackRetry(c, "retry", "A")
	ackRetry(c, "retry", "B")
	dispatchRetry(c, "retry", "B")
	cancelRetry(c, "retry", "B")
	readAttemptCancelAck(t, c, "B", true)
	if len(e.starts) != 0 || len(c.deferredAssignments) != 0 {
		t.Fatal("known-unstarted cancelled retry executed")
	}
}

func TestAttemptCancellationMatchingTokenWaitsForVerification(t *testing.T) {
	for _, verified := range []bool{false, true} {
		t.Run(fmt.Sprintf("verified=%t", verified), func(t *testing.T) {
			c := resultTestClient(t)
			started, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			var calls atomic.Int32
			c.executor = &cancellationExecutor{
				tracked:      func(string) bool { return true },
				trackedToken: func(string) (string, bool) { return "B", true },
				cancelAttempt: func(id, token string) error {
					calls.Add(1)
					if id != "retry" || token != "B" {
						return errors.New("wrong attempt")
					}
					close(started)
					<-release
					if !verified {
						return errors.New("backend stop unverified")
					}
					return nil
				},
			}
			cancelRetry(c, "retry", "B")
			<-started
			cancelRetry(c, "retry", "B")
			cancelRetry(c, "retry", "A")
			readAttemptCancelAck(t, c, "A", false)
			if len(c.outbox) != 0 || calls.Load() != 1 {
				t.Fatal("matching token ACKed before verification or duplicate cancel ran again")
			}
			unblock()
			readAttemptCancelAck(t, c, "B", verified)
			c.assignmentsMu.Lock()
			tombstone := c.cancellationTombstones[attemptResultKey("retry", "B")]
			if tombstone == nil || tombstone.stopVerified != verified {
				t.Fatal("attempt verification state was not retained")
			}
			c.assignmentsMu.Unlock()
		})
	}
}

func TestAttemptCancellationUnknownExecutionNeverClaimsStop(t *testing.T) {
	c := resultTestClient(t)
	cancelRetry(c, "unknown", "old-token")
	readAttemptCancelAck(t, c, "old-token", false)
	dispatchRetry(c, "unknown", "old-token")
	cancelRetry(c, "unknown", "old-token")
	readAttemptCancelAck(t, c, "old-token", false)
	if len(c.deferredAssignments) != 0 || len(c.cancellationTombstones) != 1 {
		t.Fatal("unknown attempt lost its cancellation tombstone")
	}
}

func TestAttemptCancellationCapacityFencesWithoutEvictingUnknownAttempts(t *testing.T) {
	c, e := newRetryClient(t)
	dispatchRetry(c, "retry", "A")
	expectRetryStart(t, e, "retry", "A")
	e.finish("retry")
	waitRetryResult(t, c, "retry", "A")
	dispatchRetry(c, "retry", "B")
	c.cancellationTombstones = make(map[string]*cancellationTombstone)
	for i := 0; i < maxCancellationTombstones; i++ {
		c.cancellationTombstones[attemptResultKey("old", fmt.Sprint(i))] = &cancellationTombstone{}
	}
	cancelRetry(c, "overflow", "overflow-token")
	ack := readAttemptCancelAck(t, c, "overflow-token", false)
	if !strings.Contains(ack.Error, "capacity") || !c.cancellationAdmissionFenced ||
		len(c.cancellationTombstones) != maxCancellationTombstones ||
		c.cancellationTombstones[attemptResultKey("old", "0")] == nil {
		t.Fatal("capacity exhaustion evicted cancellation proof instead of fencing admission")
	}
	ackRetry(c, "retry", "A")
	result := waitRetryResult(t, c, "retry", "B")
	if result.Success || !strings.Contains(result.Error, "fence") {
		t.Fatal("already-deferred assignment bypassed exhausted cancellation fence")
	}
	dispatchRetry(c, "overflow", "overflow-token")
	dispatchRetry(c, "old", "0")
	dispatchRetry(c, "legacy", "")
	if len(e.starts) != 0 || len(c.deferredAssignments) != 0 {
		t.Fatal("capacity fence admitted a late or legacy assignment")
	}
}

func TestAttemptCancellationLegacyAckNeverCarriesTrackedToken(t *testing.T) {
	c := resultTestClient(t)
	var legacyCalls atomic.Int32
	c.executor = &cancellationExecutor{
		tracked:      func(string) bool { return true },
		trackedToken: func(string) (string, bool) { return "B", true },
		cancel:       func(string) error { legacyCalls.Add(1); return nil },
	}
	cancelRetry(c, "retry", "")
	ack := readAttemptCancelAck(t, c, "", true)
	data, err := json.Marshal(ack)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "dispatch_token") || legacyCalls.Load() != 1 {
		t.Fatalf("legacy cancellation fabricated token-bearing proof: %s", data)
	}
}
