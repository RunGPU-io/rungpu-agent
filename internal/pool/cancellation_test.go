package pool

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type cancellationExecutor struct {
	*job.Executor
	start         func(context.Context, types.JobAssignment) (<-chan types.JobResult, error)
	cancel        func(string) error
	tracked       func(string) bool
	forget        func(string)
	trackedToken  func(string) (string, bool)
	cancelAttempt func(string, string) error
}

func (e *cancellationExecutor) Start(ctx context.Context, a types.JobAssignment) (<-chan types.JobResult, error) {
	return e.start(ctx, a)
}
func (e *cancellationExecutor) Cancel(id string) error { return e.cancel(id) }
func (e *cancellationExecutor) IsTracked(id string) bool {
	return e.tracked != nil && e.tracked(id)
}
func (e *cancellationExecutor) TrackedDispatchToken(id string) (string, bool) {
	if e.trackedToken != nil {
		return e.trackedToken(id)
	}
	return "", e.IsTracked(id)
}
func (e *cancellationExecutor) CancelAttempt(id, token string) error {
	if e.cancelAttempt != nil {
		return e.cancelAttempt(id, token)
	}
	return fmt.Errorf("attempt %s/%s is not tracked by this fixture", id, token)
}
func (*cancellationExecutor) Runtime() string { return "fake" }
func (e *cancellationExecutor) ForgetExecution(id string) {
	if e.forget != nil {
		e.forget(id)
	}
}

func TestCancellationAckWaitsForVerifiedStopAndPreservesResult(t *testing.T) {
	for _, confirmed := range []bool{true, false} {
		name := "negative"
		if confirmed {
			name = "positive"
		}
		t.Run(name, func(t *testing.T) {
			c := resultTestClient(t)
			c.baseCtx = context.Background()
			c.jobSlot = make(chan struct{}, 1)
			results := make(chan types.JobResult, 1)
			cancelled, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			registered := false
			c.executor = &cancellationExecutor{
				start: func(context.Context, types.JobAssignment) (<-chan types.JobResult, error) {
					registered = true
					return results, nil
				},
				cancel: func(id string) error {
					if !registered || id != "cancel-job" {
						t.Error("cancellation raced ahead of synchronous assignment registration")
					}
					close(cancelled)
					<-release
					if !confirmed {
						return errors.New("execution stop could not be verified")
					}
					return nil
				},
			}
			c.dispatch(context.Background(), nil, []byte(
				`{"type":"job_assignment","job_id":"cancel-job","model_name":"test","runtime":"docker-custom"}`))
			if !registered {
				t.Fatal("assignment returned before executor tracking")
			}
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_cancel","job_id":"cancel-job"}`))
			select {
			case <-cancelled:
			case <-time.After(time.Second):
				t.Fatal("cancel was not dispatched")
			}

			results <- types.JobResult{Type: "job_result", JobID: "cancel-job", Success: true}
			deadline := time.Now().Add(time.Second)
			for {
				if _, err := os.Stat(c.resultPath("cancel-job")); err == nil {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("racing successful result was not persisted")
				}
				time.Sleep(time.Millisecond)
			}
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_result_ack","job_id":"cancel-job","success":false}`))
			if _, err := os.Stat(c.resultPath("cancel-job")); err != nil {
				t.Fatalf("pending cancellation lost result on NACK: %v", err)
			}
			select {
			case msg := <-c.outbox:
				t.Fatalf("ACK sent before verified stop returned: %+v", msg)
			case <-time.After(30 * time.Millisecond):
			}
			unblock()
			select {
			case msg := <-c.outbox:
				ack, ok := msg.(types.JobCancelAck)
				if !ok || ack.Type != "job_cancel_ack" || ack.JobID != "cancel-job" || ack.Success != confirmed {
					t.Fatalf("incorrect cancellation ACK: %+v", msg)
				}
				if !confirmed && ack.Error == "" {
					t.Fatal("negative ACK omitted teardown error")
				}
			case <-time.After(time.Second):
				t.Fatal("missing cancellation ACK")
			}
			if _, err := os.Stat(c.resultPath("cancel-job")); err != nil {
				t.Fatalf("cancellation ACK incorrectly discarded result: %v", err)
			}
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_result_ack","job_id":"cancel-job","success":true}`))
		})
	}
}

func TestCancellationAckWaitsForConnectionRegistration(t *testing.T) {
	c := registrationClient(t)
	c.executor = &cancellationExecutor{cancel: func(string) error { return nil }}
	peer := connectRegistrationPeer(t, c, 5*time.Second)
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_cancel","job_id":"stopped"}`))
	peer.quiet(t)
	peer.send(t, `{"type":"gpu_register_ack","success":true}`)
	select {
	case frame := <-peer.frames:
		if frame.Type != "job_cancel_ack" || frame.JobID != "stopped" || !frame.Success {
			t.Fatalf("incorrect registered cancellation ACK: %+v", frame)
		}
	case <-time.After(time.Second):
		t.Fatal("cancellation ACK did not resume after registration")
	}
}

func TestCancellationCoalescesPendingRequestsAndRetriesFailure(t *testing.T) {
	c := resultTestClient(t)
	var calls atomic.Int32
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	c.executor = &cancellationExecutor{cancel: func(string) error {
		n := calls.Add(1)
		select {
		case started <- struct{}{}:
		default:
		}
		<-release
		if n == 1 {
			return context.DeadlineExceeded
		}
		return nil
	}}
	request := func(kind string) {
		c.dispatch(context.Background(), nil, []byte(fmt.Sprintf(
			`{"type":%q,"job_id":"coalesced"}`, kind)))
	}
	request("job_cancel")
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first cancellation did not start")
	}
	for i := 0; i < 1000; i++ {
		request([]string{"job_cancel", "job_stop", "stop_job"}[i%3])
	}
	select {
	case <-started:
		t.Fatal("duplicate cancellation started another teardown")
	case <-time.After(30 * time.Millisecond):
	}
	if calls.Load() != 1 {
		t.Fatalf("pending cancellation spawned %d verification calls", calls.Load())
	}
	unblock()
	select {
	case msg := <-c.outbox:
		ack, ok := msg.(types.JobCancelAck)
		if !ok || ack.Success || ack.Error == "" {
			t.Fatalf("timeout did not produce negative ACK: %+v", msg)
		}
	case <-time.After(time.Second):
		t.Fatal("missing negative ACK")
	}
	deadline := time.Now().Add(time.Second)
	for {
		if _, pending := c.cancelling.Load("coalesced"); !pending {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("completed verification blocked future retries")
		}
		time.Sleep(time.Millisecond)
	}
	request("job_cancel")
	select {
	case msg := <-c.outbox:
		ack, ok := msg.(types.JobCancelAck)
		if !ok || !ack.Success || ack.JobID != "coalesced" {
			t.Fatalf("subsequent verified retry did not produce positive ACK: %+v", msg)
		}
	case <-time.After(time.Second):
		t.Fatal("missing retry ACK")
	}
	if calls.Load() != 2 {
		t.Fatalf("expected one verification per attempt, got %d", calls.Load())
	}
}
