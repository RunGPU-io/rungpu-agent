package job

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type cancellationRuntime struct {
	prepare func(context.Context) error
	run     func(context.Context) (map[string]interface{}, error)
}

func (*cancellationRuntime) Name() string { return "fake" }
func (r *cancellationRuntime) Prepare(ctx context.Context, _ types.JobAssignment) error {
	if r.prepare != nil {
		return r.prepare(ctx)
	}
	return nil
}
func (r *cancellationRuntime) Run(ctx context.Context, _ types.JobAssignment) (map[string]interface{}, error) {
	return r.run(ctx)
}
func (*cancellationRuntime) Cleanup(bool) error { return nil }

type cancellationTeardown func(context.Context, string) error

func (f cancellationTeardown) RemoveAndVerify(ctx context.Context, name string) error {
	return f(ctx, name)
}

func TestVerifiedCancellationJoinsExecutionBeforeSuccess(t *testing.T) {
	started, cancelled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(ctx context.Context) (map[string]interface{}, error) {
			close(started)
			<-ctx.Done()
			close(cancelled)
			<-release
			return nil, ctx.Err()
		},
	}}
	var mu sync.Mutex
	verifications := 0
	e.teardown = cancellationTeardown(func(context.Context, string) error {
		mu.Lock()
		defer mu.Unlock()
		verifications++
		return nil
	})
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "join", Runtime: "docker-custom"})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	done := make(chan error, 1)
	go func() { done <- e.Cancel("join") }()
	<-cancelled
	select {
	case err := <-done:
		t.Fatalf("cancel returned before execution joined: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	unblock()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	<-results
	mu.Lock()
	if verifications != 4 {
		t.Errorf("expected both container names checked before and after join; got %d", verifications)
	}
	mu.Unlock()
	if err := e.Cancel("join"); err != nil {
		t.Fatalf("confirmed cancellation must be idempotent: %v", err)
	}
}

func TestVerifiedCancellationDispatchTokenMustMatchBeforeAnyStop(t *testing.T) {
	started, stopped := make(chan struct{}), make(chan struct{})
	var mu sync.Mutex
	verifications := 0
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(ctx context.Context) (map[string]interface{}, error) {
			close(started)
			<-ctx.Done()
			close(stopped)
			return nil, ctx.Err()
		},
	}, teardown: cancellationTeardown(func(context.Context, string) error {
		mu.Lock()
		verifications++
		mu.Unlock()
		return nil
	})}
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "attempt", DispatchToken: "A", Runtime: "docker-custom"})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Cancel("attempt")
	<-started
	if token, tracked := e.TrackedDispatchToken("attempt"); !tracked || token != "A" {
		t.Fatalf("wrong tracked attempt: token=%q tracked=%t", token, tracked)
	}
	if err := e.CancelAttempt("attempt", "B"); err == nil || !strings.Contains(err.Error(), "different dispatch token") {
		t.Fatalf("wrong token was accepted as cancellation proof: %v", err)
	}
	select {
	case <-stopped:
		t.Fatal("wrong-token cancellation stopped A")
	default:
	}
	mu.Lock()
	if verifications != 0 {
		t.Error("wrong-token cancellation touched containers")
	}
	mu.Unlock()
	if err := e.CancelAttempt("attempt", "A"); err != nil {
		t.Fatal(err)
	}
	<-results
	if err := e.CancelAttempt("attempt", "B"); err == nil {
		t.Fatal("already-confirmed A stop was incorrectly reused as proof for B")
	}
	e.ForgetExecution("attempt")
	if err := e.CancelAttempt("attempt", "A"); err == nil {
		t.Fatal("untracked execution falsely confirmed stopped")
	}
}

func TestVerifiedCancellationLegacyTrackingDoesNotMatchToken(t *testing.T) {
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(ctx context.Context) (map[string]interface{}, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}, teardown: cancellationTeardown(func(context.Context, string) error { return nil })}
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "legacy", Runtime: "docker-custom"})
	if err != nil {
		t.Fatal(err)
	}
	if err := e.CancelAttempt("legacy", "token"); err == nil {
		t.Fatal("token-bearing request matched a tokenless execution")
	}
	if err := e.Cancel("legacy"); err != nil {
		t.Fatal(err)
	}
	<-results
}

func TestVerifiedCancellationTracksBeforeExecutionStarts(t *testing.T) {
	for i := 0; i < 100; i++ {
		e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
			run: func(ctx context.Context) (map[string]interface{}, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			},
		}, teardown: cancellationTeardown(func(context.Context, string) error { return nil })}
		results, err := e.Start(context.Background(), types.JobAssignment{JobID: "early", Runtime: "docker-custom"})
		if err != nil {
			t.Fatal(err)
		}
		if err := e.Cancel("early"); err != nil {
			t.Fatalf("cancel immediately after Start: %v", err)
		}
		if result := <-results; result.Success {
			t.Fatal("immediate cancellation unexpectedly succeeded")
		}
	}
}

func TestVerifiedCancellationWorkspaceAndTeardownFailure(t *testing.T) {
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(context.Context) (map[string]interface{}, error) {
			return map[string]interface{}{"status": "running"}, nil
		},
	}, teardown: cancellationTeardown(func(context.Context, string) error {
		return errors.New("daemon unavailable")
	})}
	result := e.Execute(context.Background(), types.JobAssignment{JobID: "workspace", Runtime: "workspace"})
	if !result.Success || len(e.WorkspaceIDs()) != 1 {
		t.Fatal("workspace was not retained after startup")
	}
	e.ForgetExecution("workspace")
	if err := e.Cancel("workspace"); err == nil || !strings.Contains(err.Error(), "daemon unavailable") {
		t.Fatalf("teardown failure not propagated: %v", err)
	}
	if len(e.WorkspaceIDs()) != 1 {
		t.Fatal("failed teardown discarded workspace tracking")
	}
	if err := e.StopAll(context.Background()); err == nil {
		t.Fatal("shutdown silently ignored unverified workspace teardown")
	}
	e.teardown = cancellationTeardown(func(context.Context, string) error { return nil })
	if err := e.Cancel("workspace"); err != nil {
		t.Fatal(err)
	}
	if len(e.WorkspaceIDs()) != 0 {
		t.Fatal("verified workspace teardown did not clear tracking")
	}
}

func TestVerifiedCancellationTimeoutDoesNotClaimStop(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(context.Context) (map[string]interface{}, error) {
			close(started)
			<-release
			return nil, nil
		},
	}, teardown: cancellationTeardown(func(context.Context, string) error { return nil })}
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "slow", Runtime: "docker-custom"})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := e.CancelContext(ctx, "slow"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected bounded join failure, got %v", err)
	}
	unblock()
	<-results
	e.ForgetExecution("slow")
	if err := e.Cancel("slow"); err != nil {
		t.Fatalf("unconfirmed execution was forgotten rather than retried: %v", err)
	}
}

func TestVerifiedCancellationRejectsNativeBackendProof(t *testing.T) {
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(ctx context.Context) (map[string]interface{}, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}, teardown: cancellationTeardown(func(context.Context, string) error {
		t.Error("native runtime must not infer proof from Docker teardown")
		return nil
	})}
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "native", Runtime: "ollama"})
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Cancel("native"); err == nil || !strings.Contains(err.Error(), "cannot confirm backend") {
		t.Fatalf("native cancellation must be negative: %v", err)
	}
	<-results
}

func TestVerifiedCancellationRechecksAfterJoin(t *testing.T) {
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(ctx context.Context) (map[string]interface{}, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}}
	checks := 0
	e.teardown = cancellationTeardown(func(context.Context, string) error {
		checks++
		if checks > 2 {
			return errors.New("container created during cancellation remains")
		}
		return nil
	})
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "late", Runtime: "workspace"})
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Cancel("late"); err == nil || !strings.Contains(err.Error(), "container created") {
		t.Fatalf("missed post-join teardown failure: %v", err)
	}
	<-results
}

func TestVerifiedCancellationConcurrentRequests(t *testing.T) {
	e := &Executor{cacheDir: t.TempDir(), runtime: &cancellationRuntime{
		run: func(ctx context.Context) (map[string]interface{}, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}, teardown: cancellationTeardown(func(context.Context, string) error { return nil })}
	results, err := e.Start(context.Background(), types.JobAssignment{JobID: "concurrent", Runtime: "docker-custom"})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 10)
	for i := 0; i < cap(done); i++ {
		go func() { done <- e.Cancel("concurrent") }()
	}
	for i := 0; i < cap(done); i++ {
		if err := <-done; err != nil {
			t.Fatalf("concurrent cancellation: %v", err)
		}
	}
	<-results
}
