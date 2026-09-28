package pool

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestAdmissionAcrossRealGPUClients(t *testing.T) {

	t.Setenv("PATH", t.TempDir())
	for _, pair := range [][2]string{
		{"ollama", "ollama"}, {"ollama", "docker-custom"},
		{"docker-custom", "ollama"}, {"workspace", "ollama"},
		{"ollama", "workspace"}, {"docker-custom", "docker-custom"},
	} {
		t.Run(pair[0]+"/"+pair[1], func(t *testing.T) {
			cfg := &types.Config{MachineID: "admission-machine", ModelCacheDir: t.TempDir()}
			first, err := NewClientForGPU(cfg, 0)
			if err != nil {
				t.Fatal(err)
			}
			sibling, err := NewClientForGPU(cfg, 1)
			if err != nil {
				t.Fatal(err)
			}
			if first.maintenanceGate != sibling.maintenanceGate {
				t.Fatal("sibling client constructors did not share machine admission")
			}
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			started := []chan struct{}{make(chan struct{}, 1), make(chan struct{}, 1)}
			clients := []*Client{first, sibling}
			for i, c := range clients {
				c.baseCtx = context.Background()
				ready := started[i]
				c.executor.(*job.Executor).OnProgress = func(types.JobProgress) {
					ready <- struct{}{}
					<-release
				}
			}
			dispatch := func(c *Client, id, runtime string) {
				data, err := json.Marshal(types.JobAssignment{Type: "job_assignment", JobID: id, ModelName: "test", Runtime: runtime})
				if err != nil {
					t.Fatal(err)
				}
				c.dispatch(context.Background(), nil, data)
			}
			dispatch(first, "first", pair[0])
			select {
			case <-started[0]:
			case <-time.After(time.Second):
				t.Fatal("first client did not start")
			}
			dispatch(sibling, "sibling", pair[1])
			conflicts := pair[0] == "ollama" || pair[1] == "ollama"
			if conflicts {
				sibling.resultsMu.Lock()
				pending := sibling.results["sibling"]
				sibling.resultsMu.Unlock()
				if pending == nil || !strings.Contains(pending.result.Error, "machine is busy") {
					t.Fatalf("sibling native exclusion failed: %+v", pending)
				}
			} else {
				select {
				case <-started[1]:
				case <-time.After(time.Second):
					t.Fatal("Docker jobs did not start concurrently on sibling clients")
				}
			}
			unblock()
			deadline := time.Now().Add(time.Second)
			for len(first.jobSlot) != 0 || len(sibling.jobSlot) != 0 {
				if time.Now().After(deadline) {
					t.Fatal("completed preparation failure leaked a job slot")
				}
				time.Sleep(time.Millisecond)
			}
			if !first.maintenanceGate.TryLock() {
				t.Fatal("completed preparation failures leaked a machine lease")
			}
			first.maintenanceGate.Unlock()
		})
	}
}

func TestAdmissionStartFailureReportsErrorAndReleasesSlot(t *testing.T) {
	c := resultTestClient(t)
	c.baseCtx = context.Background()
	c.jobSlot = make(chan struct{}, 1)
	for i := 0; i < cap(c.outbox); i++ {
		c.outbox <- types.JobProgress{}
	}
	c.executor = &cancellationExecutor{start: func(context.Context, types.JobAssignment) (<-chan types.JobResult, error) {
		return nil, errors.New("machine is busy with native Ollama")
	}}
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"busy","runtime":"docker-custom","model_name":"test"}`))
	if len(c.jobSlot) != 0 {
		t.Fatal("rejected assignment leaked GPU slot")
	}
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	pending := c.results["busy"]
	if pending == nil || !pending.durable || pending.result.Success || !strings.Contains(pending.result.Error, "machine is busy") {
		t.Fatalf("missing durable admission failure with full best-effort outbox: %+v", pending)
	}
}

func TestAdmissionDuplicateDoesNotOverwriteExistingResult(t *testing.T) {
	c := resultTestClient(t)
	c.baseCtx = context.Background()
	c.jobSlot = make(chan struct{}, 1)
	c.executor = &cancellationExecutor{start: func(context.Context, types.JobAssignment) (<-chan types.JobResult, error) {
		return nil, job.ErrAlreadyTracked
	}}
	c.queueResult(types.JobResult{Type: "job_result", JobID: "existing", Success: true})
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"existing","runtime":"ollama","model_name":"test"}`))
	if len(c.outbox) != 0 || len(c.jobSlot) != 0 {
		t.Fatal("duplicate assignment emitted a replacement failure or leaked its slot")
	}
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	if !c.results["existing"].result.Success {
		t.Fatal("duplicate assignment overwrote successful pending result")
	}
}

func TestAdmissionValidationRejectionsAreDurable(t *testing.T) {
	for _, tc := range []struct {
		name, fields, wantError string
	}{
		{"model", `"runtime":"ollama"`, "model_name is required"},
		{"protocol", `"model_name":"test","protocol_version":99`, "unsupported job protocol"},
		{"runtime", `"model_name":"test","protocol_version":1`, "runtime is required"},
		{"checksum", `"model_name":"test","runtime":"docker-custom","protocol_version":2,"custom_files":[{"path":"file"}]`, "custom file sha256"},
		{"workflow", `"model_name":"test","runtime":"comfyui-batch","protocol_version":2,"workflow_json":"{"`, "workflow_json"},
		{"decode", `"model_name":42`, "invalid job_assignment"},
		{"token type", `"model_name":"test","dispatch_token":42`, "invalid job_assignment"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := resultTestClient(t)
			c.gpuID = "gpu-contract"
			c.jobSlot = make(chan struct{}, 1)
			for i := 0; i < cap(c.outbox); i++ {
				c.enqueue(types.JobProgress{})
			}
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"invalid",`+tc.fields+`}`))
			c.resultsMu.Lock()
			pending := c.results["invalid"]
			c.resultsMu.Unlock()
			if pending == nil || !pending.durable || pending.result.Success ||
				pending.result.GPUID != c.gpuID || !strings.Contains(pending.result.Error, tc.wantError) {
				t.Fatalf("missing durable correlated rejection: %+v", pending)
			}
			if len(c.jobSlot) != 0 {
				t.Fatal("invalid assignment reserved execution")
			}
			restarted := &Client{outboxDir: c.outboxDir}
			restarted.loadPendingResults()
			sends := 0
			now := time.Now()
			write := func(msg interface{}) bool {
				result := msg.(types.JobResult)
				if result.JobID != "invalid" || result.Error != pending.result.Error {
					t.Fatalf("rejection lost on restart: %+v", result)
				}
				sends++
				return true
			}
			restarted.sendPendingResults(context.Background(), now, write)
			restarted.dispatch(context.Background(), nil, []byte(`{"type":"job_result_ack","job_id":"invalid","success":false}`))
			restarted.sendPendingResults(context.Background(), now.Add(resultRetryMin), write)
			acceptResult(restarted, "invalid")
			restarted.sendPendingResults(context.Background(), now.Add(time.Hour), write)
			if sends != 2 {
				t.Fatalf("rejection retries = %d, want 2 before positive ACK", sends)
			}
		})
	}
}

func TestAdmissionRejectionsEchoDispatchToken(t *testing.T) {
	for _, reason := range []string{"validation", "decode", "busy", "start"} {
		t.Run(reason, func(t *testing.T) {
			c := resultTestClient(t)
			c.baseCtx = context.Background()
			c.jobSlot = make(chan struct{}, 1)
			fields := `"model_name":"test","runtime":"ollama"`
			switch reason {
			case "validation":
				fields = `"model_name":""`
			case "decode":
				fields = `"model_name":42`
			case "busy":
				c.jobSlot <- struct{}{}
			case "start":
				c.executor = &cancellationExecutor{start: func(context.Context, types.JobAssignment) (<-chan types.JobResult, error) {
					return nil, errors.New("start rejected")
				}}
			}
			c.dispatch(context.Background(), nil, []byte(
				`{"type":"job_assignment","job_id":"rejection","dispatch_token":"attempt-rejected",`+fields+`}`))
			pending := c.results["rejection"]
			if pending == nil || !pending.durable || pending.result.Success || pending.result.Error == "" ||
				pending.result.DispatchToken != "attempt-rejected" {
				t.Fatalf("rejection lost attempt correlation: %+v", pending)
			}
		})
	}
}

func TestAdmissionDuplicateActiveAndBusyRejection(t *testing.T) {
	c := resultTestClient(t)
	c.baseCtx = context.Background()
	c.jobSlot = make(chan struct{}, 1)
	results := make(chan types.JobResult, 1)
	var starts atomic.Int32
	c.executor = &cancellationExecutor{start: func(_ context.Context, a types.JobAssignment) (<-chan types.JobResult, error) {
		starts.Add(1)
		if a.DispatchToken != "attempt-original" {
			t.Errorf("assignment lost dispatch token: %+v", a)
		}
		return results, nil
	}}
	assignment := []byte(`{"type":"job_assignment","job_id":"active","model_name":"test","runtime":"ollama","dispatch_token":"attempt-original"}`)
	c.dispatch(context.Background(), nil, assignment)
	for i := 0; i < cap(c.outbox); i++ {
		c.enqueue(types.JobProgress{})
	}
	var duplicates sync.WaitGroup
	for i := 0; i < 30; i++ {
		duplicates.Add(1)
		go func() {
			defer duplicates.Done()
			c.dispatch(context.Background(), nil, assignment)
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"active"}`))
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"active","model_name":42}`))
			c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"active","dispatch_token":"attempt-old","model_name":"test","runtime":"ollama"}`))
		}()
	}
	duplicates.Wait()
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"busy-other","model_name":"test","runtime":"ollama"}`))
	c.resultsMu.Lock()
	active, busy := c.results["active"], c.results["busy-other"]
	c.resultsMu.Unlock()
	if active != nil || starts.Load() != 1 {
		t.Fatalf("duplicate failed or restarted original execution: pending=%+v starts=%d", active, starts.Load())
	}
	if busy == nil || !busy.durable || busy.result.Success || !strings.Contains(busy.result.Error, "already running") {
		t.Fatalf("full best-effort outbox dropped busy rejection: %+v", busy)
	}
	results <- types.JobResult{Type: "job_result", JobID: "active", Success: true}
	deadline := time.Now().Add(time.Second)
	for {
		c.assignmentsMu.Lock()
		finished := c.activeAssignment == ""
		c.assignmentsMu.Unlock()
		if finished {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("original execution did not complete")
		}
		time.Sleep(time.Millisecond)
	}
	c.dispatch(context.Background(), nil, assignment)
	c.dispatch(context.Background(), nil, []byte(`{"type":"job_assignment","job_id":"active"}`))
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	if starts.Load() != 1 || !c.results["active"].result.Success ||
		c.results["active"].result.DispatchToken != "attempt-original" || len(c.jobSlot) != 0 {
		t.Fatal("pending result duplicate overwrote or restarted original execution")
	}
}

func TestAdmissionPendingResultSuppressesReplayAfterRestart(t *testing.T) {
	c := resultTestClient(t)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "pending", Success: true, Result: map[string]interface{}{"text": "original"}})
	restarted := &Client{outboxDir: c.outboxDir, jobSlot: make(chan struct{}, 1)}
	restarted.loadPendingResults()
	original := restarted.results["pending"]

	for _, data := range []string{
		`{"type":"job_assignment","job_id":"pending","model_name":"test","runtime":"ollama"}`,
		`{"type":"job_assignment","job_id":"pending"}`,
		`{"type":"job_assignment","job_id":"pending","model_name":42}`,
	} {
		restarted.dispatch(context.Background(), nil, []byte(data))
		if restarted.results["pending"] != original || original.result.Result["text"] != "original" {
			t.Fatal("replay replaced durable result")
		}
	}
}

func TestAdmissionTrackedWorkspaceReplayDoesNotBecomeBusyFailure(t *testing.T) {
	c := resultTestClient(t)
	c.jobSlot = make(chan struct{}, 1)
	c.jobSlot <- struct{}{}
	c.activeAssignment = "another-job"
	c.executor = &cancellationExecutor{tracked: func(id string) bool { return id == "workspace" }}

	for _, data := range []string{
		`{"type":"job_assignment","job_id":"workspace","model_name":"test","runtime":"workspace"}`,
		`{"type":"job_assignment","job_id":"workspace"}`,
	} {
		c.dispatch(context.Background(), nil, []byte(data))
	}
	if len(c.results) != 0 || len(c.outbox) != 0 || len(c.jobSlot) != 1 {
		t.Fatal("tracked workspace replay failed the original job or disturbed another job")
	}
}

func TestAdmissionUnsafeIdentifiersNeverReachOutboxOrExecution(t *testing.T) {
	c := resultTestClient(t)
	c.jobSlot = make(chan struct{}, 1)
	c.queueResult(types.JobResult{Type: "job_result", JobID: "victim", Success: true})
	for _, id := range []string{"", "../victim", "/victim", `..\victim`, ".", "..", ".hidden", "a/b", "a\nb", strings.Repeat("a", 129)} {
		for _, kind := range []string{"job_assignment", "job_result_ack", "job_cancel"} {
			data, err := json.Marshal(map[string]interface{}{
				"type": kind, "job_id": id, "model_name": "test", "success": true,
			})
			if err != nil {
				t.Fatal(err)
			}
			c.dispatch(context.Background(), nil, data)
		}
	}
	if len(c.results) != 1 || c.results["victim"] == nil || len(c.jobSlot) != 0 || len(c.outbox) != 0 {
		t.Fatal("unsafe identifier was used as an execution/outbox key or aliased another job")
	}
	files, err := os.ReadDir(c.outboxDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 || files[0].Name() != "victim.json" {
		t.Fatalf("unsafe identifier changed durable outbox: %v", files)
	}
}

func TestAdmissionCleanupRejectsActiveMachineWithoutWaiting(t *testing.T) {
	for _, native := range []bool{true, false} {
		c := resultTestClient(t)
		c.baseCtx = context.Background()
		c.maintenanceGate = &sync.RWMutex{}
		if native {
			c.maintenanceGate.Lock()
		} else {
			c.maintenanceGate.RLock()
		}
		_, _, skipped, err := pruneCustomAssetsIfIdle(c.cfg, []*Client{c}, time.Now())
		if err != nil || !skipped {
			t.Fatalf("automatic cleanup did not skip active machine: skipped=%v err=%v", skipped, err)
		}

		c.dispatch(context.Background(), nil, []byte(`{"type":"asset_cleanup","request_id":"cleanup","phase":"execute","categories":["job_files"]}`))
		select {
		case msg := <-c.outbox:
			result, ok := msg.(types.AssetCleanupResult)
			if !ok || result.Success || !strings.Contains(result.Error, "machine jobs are active") {
				t.Fatalf("cleanup did not fail closed: %+v", msg)
			}
		case <-time.After(time.Second):
			t.Fatal("cleanup blocked indefinitely on active machine lease")
		}
		if native {
			c.maintenanceGate.Unlock()
		} else {
			c.maintenanceGate.RUnlock()
		}
	}
}
