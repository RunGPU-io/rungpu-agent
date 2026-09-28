package pool

import (
	"context"
	"fmt"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func newOfflineFixtureClient(t *testing.T, cfg *types.Config, run func(types.JobAssignment) (map[string]interface{}, error)) *Client {
	t.Helper()
	t.Setenv("PATH", t.TempDir())
	c, err := NewClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.executor = &cancellationExecutor{
		Executor: &job.Executor{},
		start: func(ctx context.Context, a types.JobAssignment) (<-chan types.JobResult, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			result := types.JobResult{
				Type: "job_result", JobID: a.JobID, GPUID: c.gpuID,
				DispatchToken: a.DispatchToken, DurationMS: 1,
			}
			for _, stage := range []string{"pulling_image", "running"} {
				c.enqueue(types.JobProgress{Type: "job_progress", JobID: a.JobID, GPUID: c.gpuID,
					DispatchToken: a.DispatchToken, Stage: stage, Progress: 0.5})
			}
			var err error
			if run != nil {
				result.Result, err = run(a)
			} else {
				result.Result = map[string]interface{}{
					"response": fmt.Sprintf("fixture:%v", a.Input["prompt"]),
					"model":    a.ModelName, "runtime": a.Runtime, "backend": "fixture",
				}
			}
			result.Success = err == nil
			if err != nil {
				result.Error = err.Error()
			}
			done := make(chan types.JobResult, 1)
			done <- result
			return done, nil
		},
	}
	return c
}

func assertFixtureResponse(t *testing.T, result map[string]interface{}, prompt string) {
	t.Helper()
	data, ok := result["result"].(map[string]interface{})
	if result["success"] != true || !ok || data["response"] != "fixture:"+prompt || data["backend"] != "fixture" {
		t.Fatalf("fixture execution did not round-trip its prompt: %+v", result)
	}
}
