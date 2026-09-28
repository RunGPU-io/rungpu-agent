package job

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestParseCoordinatorDrive(t *testing.T) {
	drive := parseCoordinatorDrive(map[string]interface{}{
		"_drive_script":    "print(1)",
		"_entrypoint":      "python",
		"_command":         []interface{}{"-c", "pass"},
		"_output_mount":    "/app/out",
		"_use_host_memory": true,
		"_extra_env":       map[string]interface{}{"HF_HUB_OFFLINE": "1"},
		"_stage_mounts":    []interface{}{map[string]interface{}{"from": "models", "to": "/models", "ro": true}},
		"timeout_minutes":  "10",
	})
	if drive.Script != "print(1)" || drive.Entrypoint != "python" || drive.OutputMount != "/app/out" || !drive.UseHostMemory {
		t.Fatalf("drive = %+v", drive)
	}
	if len(drive.Command) != 2 || drive.Command[1] != "pass" {
		t.Fatalf("command = %v", drive.Command)
	}
	if drive.ExtraEnv["HF_HUB_OFFLINE"] != "1" {
		t.Fatalf("extra env = %v", drive.ExtraEnv)
	}
	if len(drive.StageMounts) != 1 || drive.StageMounts[0].From != "models" {
		t.Fatalf("stage mounts = %+v", drive.StageMounts)
	}
}

func TestParseCoordinatorDriveRejectsTraversalMount(t *testing.T) {
	drive := parseCoordinatorDrive(map[string]interface{}{
		"_stage_mounts": []interface{}{
			map[string]interface{}{"from": "../etc", "to": "/etc"},
			map[string]interface{}{"from": "/root", "to": "/root"},
			map[string]interface{}{"from": `C:\Windows`, "to": "/windows"},
			map[string]interface{}{"from": "C:/Windows", "to": "/windows"},
		},
	})
	if len(drive.StageMounts) != 0 {
		t.Fatalf("host and traversal mounts should be dropped: %+v", drive.StageMounts)
	}
}

func TestCoordinatorParamsAreNotCopiedToEnv(t *testing.T) {
	if !isCoordinatorParam("_drive_script") || isCoordinatorParam("timeout_minutes") {
		t.Fatal("coordinator param detection")
	}
}

func TestBuildContainerEnvPreservesReservedValues(t *testing.T) {
	assignment := types.JobAssignment{
		JobID:     "job-123",
		ModelName: "managed-model",
		Input:     map[string]interface{}{"prompt": "hello"},
		Parameters: map[string]interface{}{
			"job_id":     "attacker-job",
			"model_name": "attacker-model",
			"input_data": "attacker-input",
			"seed":       "42",
		},
	}
	drive := coordinatorDrive{ExtraEnv: map[string]string{
		"JOB_ID":         "coordinator-job",
		"MODEL_NAME":     "coordinator-model",
		"INPUT_DATA":     "coordinator-input",
		"HF_HUB_OFFLINE": "1",
	}}

	env := buildContainerEnv(assignment, drive)
	if env["JOB_ID"] != assignment.JobID || env["MODEL_NAME"] != assignment.ModelName ||
		env["INPUT_DATA"] != `{"prompt":"hello"}` {
		t.Fatalf("reserved environment was overwritten: %v", env)
	}
	if env["SEED"] != "42" || env["HF_HUB_OFFLINE"] != "1" {
		t.Fatalf("ordinary environment was not preserved: %v", env)
	}
}

func TestAssignmentNeedsCustomMount(t *testing.T) {
	if !assignmentNeedsCustomMount(types.JobAssignment{WorkflowJSON: "{}"}, coordinatorDrive{}) {
		t.Fatal("workflow should mount /custom")
	}
	if !assignmentNeedsCustomMount(types.JobAssignment{}, coordinatorDrive{Script: "x"}) {
		t.Fatal("drive script should mount /custom")
	}
}

func TestApplyStageMountsSkipsMissing(t *testing.T) {
	staging := t.TempDir()
	mounts := applyStageMounts(nil, staging, []stageMount{{From: "models", To: "/models", RO: true}})
	if len(mounts) != 0 {
		t.Fatalf("missing stage dir should not mount: %v", mounts)
	}
}

func TestApplyStageMountsUsesExistingDir(t *testing.T) {
	staging := t.TempDir()
	models := filepath.Join(staging, "models")
	if err := os.Mkdir(models, 0o700); err != nil {
		t.Fatal(err)
	}
	mounts := applyStageMounts(nil, staging, []stageMount{{From: "models", To: "/models", RO: true}})
	if len(mounts) != 1 || mounts[0] != models+":/models:ro" {
		t.Fatalf("mounts = %v", mounts)
	}
}

func TestCollectDriveOutputRequiresAFile(t *testing.T) {
	rt := &customDockerRuntime{}
	dir := t.TempDir()
	if _, err := rt.collectDriveOutput("img:v1", dir); err == nil || !strings.Contains(err.Error(), "no file") {
		t.Fatalf("empty output dir should fail: %v", err)
	}
}

func TestCollectDriveOutputPrefersMediaFile(t *testing.T) {
	rt := &customDockerRuntime{}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("log"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "out.png"), []byte("png"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := rt.collectDriveOutput("ghcr.io/example/comfy@sha256:abc", dir)
	if err != nil {
		t.Fatal(err)
	}
	if got["status"] != "completed" || got["backend"] != "docker-custom" {
		t.Fatalf("result = %+v", got)
	}
	output, _ := got["output_file"].(string)
	if filepath.Base(output) != "out.png" {
		t.Fatalf("output_file = %q", output)
	}
}

type recordingRuntime struct {
	prepared []types.JobAssignment
	ran      []types.JobAssignment
}

func (r *recordingRuntime) Name() string { return "docker-custom" }
func (r *recordingRuntime) Prepare(_ context.Context, a types.JobAssignment) error {
	r.prepared = append(r.prepared, a)
	return nil
}
func (r *recordingRuntime) Run(_ context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	r.ran = append(r.ran, a)
	return map[string]interface{}{"status": "completed", "backend": "docker-custom"}, nil
}
func (r *recordingRuntime) Cleanup(bool) error { return nil }

func TestThinLayerExecutorRunsComfyBatchThroughDockerCustom(t *testing.T) {
	rec := &recordingRuntime{}
	exec := &Executor{
		runtime:  &multiRuntime{dockerCustom: rec, hasDocker: true},
		gpuID:    "gpu-thin",
		backend:  "cuda",
		cacheDir: t.TempDir(),
	}
	assignment := types.JobAssignment{
		JobID:       "thin-comfy-1",
		ModelName:   "sdxl-base",
		Runtime:     "comfyui-batch",
		DockerImage: "ghcr.io/example/comfy@sha256:abc",
		Input:       map[string]interface{}{"prompt": "a lighthouse"},
		Parameters: map[string]interface{}{
			"_drive_script": "print(1)",
			"_entrypoint":   "python",
			"_command":      []interface{}{"-c", "pass"},
		},
	}
	result := exec.Execute(context.Background(), assignment)
	if !result.Success {
		t.Fatalf("failed: %s", result.Error)
	}
	if len(rec.prepared) != 1 || len(rec.ran) != 1 {
		t.Fatalf("prepared=%d ran=%d", len(rec.prepared), len(rec.ran))
	}
	if rec.ran[0].Runtime != "comfyui-batch" {
		t.Fatalf("runtime = %q", rec.ran[0].Runtime)
	}
	drive := parseCoordinatorDrive(rec.ran[0].Parameters)
	if drive.Script != "print(1)" || drive.Entrypoint != "python" || len(drive.Command) != 2 {
		t.Fatalf("drive not preserved: %+v", drive)
	}
	if result.Result["backend"] != "cuda" {
		t.Fatalf("host backend = %v", result.Result["backend"])
	}
}
