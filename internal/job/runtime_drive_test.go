package job

import (
	"os"
	"path/filepath"
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
		},
	})
	if len(drive.StageMounts) != 0 {
		t.Fatalf("traversal mounts should be dropped: %+v", drive.StageMounts)
	}
}

func TestCoordinatorParamsAreNotCopiedToEnv(t *testing.T) {
	if !isCoordinatorParam("_drive_script") || isCoordinatorParam("timeout_minutes") {
		t.Fatal("coordinator param detection")
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
