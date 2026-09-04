package job

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type stageMount struct {
	From string
	To   string
	RO   bool
}

type coordinatorDrive struct {
	Script        string
	Entrypoint    string
	Command       []string
	OutputMount   string
	UseHostMemory bool
	ExtraEnv      map[string]string
	StageMounts   []stageMount
}

func isCoordinatorParam(key string) bool {
	return strings.HasPrefix(key, "_")
}

func parseCoordinatorDrive(params map[string]interface{}) coordinatorDrive {
	drive := coordinatorDrive{ExtraEnv: map[string]string{}}
	if params == nil {
		return drive
	}
	if script, ok := params["_drive_script"].(string); ok {
		drive.Script = script
	}
	if entry, ok := params["_entrypoint"].(string); ok {
		drive.Entrypoint = strings.TrimSpace(entry)
	}
	drive.Command = stringSliceParam(params["_command"])
	if mount, ok := params["_output_mount"].(string); ok {
		drive.OutputMount = strings.TrimSpace(mount)
	}
	switch v := params["_use_host_memory"].(type) {
	case bool:
		drive.UseHostMemory = v
	case string:
		drive.UseHostMemory = strings.EqualFold(v, "true")
	}
	if extra, ok := params["_extra_env"].(map[string]interface{}); ok {
		for key, value := range extra {
			if s, ok := value.(string); ok && key != "" {
				drive.ExtraEnv[key] = s
			}
		}
	}
	if mounts, ok := params["_stage_mounts"].([]interface{}); ok {
		for _, raw := range mounts {
			item, ok := raw.(map[string]interface{})
			if !ok {
				continue
			}
			from, _ := item["from"].(string)
			to, _ := item["to"].(string)
			from = filepath.ToSlash(strings.TrimSpace(from))
			to = strings.TrimSpace(to)
			if from == "" || to == "" || strings.Contains(from, "..") || strings.HasPrefix(from, "/") {
				continue
			}
			ro := true
			if explicit, ok := item["ro"].(bool); ok {
				ro = explicit
			}
			drive.StageMounts = append(drive.StageMounts, stageMount{From: from, To: to, RO: ro})
		}
	}
	return drive
}

func stringSliceParam(value interface{}) []string {
	switch items := value.(type) {
	case []interface{}:
		out := make([]string, 0, len(items))
		for _, item := range items {
			if s, ok := item.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return items
	default:
		return nil
	}
}

func (r *customDockerRuntime) execDriveScript(ctx context.Context, name, outputDir, script string, timeout time.Duration) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}
	clientPath := filepath.Join(outputDir, ".rungpu_drive.py")
	if err := os.WriteFile(clientPath, []byte(script), 0o644); err != nil {
		return fmt.Errorf("write drive script: %w", err)
	}
	defer os.Remove(clientPath)

	readyTimeout := 3 * time.Minute
	if timeout > 0 && timeout < readyTimeout {
		readyTimeout = timeout
	}
	jobTimeout := timeout - readyTimeout
	if jobTimeout < 30*time.Second {
		jobTimeout = 30 * time.Second
	}

	execCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		execCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	args := []string{
		"python", "/output/.rungpu_drive.py",
		strconv.Itoa(int(readyTimeout.Seconds())),
		strconv.Itoa(int(jobTimeout.Seconds())),
	}
	if _, err := r.docker.Exec(execCtx, name, args); err != nil {
		if _, fallbackErr := r.docker.Exec(execCtx, name, append([]string{"python3"}, args[1:]...)); fallbackErr != nil {
			logs, _ := r.docker.Logs(ctx, name, 200)
			return fmt.Errorf("drive script failed: %v\n%s", err, strings.TrimSpace(logs))
		}
	}
	return nil
}

func (r *customDockerRuntime) collectDriveOutput(img, outputDir string) (map[string]interface{}, error) {
	parsed := map[string]interface{}{
		"status":  "completed",
		"backend": "docker-custom",
		"image":   img,
	}
	if outputFile := findOutputFile(outputDir); outputFile != "" {
		parsed["output_file"] = outputFile
		return parsed, nil
	}
	return nil, fmt.Errorf("drive script produced no file under /output")
}

func applyStageMounts(mounts []string, stagingDir string, specs []stageMount) []string {
	for _, spec := range specs {
		hostPath := filepath.Join(stagingDir, filepath.FromSlash(spec.From))
		if info, err := os.Stat(hostPath); err != nil || !info.IsDir() {
			continue
		}
		mount := hostPath + ":" + spec.To
		if spec.RO {
			mount += ":ro"
		}
		mounts = append(mounts, mount)
	}
	return mounts
}

func hasCustomMount(mounts []string) bool {
	for _, mount := range mounts {
		if strings.Contains(mount, ":/custom") {
			return true
		}
	}
	return false
}

func assignmentNeedsCustomMount(a types.JobAssignment, drive coordinatorDrive) bool {
	return drive.Script != "" || a.WorkflowJSON != "" || len(a.CustomFiles) > 0
}
