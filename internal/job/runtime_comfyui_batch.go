package job

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

var comfyUIBatchVolumes = []string{
	"tokenize-comfyui-models:/opt/ComfyUI/models",
	"tokenize-comfyui-nodes:/opt/ComfyUI/custom_nodes",
	"tokenize-comfyui-input:/opt/ComfyUI/input",
}

type comfyUIBatchRuntime struct {
	docker    *dockermgr.Manager
	cacheDir  string
	useGPU    bool
	gpuDevice string
}

func newComfyUIBatchRuntime(cacheDir string, useGPU bool, gpuDevice string, policy dockermgr.SecurityPolicy) *comfyUIBatchRuntime {
	return &comfyUIBatchRuntime{
		docker: dockermgr.NewWithPolicy(policy), cacheDir: cacheDir,
		useGPU: useGPU, gpuDevice: gpuDevice,
	}
}

func (r *comfyUIBatchRuntime) Name() string { return "comfyui-batch" }

func (r *comfyUIBatchRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	if _, err := r.workflowPath(a); err != nil {
		return err
	}
	if a.DockerImage == "" {
		return fmt.Errorf("ComfyUI generation requires a coordinator-selected docker_image")
	}
	if !r.docker.Available(ctx) {
		return fmt.Errorf("ComfyUI generation requires Docker on the GPU host")
	}
	if err := dockermgr.ValidateImage(a.DockerImage, r.docker.Policy); err != nil {
		return fmt.Errorf("security: %w", err)
	}
	if out, err := exec.CommandContext(ctx, "docker", "image", "inspect", a.DockerImage).CombinedOutput(); err != nil || len(out) < 3 {
		if pullOut, pullErr := exec.CommandContext(ctx, "docker", "pull", a.DockerImage).CombinedOutput(); pullErr != nil {
			return fmt.Errorf("pull internal ComfyUI runtime: %v: %s", pullErr, strings.TrimSpace(string(pullOut)))
		}
	}
	return nil
}

func (r *comfyUIBatchRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	workflowPath, err := r.workflowPath(a)
	if err != nil {
		return nil, err
	}
	workflowData, err := os.ReadFile(workflowPath)
	if err != nil {
		return nil, fmt.Errorf("read staged workflow: %w", err)
	}
	var workflow map[string]interface{}
	if err := json.Unmarshal(workflowData, &workflow); err != nil {
		return nil, fmt.Errorf("workflow is not valid JSON: %w", err)
	}
	if wrapped, ok := workflow["workflow"].(map[string]interface{}); ok {
		workflow = wrapped
	}
	if !injectComfyPrompt(workflow, extractPrompt(a)) {
		return nil, fmt.Errorf("workflow must be ComfyUI API format and contain {{prompt}} or a positive CLIPTextEncode node")
	}

	requestPath := filepath.Join(filepath.Dir(workflowPath), "rungpu-request.json")
	requestData, err := json.Marshal(map[string]interface{}{"prompt": workflow, "client_id": "rungpu-" + a.JobID})
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(requestPath, requestData, 0o600); err != nil {
		return nil, fmt.Errorf("write prepared workflow: %w", err)
	}

	outputDir := filepath.Join(r.cacheDir, "output", a.JobID)
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		return nil, fmt.Errorf("create output directory: %w", err)
	}
	containerName := CustomContainerName(a.JobID)
	stagingDir := filepath.Join(r.cacheDir, "staging", a.JobID)
	mounts, volumes, err := comfyUIBatchStorage(stagingDir)
	if err != nil {
		return nil, err
	}
	mounts = append(mounts, stagingDir+":/custom:ro", outputDir+":/opt/ComfyUI/output")
	if _, err := r.docker.Run(ctx, dockermgr.RunOptions{
		Image: a.DockerImage, Name: containerName,
		UseGPU: r.useGPU, GPUDevice: r.gpuDevice,
		Network: "none",
		Mounts:  mounts, Volumes: volumes, ShmSize: "8g",
		// ai-dock documents these settings in its docker-compose example, but
		// they are not baked into the image config. In particular, WAN address
		// discovery and quick tunnels cannot succeed under Network=none and may
		// delay or prevent the supervised ComfyUI service from becoming ready.
		// Batch inference needs only the container-local API, so keep it bound
		// to loopback and disable every network-dependent startup feature.
		Env: comfyUIBatchEnv(),
	}); err != nil {
		return nil, fmt.Errorf("start internal ComfyUI runtime: %w", err)
	}
	defer func() {
		_ = r.docker.Stop(context.Background(), containerName)
		_ = r.docker.Remove(context.Background(), containerName)
	}()

	requestContainerPath := "/custom/workflows/rungpu-request.json"
	cmd := exec.CommandContext(ctx, "docker", "exec", containerName, "python3", "-c", comfyUIRunner(), requestContainerPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			message = err.Error()
		}
		if running, exitCode, inspectErr := r.docker.Inspect(context.Background(), containerName); inspectErr == nil && !running {
			message = fmt.Sprintf("%s (ComfyUI container exited with code %d)", message, exitCode)
		}
		diagnostics := r.comfyUIDiagnostics(containerName)
		if diagnostics != "" {
			message += "\n" + diagnostics
		}
		return nil, fmt.Errorf("ComfyUI workflow failed: %s", message)
	}

	var generated struct {
		Filename  string `json:"filename"`
		Subfolder string `json:"subfolder"`
		Type      string `json:"type"`
	}
	if err := json.Unmarshal(output, &generated); err != nil || generated.Filename == "" {
		return nil, fmt.Errorf("ComfyUI returned no generated media: %s", strings.TrimSpace(string(output)))
	}
	outputPath, err := safeComfyOutputPath(outputDir, generated.Subfolder, generated.Filename)
	if err != nil {
		return nil, err
	}
	outputPath, err = validateOutputFile(outputDir, outputPath)
	if err != nil {
		return nil, fmt.Errorf("generated media is unavailable: %w", err)
	}
	return map[string]interface{}{
		"status": "completed", "backend": "comfyui-batch",
		"output_file": outputPath, "filename": generated.Filename, "output_type": generated.Type,
	}, nil
}

func (r *comfyUIBatchRuntime) Cleanup(force bool) error { return nil }

func comfyUIBatchEnv() map[string]string {
	return map[string]string{
		"AUTO_UPDATE":            "false",
		"DIRECT_ADDRESS":         "127.0.0.1",
		"DIRECT_ADDRESS_GET_WAN": "false",
		"WORKSPACE":              "/workspace",
		"WORKSPACE_SYNC":         "false",
		"CF_QUICK_TUNNELS":       "false",
		"WEB_ENABLE_AUTH":        "false",
		"WEB_ENABLE_HTTPS":       "false",
		"COMFYUI_ARGS":           "--listen 127.0.0.1 --port 8188",
		"COMFYUI_PORT_HOST":      "8188",
		"SERVERLESS":             "false",
	}
}

// comfyUIDiagnostics captures service state and recent logs before the deferred
// cleanup removes the failed container. This is intentionally best-effort:
// diagnostics must never replace the original workflow error.
func (r *comfyUIBatchRuntime) comfyUIDiagnostics(containerName string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	parts := make([]string, 0, 2)
	if output, err := exec.CommandContext(
		ctx, "docker", "exec", containerName,
		"supervisorctl", "status", "comfyui",
	).CombinedOutput(); err == nil || len(output) > 0 {
		parts = append(parts, "ComfyUI service status:\n"+strings.TrimSpace(string(output)))
	}
	if logs, err := r.docker.Logs(ctx, containerName, 200); err == nil && strings.TrimSpace(logs) != "" {
		parts = append(parts, "Recent container logs:\n"+strings.TrimSpace(logs))
	}
	return strings.Join(parts, "\n")
}

func comfyUIBatchStorage(stagingDir string) ([]string, []string, error) {
	modelsDir := filepath.Join(stagingDir, "models")
	info, err := os.Stat(modelsDir)
	if os.IsNotExist(err) {
		return nil, append([]string(nil), comfyUIBatchVolumes...), nil
	}
	if err != nil {
		return nil, nil, fmt.Errorf("inspect staged ComfyUI models: %w", err)
	}
	if !info.IsDir() {
		return nil, nil, fmt.Errorf("staged ComfyUI models path is not a directory")
	}
	volumes := make([]string, 0, len(comfyUIBatchVolumes)-1)
	for _, volume := range comfyUIBatchVolumes {
		if !strings.HasSuffix(volume, ":/opt/ComfyUI/models") {
			volumes = append(volumes, volume)
		}
	}
	return []string{modelsDir + ":/opt/ComfyUI/models:ro"}, volumes, nil
}

func (r *comfyUIBatchRuntime) workflowPath(a types.JobAssignment) (string, error) {
	if a.Parameters == nil || a.Parameters["runtime"] != "comfyui-batch" {
		return "", fmt.Errorf("ComfyUI batch runtime was not requested")
	}
	path := filepath.Join(r.cacheDir, "staging", a.JobID, "workflows", "workflow.json")
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return "", fmt.Errorf("public ComfyUI API workflow was not staged")
	}
	return path, nil
}

func injectComfyPrompt(workflow map[string]interface{}, prompt string) bool {
	replaced := replacePromptPlaceholders(workflow, prompt)
	if replaced {
		return true
	}
	for _, value := range workflow {
		node, ok := value.(map[string]interface{})
		if !ok || node["class_type"] != "CLIPTextEncode" {
			continue
		}
		meta, _ := node["_meta"].(map[string]interface{})
		title, _ := meta["title"].(string)
		inputs, _ := node["inputs"].(map[string]interface{})
		text, _ := inputs["text"].(string)
		if inputs == nil || strings.Contains(strings.ToLower(title+" "+text), "negative") {
			continue
		}
		inputs["text"] = prompt
		return true
	}
	return false
}

func replacePromptPlaceholders(value interface{}, prompt string) bool {
	replaced := false
	switch current := value.(type) {
	case map[string]interface{}:
		for key, child := range current {
			if text, ok := child.(string); ok && strings.Contains(text, "{{prompt}}") {
				current[key] = strings.ReplaceAll(text, "{{prompt}}", prompt)
				replaced = true
				continue
			}
			if replacePromptPlaceholders(child, prompt) {
				replaced = true
			}
		}
	case []interface{}:
		for index, child := range current {
			if text, ok := child.(string); ok && strings.Contains(text, "{{prompt}}") {
				current[index] = strings.ReplaceAll(text, "{{prompt}}", prompt)
				replaced = true
				continue
			}
			if replacePromptPlaceholders(child, prompt) {
				replaced = true
			}
		}
	}
	return replaced
}

func safeComfyOutputPath(root, subfolder, filename string) (string, error) {
	candidate := filepath.Join(root, filepath.Clean(subfolder), filepath.Base(filename))
	relative, err := filepath.Rel(root, candidate)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("ComfyUI returned an invalid output path")
	}
	return candidate, nil
}

func validateOutputFile(root, candidate string) (string, error) {
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(resolvedRoot, resolved)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("output resolves outside the job directory")
	}
	info, err := os.Lstat(resolved)
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("output is not a regular file")
	}
	return resolved, nil
}

const comfyUIRunnerScript = `
import json, sys, time, urllib.error, urllib.request
request_path = sys.argv[1]
base = "http://127.0.0.1:8188"
last_error = "no readiness attempt completed"

def read_json(url, timeout):
	with urllib.request.urlopen(url, timeout=timeout) as response:
		body = response.read()
		if not body.strip():
			raise ValueError("empty response")
		return json.loads(body)

for _ in range(300):
    try:
		read_json(base + "/system_stats", 2)
		object_info = read_json(base + "/object_info", 10)
		if not isinstance(object_info, dict) or not object_info:
			raise ValueError("empty object registry")
        break
    except Exception as error:
		last_error = type(error).__name__ + ": " + str(error)
        time.sleep(1)
else:
    raise SystemExit("ComfyUI did not become ready after 300 seconds; last probe: " + last_error)
with open(request_path, "rb") as handle:
    payload = handle.read()
request = urllib.request.Request(base + "/prompt", data=payload, headers={"Content-Type": "application/json"})
try:
    queued = read_json(request, 30)
except urllib.error.HTTPError as error:
    detail = error.read().decode("utf-8", "replace").strip()
    raise SystemExit("workflow rejected: HTTP " + str(error.code) + (": " + detail[-4000:] if detail else ""))
except Exception as error:
    raise SystemExit("workflow submission failed: " + type(error).__name__ + ": " + str(error))
prompt_id = queued.get("prompt_id")
if not prompt_id:
    raise SystemExit("workflow rejected: " + json.dumps(queued))
for _ in range(3600):
	history = read_json(base + "/history/" + prompt_id, 10)
    item = history.get(prompt_id)
    if item:
        status = item.get("status", {})
        if status.get("status_str") == "error":
            raise SystemExit("workflow execution failed: " + json.dumps(status.get("messages", []))[-4000:])
        for output in item.get("outputs", {}).values():
            for key in ("images", "gifs", "videos", "audio"):
                files = output.get(key, [])
                if files:
                    print(json.dumps(files[0]))
                    raise SystemExit(0)
        if status.get("completed"):
            raise SystemExit("workflow completed without media output")
    time.sleep(1)
raise SystemExit("workflow timed out")
`

func comfyUIRunner() string {
	return strings.ReplaceAll(comfyUIRunnerScript, "\t", "    ")
}
