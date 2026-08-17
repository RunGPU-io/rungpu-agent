package job

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const comfyUIBatchImage = "ghcr.io/ai-dock/comfyui:latest"

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
	if !r.docker.Available(ctx) {
		return fmt.Errorf("ComfyUI generation requires Docker on the GPU host")
	}
	if err := dockermgr.ValidateImage(comfyUIBatchImage, r.docker.Policy); err != nil {
		return fmt.Errorf("security: %w", err)
	}
	if out, err := exec.CommandContext(ctx, "docker", "image", "inspect", comfyUIBatchImage).CombinedOutput(); err != nil || len(out) < 3 {
		if pullOut, pullErr := exec.CommandContext(ctx, "docker", "pull", comfyUIBatchImage).CombinedOutput(); pullErr != nil {
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
		Image: comfyUIBatchImage, Name: containerName,
		UseGPU: r.useGPU, GPUDevice: r.gpuDevice,
		Network: "none",
		Mounts: mounts, Volumes: volumes, ShmSize: "8g",
	}); err != nil {
		return nil, fmt.Errorf("start internal ComfyUI runtime: %w", err)
	}
	defer func() {
		_ = r.docker.Stop(context.Background(), containerName)
		_ = r.docker.Remove(context.Background(), containerName)
	}()

	requestContainerPath := "/custom/workflows/rungpu-request.json"
	cmd := exec.CommandContext(ctx, "docker", "exec", containerName, "python3", "-c", comfyUIRunnerScript, requestContainerPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			message = err.Error()
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
import json, sys, time, urllib.request
request_path = sys.argv[1]
base = "http://127.0.0.1:8188"
for _ in range(180):
    try:
        urllib.request.urlopen(base + "/system_stats", timeout=2).read()
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("ComfyUI did not become ready")
with open(request_path, "rb") as handle:
    payload = handle.read()
request = urllib.request.Request(base + "/prompt", data=payload, headers={"Content-Type": "application/json"})
try:
    queued = json.loads(urllib.request.urlopen(request, timeout=30).read())
except Exception as error:
    raise SystemExit("workflow rejected: " + str(error))
prompt_id = queued.get("prompt_id")
if not prompt_id:
    raise SystemExit("workflow rejected: " + json.dumps(queued))
for _ in range(3600):
    history = json.loads(urllib.request.urlopen(base + "/history/" + prompt_id, timeout=10).read())
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
