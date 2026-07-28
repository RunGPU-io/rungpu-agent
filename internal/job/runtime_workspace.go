package job

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/RunGPU-io/gpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/gpu-agent/internal/types"
)

// workspaceRuntime runs long-lived interactive containers (ComfyUI, Jupyter,
// custom workflows). Unlike the inference runtimes that run a job and exit,
// a workspace stays running for the duration of the rental with exposed ports.
//
// Storage strategy — Docker named volumes for persistence:
//
//   Models, custom nodes, and other large assets are stored in Docker named
//   volumes (e.g. "tokenize-comfyui-models"). These survive container removal
//   and are reused across jobs, so a 10 GB SDXL checkpoint downloaded once is
//   available instantly for every subsequent job. The host never re-downloads
//   models it already has.
//
//   Per-job workspace data (user files, outputs) uses a separate bind mount
//   so it can be cleaned up when the rental ends.
//
// The renter specifies:
//   - docker_image: the full environment (e.g. "comfyanonymous/comfyui:latest")
//   - ports: which ports to expose (e.g. "8188" for ComfyUI, "8888" for Jupyter)
//   - command: optional CMD override
//
// The agent starts the container, reports the access URL back, and keeps it
// running until the rental ends (stop signal from the pool coordinator).
type workspaceRuntime struct {
	docker    *dockermgr.Manager
	cacheDir  string
	useGPU    bool
	gpuDevice string
	hostIP    string // the host's external IP for access URLs
}

func newWorkspaceRuntime(cacheDir string, useGPU bool, gpuDevice string, policy dockermgr.SecurityPolicy) *workspaceRuntime {
	return &workspaceRuntime{
		docker:    dockermgr.NewWithPolicy(policy),
		cacheDir:  cacheDir,
		useGPU:    useGPU,
		gpuDevice: gpuDevice,
		hostIP:    "localhost",
	}
}

func (r *workspaceRuntime) Name() string { return "workspace" }

// workspaceSpec defines a well-known workspace image with its default settings
// and the Docker named volumes that should persist across container restarts.
type workspaceSpec struct {
	Image   string
	Ports   []string
	ShmSize string
	// Volumes maps Docker named volume names → container paths.
	// These persist across container restarts so models, custom nodes, etc.
	// are never re-downloaded.
	Volumes map[string]string
}

// Well-known workspace images with their default ports, commands, and
// persistent volume mappings.
var knownWorkspaces = map[string]workspaceSpec{
	"comfyui": {
		Image:   "ghcr.io/ai-dock/comfyui:latest",
		Ports:   []string{"8188:8188"},
		ShmSize: "8g",
		Volumes: map[string]string{
			// Models (checkpoints, LoRAs, VAEs, ControlNets) — biggest cache, ~50-200 GB
			"tokenize-comfyui-models": "/opt/ComfyUI/models",
			// Custom nodes (ComfyUI Manager installs here)
			"tokenize-comfyui-nodes": "/opt/ComfyUI/custom_nodes",
			// User outputs (generated images/videos)
			"tokenize-comfyui-output": "/opt/ComfyUI/output",
			// Input images for img2img, ControlNet, etc.
			"tokenize-comfyui-input": "/opt/ComfyUI/input",
		},
	},
	"jupyter": {
		Image:   "jupyter/pytorch-notebook:latest",
		Ports:   []string{"8888:8888"},
		ShmSize: "4g",
		Volumes: map[string]string{
			"tokenize-jupyter-work": "/home/jovyan/work",
		},
	},
	"a1111": {
		Image:   "ghcr.io/ai-dock/stable-diffusion-webui:latest",
		Ports:   []string{"7860:7860"},
		ShmSize: "8g",
		Volumes: map[string]string{
			"tokenize-a1111-models":     "/stable-diffusion-webui/models",
			"tokenize-a1111-extensions": "/stable-diffusion-webui/extensions",
			"tokenize-a1111-outputs":    "/stable-diffusion-webui/outputs",
		},
	},
	"invokeai": {
		Image:   "ghcr.io/invoke-ai/invokeai:latest",
		Ports:   []string{"9090:9090"},
		ShmSize: "8g",
		Volumes: map[string]string{
			"tokenize-invokeai-models":  "/invokeai/models",
			"tokenize-invokeai-outputs": "/invokeai/outputs",
		},
	},
	"text-generation-webui": {
		Image:   "atinoda/text-generation-webui:latest",
		Ports:   []string{"7860:7860", "5000:5000"},
		ShmSize: "8g",
		Volumes: map[string]string{
			"tokenize-tgw-models":     "/app/models",
			"tokenize-tgw-characters": "/app/characters",
			"tokenize-tgw-loras":      "/app/loras",
		},
	},
}

// resolveWorkspace figures out the Docker image, ports, and settings for a job.
func resolveWorkspace(a types.JobAssignment) (image string, ports []string, shmSize string, volumes map[string]string, cmd []string) {
	params := a.Parameters
	if params == nil {
		params = map[string]interface{}{}
	}

	// Explicit docker_image always wins
	if img, ok := params["docker_image"].(string); ok && img != "" {
		image = img
	}

	// Check known workspace names
	if image == "" {
		lower := strings.ToLower(a.ModelName)
		if ws, ok := knownWorkspaces[lower]; ok {
			image = ws.Image
			ports = ws.Ports
			shmSize = ws.ShmSize
			volumes = ws.Volumes
		}
	}

	// Override ports from params
	if p, ok := params["ports"].(string); ok && p != "" {
		ports = strings.Split(p, ",")
		for i := range ports {
			ports[i] = strings.TrimSpace(ports[i])
			// If user just says "8188", expand to "8188:8188"
			if !strings.Contains(ports[i], ":") {
				ports[i] = ports[i] + ":" + ports[i]
			}
		}
	}

	// Override shm_size
	if s, ok := params["shm_size"].(string); ok && s != "" {
		shmSize = s
	}
	if shmSize == "" {
		shmSize = "4g"
	}

	// Override command
	if c, ok := params["command"].(string); ok && c != "" {
		cmd = strings.Fields(c)
	}

	return
}

func (r *workspaceRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	image, _, _, _, _ := resolveWorkspace(a)
	if image == "" {
		names := make([]string, 0, len(knownWorkspaces))
		for k := range knownWorkspaces {
			names = append(names, k)
		}
		return fmt.Errorf(
			"no docker_image specified for workspace %q — set Parameters.docker_image or use a known workspace: %s",
			a.ModelName, strings.Join(names, ", "),
		)
	}

	// Enforce the image allowlist before pulling (fail fast, don't pull untrusted).
	if err := dockermgr.ValidateImage(image, r.docker.Policy); err != nil {
		return fmt.Errorf("security: %w", err)
	}

	if !r.docker.Available(ctx) {
		return fmt.Errorf("docker is not available — install Docker to run workspace containers")
	}

	// Pull image if not present
	out, err := exec.CommandContext(ctx, "docker", "image", "inspect", image).CombinedOutput()
	if err != nil || len(out) < 3 {
		fmt.Printf("[workspace] Pulling image %s...\n", image)
		cmd := exec.CommandContext(ctx, "docker", "pull", image)
		if pullOut, pullErr := cmd.CombinedOutput(); pullErr != nil {
			return fmt.Errorf("docker pull %s failed: %v: %s", image, pullErr, strings.TrimSpace(string(pullOut)))
		}
	}

	return nil
}

func (r *workspaceRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	image, ports, shmSize, persistVolumes, cmd := resolveWorkspace(a)
	if image == "" {
		return nil, fmt.Errorf("no docker_image for workspace %q", a.ModelName)
	}

	containerName := WorkspaceContainerName(a.JobID)

	// Bind exposed ports to loopback only. An unauthenticated ComfyUI/Jupyter
	// bound to 0.0.0.0 would be reachable by anyone on the host's network;
	// access is instead brokered through the agent's outbound connection.
	ports = bindPortsLoopback(ports)

	env := map[string]string{
		"JOB_ID":     a.JobID,
		"MODEL_NAME": a.ModelName,
	}
	if a.Parameters != nil {
		for k, v := range a.Parameters {
			if k == "docker_image" || k == "ports" || k == "shm_size" || k == "command" || k == "workspace" {
				continue
			}
			if s, ok := v.(string); ok {
				env[strings.ToUpper(k)] = s
			}
		}
	}

	// Per-job workspace bind mount (user files, scratch space)
	workspaceDir := r.cacheDir + "/workspaces/" + a.JobID
	mounts := []string{workspaceDir + ":/workspace"}

	// Persistent Docker named volumes — survive container removal.
	// Models, custom nodes, extensions, etc. are stored here so they
	// never need to be re-downloaded.
	var namedVolumes []string
	var volumeInfo []string
	if persistVolumes != nil {
		for volName, containerPath := range persistVolumes {
			namedVolumes = append(namedVolumes, volName+":"+containerPath)
			volumeInfo = append(volumeInfo, fmt.Sprintf("  %s → %s", volName, containerPath))
		}
		fmt.Printf("[workspace] Persistent volumes (models survive container restarts):\n")
		for _, info := range volumeInfo {
			fmt.Println(info)
		}
	}

	containerID, err := r.docker.Run(ctx, dockermgr.RunOptions{
		Image:     image,
		Name:      containerName,
		UseGPU:    r.useGPU,
		GPUDevice: r.gpuDevice,
		Ports:     ports,
		Mounts:    mounts,
		Volumes:   namedVolumes,
		Env:       env,
		ShmSize:   shmSize,
		Command:   cmd,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to start workspace %s: %w", image, err)
	}

	// Wait briefly for the container to start
	time.Sleep(3 * time.Second)

	// Check it's actually running
	running, _, err := r.docker.Inspect(ctx, containerName)
	if err != nil || !running {
		logs, _ := r.docker.Logs(ctx, containerName, 50)
		_ = r.docker.Remove(context.Background(), containerName)
		return nil, fmt.Errorf("workspace container failed to start:\n%s", logs)
	}

	// Build access URLs (host port is the second-to-last colon field once bound
	// to loopback, e.g. "127.0.0.1:8188:8188").
	accessURLs := make([]string, 0, len(ports))
	for _, p := range ports {
		if hp := hostPortOf(p); hp != "" {
			accessURLs = append(accessURLs, fmt.Sprintf("http://%s:%s", r.hostIP, hp))
		}
	}

	result := map[string]interface{}{
		"status":       "running",
		"container_id": containerID,
		"container":    containerName,
		"image":        image,
		"access_urls":  accessURLs,
		"ports":        ports,
		"backend":      "workspace",
		"message":      fmt.Sprintf("Workspace is running. Access at: %s", strings.Join(accessURLs, ", ")),
	}

	// Report persistent volumes so the user knows what's cached
	if len(namedVolumes) > 0 {
		result["persistent_volumes"] = namedVolumes
		result["storage_note"] = "Models and custom nodes are stored in Docker named volumes. " +
			"They persist across container restarts — no re-downloading needed."
	}

	// For known workspaces, add specific instructions
	lower := strings.ToLower(a.ModelName)
	switch {
	case lower == "comfyui" || strings.Contains(image, "comfyui"):
		result["instructions"] = "Open the ComfyUI interface in your browser. " +
			"Models, custom nodes, and outputs are stored in persistent Docker volumes — " +
			"they survive container restarts. Install custom nodes via the Manager; " +
			"they'll still be there next time."
	case lower == "jupyter" || strings.Contains(image, "jupyter"):
		result["instructions"] = "Open Jupyter in your browser. The token may be in the container logs. " +
			"Files in /home/jovyan/work are persisted in a Docker volume."
	case lower == "a1111" || strings.Contains(image, "stable-diffusion-webui"):
		result["instructions"] = "Open the Automatic1111 WebUI in your browser. " +
			"Models and extensions are stored in persistent Docker volumes."
	}

	return result, nil
}

// bindPortsLoopback rewrites each port spec to bind on 127.0.0.1 so exposed
// workspace services aren't reachable on the host's public interfaces.
// Accepts "CP", "HP:CP", or "IP:HP:CP" and returns "127.0.0.1:HP:CP".
func bindPortsLoopback(ports []string) []string {
	out := make([]string, 0, len(ports))
	for _, p := range ports {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		f := strings.Split(p, ":")
		var host, cont string
		switch len(f) {
		case 1:
			host, cont = f[0], f[0]
		case 2:
			host, cont = f[0], f[1]
		default:
			host, cont = f[len(f)-2], f[len(f)-1]
		}
		out = append(out, "127.0.0.1:"+host+":"+cont)
	}
	return out
}

// hostPortOf extracts the host port from a docker port spec
// ("CP", "HP:CP", or "IP:HP:CP").
func hostPortOf(p string) string {
	f := strings.Split(strings.TrimSpace(p), ":")
	switch len(f) {
	case 0:
		return ""
	case 1, 2:
		return f[0]
	default:
		return f[len(f)-2]
	}
}

func (r *workspaceRuntime) Cleanup(force bool) error {
	// Workspaces are cleaned up when the rental ends (via pool coordinator).
	// Named volumes are intentionally NOT cleaned up here — they're the
	// persistent model cache that makes subsequent jobs fast.
	// To reclaim space: docker volume prune
	return nil
}
