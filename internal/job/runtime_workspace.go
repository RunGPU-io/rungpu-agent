package job

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type workspaceRuntime struct {
	docker    *dockermgr.Manager
	cacheDir  string
	useGPU    bool
	gpuDevice string
	hostIP    string
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

func resolveWorkspace(a types.JobAssignment) (image string, ports []string, shmSize string, volumes map[string]string, cmd []string) {
	params := a.Parameters
	if params == nil {
		params = map[string]interface{}{}
	}

	if a.DockerImage != "" {
		image = a.DockerImage
	} else if img, ok := params["docker_image"].(string); ok && img != "" {
		image = img
	}

	if len(a.Ports) > 0 {
		ports = append([]string(nil), a.Ports...)
	} else if p, ok := params["ports"].(string); ok && p != "" {
		ports = strings.Split(p, ",")
	}
	for i := range ports {
		ports[i] = strings.TrimSpace(ports[i])
		if !strings.Contains(ports[i], ":") {
			ports[i] = ports[i] + ":" + ports[i]
		}
	}

	if s, ok := params["shm_size"].(string); ok && s != "" {
		shmSize = s
	}
	if shmSize == "" {
		shmSize = "4g"
	}

	if c, ok := params["command"].(string); ok && c != "" {
		cmd = strings.Fields(c)
	}

	return
}

func (r *workspaceRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	image, _, _, _, _ := resolveWorkspace(a)
	if image == "" {
		return fmt.Errorf("workspace assignment requires docker_image")
	}

	if err := dockermgr.ValidateImage(image, dockermgr.WithAssignedImage(r.docker.Policy, image)); err != nil {
		return fmt.Errorf("security: %w", err)
	}

	if !r.docker.Available(ctx) {
		return fmt.Errorf("docker is not available — install Docker to run workspace containers")
	}

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
	if a.ResourceName != "" {
		containerName = a.ResourceName
	}

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

	workspaceDir := r.cacheDir + "/workspaces/" + a.JobID
	mounts := []string{workspaceDir + ":/workspace"}
	if len(a.CustomFiles) > 0 {
		stagingDir := filepath.Join(r.cacheDir, "staging", a.JobID)
		mounts = append(mounts, stagingDir+":/custom:ro")
	}

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
		Image:            image,
		Name:             containerName,
		Labels:           a.ResourceLabels,
		ExpectedDaemonID: a.ResourceDaemonID,
		Network:          "bridge",
		UseGPU:           r.useGPU,
		GPUDevice:        r.gpuDevice,
		Ports:            ports,
		Mounts:           mounts,
		Volumes:          namedVolumes,
		Env:              env,
		ShmSize:          shmSize,
		Command:          cmd,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to start workspace %s: %w", image, err)
	}

	time.Sleep(3 * time.Second)

	running, _, err := r.docker.Inspect(ctx, containerName)
	if err != nil || !running {
		logs, _ := r.docker.Logs(ctx, containerName, 50)
		if a.ResourceName != "" {
			_ = r.docker.RemoveOwnedAndVerify(context.Background(), dockermgr.OwnedResource{Name: containerName, Labels: a.ResourceLabels, DaemonID: a.ResourceDaemonID})
		} else {
			_ = r.docker.Remove(context.Background(), containerName)
		}
		return nil, fmt.Errorf("workspace container failed to start:\n%s", logs)
	}

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

	if len(namedVolumes) > 0 {
		result["persistent_volumes"] = namedVolumes
		result["storage_note"] = "Models and custom nodes are stored in Docker named volumes. " +
			"They persist across container restarts — no re-downloading needed."
	}

	return result, nil
}

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

	return nil
}
