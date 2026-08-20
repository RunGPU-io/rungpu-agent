// Package dockermgr drives container lifecycle via the local `docker` CLI.
// Every container is sandboxed: no-new-privileges, pids-limit, mount validation,
// and image allowlist enforcement.
package dockermgr

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

type Manager struct {
	Policy SecurityPolicy
}

func New() *Manager {
	return &Manager{Policy: DefaultPolicy()}
}

// NewWithPolicy creates a manager with a custom security policy.
func NewWithPolicy(policy SecurityPolicy) *Manager {
	return &Manager{Policy: policy}
}

// RunOptions configures a detached container run.
type RunOptions struct {
	Image      string
	Name       string
	Env        map[string]string
	Mounts     []string // "hostPath:containerPath[:ro]" (bind mounts)
	Volumes    []string // "namedVolume:containerPath" (Docker named volumes — persistent)
	Ports      []string // "hostPort:containerPort"
	Network    string   // required: "none" for batch jobs or "bridge" for workspaces
	UseGPU     bool
	GPUDevice  string   // "" or "all" → all GPUs; else passed as --gpus device=<GPUDevice>
	ShmSize    string   // shared memory size (e.g. "8g")
	Entrypoint string   // trusted runtime entrypoint override
	Command    []string // override CMD
	// UseHostMemory removes the Docker memory cap for a trusted platform
	// runtime. The Docker host/VM remains the hard limit. Never set this from
	// renter-controlled input.
	UseHostMemory bool
}

// Run starts a sandboxed detached container and returns its ID.
// Enforces: image allowlist, mount validation, no-new-privileges, pids-limit.
func (m *Manager) Run(ctx context.Context, opts RunOptions) (string, error) {
	// ── Security checks ─────────────────────────────────────────────────
	if err := ValidateImage(opts.Image, m.Policy); err != nil {
		return "", fmt.Errorf("security: %w", err)
	}
	if err := ValidateMounts(opts.Mounts); err != nil {
		return "", fmt.Errorf("security: %w", err)
	}
	if opts.Network != "none" && opts.Network != "bridge" {
		return "", fmt.Errorf("security: container network must be none or bridge")
	}

	// ── Build docker run command ────────────────────────────────────────
	args := m.buildRunArgs(opts)

	out, err := exec.CommandContext(ctx, "docker", args...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker run failed: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

// buildRunArgs assembles the `docker run` arguments for opts, applying the
// sandbox policy and GPU scoping. Pure (no side effects) so it can be tested.
func (m *Manager) buildRunArgs(opts RunOptions) []string {
	args := []string{"run", "-d", "--name", opts.Name}

	// Sandbox args (always applied — renter cannot override). Managed media
	// decoding can temporarily need nearly all RAM exposed by Docker, so its
	// trusted runtime may rely on the host/VM limit instead of a lower
	// per-container cap.
	policy := m.Policy
	if opts.UseHostMemory {
		policy.MaxMemoryGB = 0
	}
	args = append(args, SandboxArgs(policy)...)
	args = append(args, "--network", opts.Network)

	if opts.UseGPU {
		// Scope to a specific device when configured so a single job on a
		// multi-GPU host can't grab every GPU; default to all otherwise.
		if dev := strings.TrimSpace(opts.GPUDevice); dev != "" && strings.ToLower(dev) != "all" {
			args = append(args, "--gpus", "device="+dev)
		} else {
			args = append(args, "--gpus", "all")
		}
	}
	if opts.ShmSize != "" {
		args = append(args, "--shm-size", opts.ShmSize)
	}
	for _, mnt := range opts.Mounts {
		args = append(args, "-v", mnt)
	}
	// Named volumes — Docker manages these; they persist across container
	// restarts and removals. Used for model caches, custom nodes, etc.
	for _, vol := range opts.Volumes {
		args = append(args, "-v", vol)
	}
	for _, p := range opts.Ports {
		args = append(args, "-p", p)
	}
	for k, v := range opts.Env {
		args = append(args, "-e", fmt.Sprintf("%s=%s", k, v))
	}
	if opts.Entrypoint != "" {
		args = append(args, "--entrypoint", opts.Entrypoint)
	}
	args = append(args, opts.Image)
	args = append(args, opts.Command...)
	return args
}

// Logs returns up to `tail` lines of a container's combined output.
func (m *Manager) Logs(ctx context.Context, name string, tail int) (string, error) {
	out, err := exec.CommandContext(ctx, "docker", "logs", "--tail", strconv.Itoa(tail), name).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker logs failed: %w", err)
	}
	return string(out), nil
}

// Inspect returns the container's running state and exit code.
func (m *Manager) Inspect(ctx context.Context, name string) (running bool, exitCode int, err error) {
	out, err := exec.CommandContext(ctx, "docker", "inspect",
		"-f", "{{.State.Running}} {{.State.ExitCode}}", name).Output()
	if err != nil {
		return false, 0, err
	}
	parts := strings.Fields(strings.TrimSpace(string(out)))
	if len(parts) != 2 {
		return false, 0, fmt.Errorf("unexpected inspect output: %q", string(out))
	}
	running = parts[0] == "true"
	exitCode, _ = strconv.Atoi(parts[1])
	return running, exitCode, nil
}

// Stop stops a container (best-effort).
func (m *Manager) Stop(ctx context.Context, name string) error {
	return exec.CommandContext(ctx, "docker", "stop", name).Run()
}

// Remove force-removes a container (best-effort).
func (m *Manager) Remove(ctx context.Context, name string) error {
	return exec.CommandContext(ctx, "docker", "rm", "-f", name).Run()
}

// Available reports whether the docker CLI is usable.
func (m *Manager) Available(ctx context.Context) bool {
	return exec.CommandContext(ctx, "docker", "version").Run() == nil
}

// ── Cleanup helpers ─────────────────────────────────────────────────────────

// ListTokenizeContainers returns all containers (running + stopped) whose name
// starts with "tokenize-".
func (m *Manager) ListTokenizeContainers(ctx context.Context) ([]ContainerInfo, error) {
	out, err := exec.CommandContext(ctx, "docker", "ps", "-a",
		"--filter", "name=tokenize-",
		"--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}").Output()
	if err != nil {
		return nil, fmt.Errorf("docker ps failed: %w", err)
	}
	var containers []ContainerInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 4)
		if len(parts) < 4 {
			continue
		}
		containers = append(containers, ContainerInfo{
			ID:     parts[0],
			Name:   parts[1],
			Status: parts[2],
			Image:  parts[3],
		})
	}
	return containers, nil
}

// ContainerInfo holds basic info about a Docker container.
type ContainerInfo struct {
	ID     string
	Name   string
	Status string
	Image  string
}

// RemoveAllTokenizeContainers stops and removes all tokenize-* containers.
func (m *Manager) RemoveAllTokenizeContainers(ctx context.Context) (int, error) {
	containers, err := m.ListTokenizeContainers(ctx)
	if err != nil {
		return 0, err
	}
	removed := 0
	for _, c := range containers {
		_ = m.Stop(ctx, c.Name)
		if err := m.Remove(ctx, c.Name); err == nil {
			removed++
		}
	}
	return removed, nil
}

// ListTokenizeVolumes returns all Docker named volumes whose name starts with
// "tokenize-".
func (m *Manager) ListTokenizeVolumes(ctx context.Context) ([]VolumeInfo, error) {
	out, err := exec.CommandContext(ctx, "docker", "volume", "ls",
		"--filter", "name=tokenize-",
		"--format", "{{.Name}}\t{{.Driver}}").Output()
	if err != nil {
		return nil, fmt.Errorf("docker volume ls failed: %w", err)
	}
	var volumes []VolumeInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		driver := "local"
		if len(parts) == 2 {
			driver = parts[1]
		}
		volumes = append(volumes, VolumeInfo{Name: parts[0], Driver: driver})
	}
	return volumes, nil
}

// VolumeInfo holds basic info about a Docker volume.
type VolumeInfo struct {
	Name   string
	Driver string
}

// RemoveVolume removes a single Docker named volume.
func (m *Manager) RemoveVolume(ctx context.Context, name string) error {
	return exec.CommandContext(ctx, "docker", "volume", "rm", name).Run()
}

// ListTokenizeImages returns Docker images used by tokenize (workspace images,
// model images, etc.).
func (m *Manager) ListTokenizeImages(ctx context.Context) ([]ImageInfo, error) {
	// Get images from known workspace specs + our own images
	knownPrefixes := []string{
		"ghcr.io/ai-dock/",
		"ghcr.io/tokenize/",
		"ghcr.io/invoke-ai/",
		"registry.hf.space/",
		"jupyter/pytorch-notebook",
		"atinoda/text-generation-webui",
		"tokenize-worker",
	}

	out, err := exec.CommandContext(ctx, "docker", "images",
		"--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}").Output()
	if err != nil {
		return nil, fmt.Errorf("docker images failed: %w", err)
	}

	var images []ImageInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) < 3 {
			continue
		}
		repo := parts[0]
		for _, prefix := range knownPrefixes {
			if strings.HasPrefix(repo, prefix) {
				images = append(images, ImageInfo{
					Repository: repo,
					ID:         parts[1],
					Size:       parts[2],
				})
				break
			}
		}
	}
	return images, nil
}

// ImageInfo holds basic info about a Docker image.
type ImageInfo struct {
	Repository string
	ID         string
	Size       string
}

// RemoveImage removes a Docker image by ID.
func (m *Manager) RemoveImage(ctx context.Context, id string) error {
	return exec.CommandContext(ctx, "docker", "rmi", "-f", id).Run()
}
