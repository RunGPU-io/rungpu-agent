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

func NewWithPolicy(policy SecurityPolicy) *Manager {
	return &Manager{Policy: policy}
}

type RunOptions struct {
	Image            string
	Name             string
	Env              map[string]string
	Labels           map[string]string
	ExpectedDaemonID string
	Mounts           []string
	Volumes          []string
	Ports            []string
	Network          string
	UseGPU           bool
	GPUDevice        string
	ShmSize          string
	Entrypoint       string
	Command          []string

	UseHostMemory bool
}

func (m *Manager) Run(ctx context.Context, opts RunOptions) (string, error) {
	if opts.ExpectedDaemonID != "" {
		id, err := m.DaemonID(ctx)
		if err != nil || id != opts.ExpectedDaemonID {
			return "", fmt.Errorf("Docker daemon changed before container creation")
		}
	}

	if err := ValidateImage(opts.Image, WithAssignedImage(m.Policy, opts.Image)); err != nil {
		return "", fmt.Errorf("security: %w", err)
	}
	if err := ValidateMounts(opts.Mounts); err != nil {
		return "", fmt.Errorf("security: %w", err)
	}
	if opts.Network != "none" && opts.Network != "bridge" {
		return "", fmt.Errorf("security: container network must be none or bridge")
	}

	args := m.buildRunArgs(opts)

	out, err := exec.CommandContext(ctx, "docker", args...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker run failed: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

func (m *Manager) buildRunArgs(opts RunOptions) []string {
	args := []string{"run", "-d", "--name", opts.Name}

	policy := m.Policy
	if opts.UseHostMemory {
		policy.MaxMemoryGB = 0
	}
	args = append(args, SandboxArgs(policy)...)
	args = append(args, "--network", opts.Network)

	if opts.UseGPU {

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

	for _, vol := range opts.Volumes {
		args = append(args, "-v", vol)
	}
	for _, p := range opts.Ports {
		args = append(args, "-p", p)
	}
	for k, v := range opts.Env {
		args = append(args, "-e", fmt.Sprintf("%s=%s", k, v))
	}
	for k, v := range opts.Labels {
		args = append(args, "--label", k+"="+v)
	}
	if opts.Entrypoint != "" {
		args = append(args, "--entrypoint", opts.Entrypoint)
	}
	args = append(args, opts.Image)
	args = append(args, opts.Command...)
	return args
}

func (m *Manager) Exec(ctx context.Context, name string, command []string) (string, error) {
	if strings.TrimSpace(name) == "" || len(command) == 0 {
		return "", fmt.Errorf("docker exec requires a container name and command")
	}
	args := append([]string{"exec", name}, command...)
	out, err := exec.CommandContext(ctx, "docker", args...).CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("docker exec failed: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func (m *Manager) Logs(ctx context.Context, name string, tail int) (string, error) {
	out, err := exec.CommandContext(ctx, "docker", "logs", "--tail", strconv.Itoa(tail), name).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker logs failed: %w", err)
	}
	return string(out), nil
}

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

func (m *Manager) Stop(ctx context.Context, name string) error {
	return exec.CommandContext(ctx, "docker", "stop", name).Run()
}

func (m *Manager) Remove(ctx context.Context, name string) error {
	return exec.CommandContext(ctx, "docker", "rm", "-f", name).Run()
}

func (m *Manager) RemoveAndVerify(ctx context.Context, name string) error {
	exists := func() (bool, error) {
		out, err := exec.CommandContext(ctx, "docker", "ps", "-a", "--format", "{{.Names}}").Output()
		if err != nil {
			return false, fmt.Errorf("verify container %s: %w", name, err)
		}

		for _, entry := range strings.Fields(string(out)) {
			if entry == name {
				return true, nil
			}
		}
		return false, nil
	}
	present, err := exists()
	if err != nil || !present {
		return err
	}
	if err := m.Remove(ctx, name); err != nil {
		return fmt.Errorf("remove container %s: %w", name, err)
	}
	present, err = exists()
	if err != nil {
		return err
	}
	if present {
		return fmt.Errorf("container %s still exists after removal", name)
	}
	return nil
}

func (m *Manager) Available(ctx context.Context) bool {
	return exec.CommandContext(ctx, "docker", "version").Run() == nil
}

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

type ContainerInfo struct {
	ID     string
	Name   string
	Status string
	Image  string
}

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

type VolumeInfo struct {
	Name   string
	Driver string
}

func (m *Manager) RemoveVolume(ctx context.Context, name string) error {
	return exec.CommandContext(ctx, "docker", "volume", "rm", name).Run()
}

func (m *Manager) ListTokenizeImages(ctx context.Context) ([]ImageInfo, error) {
	used, err := exec.CommandContext(ctx, "docker", "ps", "-a",
		"--filter", "name=tokenize-", "--format", "{{.Image}}").Output()
	if err != nil {
		return nil, fmt.Errorf("docker ps failed: %w", err)
	}
	wanted := map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(string(used)), "\n") {
		if line != "" {
			wanted[line] = true
		}
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
		if !wanted[repo] {
			continue
		}
		images = append(images, ImageInfo{
			Repository: repo,
			ID:         parts[1],
			Size:       parts[2],
		})
	}
	return images, nil
}

type ImageInfo struct {
	Repository string
	ID         string
	Size       string
}

func (m *Manager) RemoveImage(ctx context.Context, id string) error {
	return exec.CommandContext(ctx, "docker", "rmi", "-f", id).Run()
}
