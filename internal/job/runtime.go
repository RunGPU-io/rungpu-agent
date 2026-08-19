package job

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

type Runtime interface {
	Name() string
	Prepare(ctx context.Context, assignment types.JobAssignment) error
	Run(ctx context.Context, assignment types.JobAssignment) (map[string]interface{}, error)
	Cleanup(force bool) error
}

type multiRuntime struct {
	ollama       Runtime
	dockerCustom Runtime
	comfyUIBatch Runtime
	workspace    Runtime
	hasOllama    bool
	hasDocker    bool
}

type RuntimeOptions struct {
	GPUDevice  string
	JobTimeout time.Duration
	Policy     dockermgr.SecurityPolicy
}

func NewRuntime(backend, cacheDir string, maxCacheGB int) (Runtime, error) {
	return NewRuntimeOpts(backend, cacheDir, maxCacheGB, RuntimeOptions{})
}

func NewRuntimeOpts(backend, cacheDir string, maxCacheGB int, options RuntimeOptions) (Runtime, error) {
	hasOllama := runtimeCommandAvailable("ollama", "--version")
	hasDocker := runtimeCommandAvailable("docker", "version")
	useGPU := strings.EqualFold(backend, "cuda")
	runtime := &multiRuntime{hasOllama: hasOllama, hasDocker: hasDocker}
	if hasOllama {
		runtime.ollama = newOllamaRuntime(cacheDir)
	}
	if hasDocker {
		runtime.dockerCustom = newCustomDockerRuntime(cacheDir, useGPU, options.GPUDevice, options.JobTimeout, options.Policy)
		runtime.comfyUIBatch = newComfyUIBatchRuntime(cacheDir, useGPU, options.GPUDevice, options.Policy)
		runtime.workspace = newWorkspaceRuntime(cacheDir, useGPU, options.GPUDevice, options.Policy)
	}
	if !hasOllama && !hasDocker {
		fmt.Println("[runtime] No supported runtime detected. Run: rungpu-agent setup")
	}
	return runtime, nil
}

func runtimeCommandAvailable(name string, args ...string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).Run() == nil
}

func (m *multiRuntime) Name() string {
	parts := []string{}
	if m.hasOllama {
		parts = append(parts, "ollama")
	}
	if m.hasDocker {
		parts = append(parts, "docker")
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, "+")
}

func (m *multiRuntime) selectRuntime(assignment types.JobAssignment) (Runtime, error) {
	switch assignment.Runtime {
	case "comfyui-batch":
		if m.comfyUIBatch != nil {
			return m.comfyUIBatch, nil
		}
		return nil, fmt.Errorf("requested runtime is unavailable: comfyui-batch")
	case "workspace":
		if m.workspace != nil {
			return m.workspace, nil
		}
		return nil, fmt.Errorf("requested runtime is unavailable: workspace")
	case "docker-custom":
		if m.dockerCustom != nil {
			return m.dockerCustom, nil
		}
		return nil, fmt.Errorf("requested runtime is unavailable: docker-custom")
	case "ollama":
		if m.ollama != nil {
			return m.ollama, nil
		}
		return nil, fmt.Errorf("requested runtime is unavailable: ollama")
	case "":
		return nil, fmt.Errorf("execution runtime is required")
	default:
		return nil, fmt.Errorf("unsupported execution runtime %q", assignment.Runtime)
	}
}

func (m *multiRuntime) Prepare(ctx context.Context, assignment types.JobAssignment) error {
	runtime, err := m.selectRuntime(assignment)
	if err != nil {
		return err
	}
	return runtime.Prepare(ctx, assignment)
}

func (m *multiRuntime) Run(ctx context.Context, assignment types.JobAssignment) (map[string]interface{}, error) {
	runtime, err := m.selectRuntime(assignment)
	if err != nil {
		return nil, err
	}
	return runtime.Run(ctx, assignment)
}

func (m *multiRuntime) Cleanup(force bool) error {
	if m.ollama != nil {
		m.ollama.Cleanup(force)
	}
	if m.dockerCustom != nil {
		m.dockerCustom.Cleanup(force)
	}
	if m.comfyUIBatch != nil {
		m.comfyUIBatch.Cleanup(force)
	}
	if m.workspace != nil {
		m.workspace.Cleanup(force)
	}
	return nil
}

func modelID(name string) string {
	return strings.ReplaceAll(name, "/", "_")
}
