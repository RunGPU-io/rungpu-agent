package job

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/tokenize/gpu-agent/internal/dockermgr"
	"github.com/tokenize/gpu-agent/internal/types"
)

// Runtime executes inference jobs for a particular accelerator backend.
type Runtime interface {
	Name() string
	Prepare(ctx context.Context, a types.JobAssignment) error
	Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error)
	Cleanup(force bool) error
}

// multiRuntime holds all available runtimes and picks the right one per job.
type multiRuntime struct {
	ollama       Runtime // text LLMs via Ollama
	dockerCustom Runtime // batch inference (video/image gen)
	workspace    Runtime // long-running interactive (ComfyUI, Jupyter)
	hasOllama    bool
	hasDocker    bool
	backend      string
}

// RuntimeOptions tunes container execution beyond the basic backend/cache args.
type RuntimeOptions struct {
	GPUDevice  string                    // "" or "all" → all GPUs; else --gpus device=<GPUDevice>
	JobTimeout time.Duration             // batch job cap; 0 → 60m default
	Policy     dockermgr.SecurityPolicy // container sandbox policy (zero value == DefaultPolicy)
}

// NewRuntime creates a runtime with default options (all GPUs, 60m timeout).
func NewRuntime(backend, cacheDir string, maxCacheGB int) (Runtime, error) {
	return NewRuntimeOpts(backend, cacheDir, maxCacheGB, RuntimeOptions{})
}

// NewRuntimeOpts creates a runtime that can handle multiple model types:
//   - Text LLMs → Ollama (auto-installed if missing)
//   - Video/image generation → Docker custom image (batch)
//   - Interactive environments (ComfyUI, Jupyter) → Docker workspace (long-running with ports)
//
// If Ollama is not installed, the agent will automatically install it.
// Docker is not auto-installed (requires more complex setup).
func NewRuntimeOpts(backend, cacheDir string, maxCacheGB int, opts RuntimeOptions) (Runtime, error) {
	hasOllama := exec.Command("ollama", "--version").Run() == nil
	hasDocker := exec.Command("docker", "version").Run() == nil
	useGPU := strings.ToLower(backend) == "cuda"

	// Auto-install Ollama if not present
	if !hasOllama {
		fmt.Println("[runtime] Ollama not found — installing automatically...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if err := installOllama(ctx); err != nil {
			fmt.Printf("[runtime] ⚠ Ollama auto-install failed: %v\n", err)
		} else {
			hasOllama = true
		}
	}

	mr := &multiRuntime{
		hasOllama: hasOllama,
		hasDocker: hasDocker,
		backend:   backend,
	}

	if hasOllama {
		mr.ollama = newOllamaRuntime(cacheDir)
	}
	if hasDocker {
		mr.dockerCustom = newCustomDockerRuntime(cacheDir, useGPU, opts.GPUDevice, opts.JobTimeout, opts.Policy)
		mr.workspace = newWorkspaceRuntime(cacheDir, useGPU, opts.GPUDevice, opts.Policy)
	}

	if !hasOllama && !hasDocker {
		fmt.Println("[runtime] WARNING: neither ollama nor docker found")
		fmt.Println("  Ollama auto-install failed — install manually: https://ollama.com/download")
		fmt.Println("  Install docker for video/image/workspace jobs: https://docs.docker.com/get-docker/")
	} else {
		if hasOllama {
			fmt.Println("[runtime] ✅ Ollama available — text LLM inference supported")
		} else {
			fmt.Println("[runtime] ⚠ Ollama not available — text LLM jobs will fail")
		}
		if hasDocker {
			fmt.Printf("[runtime] ✅ Docker available (GPU=%v) — video/image gen + workspaces supported\n", useGPU)
		} else {
			fmt.Println("[runtime] ⚠ Docker not found — video/image/workspace jobs will fail")
		}
	}

	return mr, nil
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

// selectRuntime picks the right runtime for a given job:
//
//  1. workspace=true or known workspace name → workspace runtime (long-running, ports exposed)
//  2. docker_image param or known video/image model → custom Docker runtime (batch)
//  3. Everything else → Ollama (text LLMs)
func (m *multiRuntime) selectRuntime(a types.JobAssignment) (Runtime, error) {
	// Check for workspace mode
	if isWorkspaceJob(a) {
		if m.workspace != nil {
			return m.workspace, nil
		}
		return nil, fmt.Errorf("workspace jobs require Docker — install Docker to run ComfyUI, Jupyter, etc.")
	}

	// Check for Docker-based model (video/image gen)
	if hasExplicitDockerImage(a) || isKnownDockerModel(a.ModelName) {
		if m.dockerCustom != nil {
			return m.dockerCustom, nil
		}
		return nil, fmt.Errorf("model %q requires Docker but docker is not installed", a.ModelName)
	}

	// Default: Ollama for text LLMs
	if m.ollama != nil {
		return m.ollama, nil
	}

	// No Ollama — try Docker as fallback
	if m.dockerCustom != nil {
		return m.dockerCustom, nil
	}

	return nil, fmt.Errorf("no runtime available for model %q — install ollama (text LLMs) or docker (video/image/workspace)", a.ModelName)
}

// isWorkspaceJob returns true if the job should run as a long-lived workspace.
func isWorkspaceJob(a types.JobAssignment) bool {
	// Check the dedicated Workspace field first
	if a.Workspace {
		return true
	}
	// Check Parameters for backward compat
	if a.Parameters != nil {
		if ws, ok := a.Parameters["workspace"].(bool); ok && ws {
			return true
		}
		if ws, ok := a.Parameters["workspace"].(string); ok && (ws == "true" || ws == "1") {
			return true
		}
	}
	// Check known workspace names
	lower := strings.ToLower(a.ModelName)
	for name := range knownWorkspaces {
		if lower == name {
			return true
		}
	}
	return false
}

func hasExplicitDockerImage(a types.JobAssignment) bool {
	if a.DockerImage != "" {
		return true
	}
	if a.Parameters == nil {
		return false
	}
	img, ok := a.Parameters["docker_image"].(string)
	return ok && img != ""
}

// isKnownDockerModel returns true for models that need Docker (not Ollama).
func isKnownDockerModel(name string) bool {
	lower := strings.ToLower(name)
	dockerModels := []string{
		"ltx", "ltx-video", "ltx2", "ltx2-video",
		"wan2", "wan2.1", "wan2-video",
		"stable-diffusion", "sdxl", "sd", "sd3",
		"flux", "flux.1", "flux-dev", "flux-schnell",
		"whisper", "musicgen",
	}
	for _, m := range dockerModels {
		if lower == m || strings.HasPrefix(lower, m+"-") || strings.HasPrefix(lower, m+"/") {
			return true
		}
	}
	for _, kw := range []string{"video", "image-gen", "diffusion", "img2img", "txt2img"} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func (m *multiRuntime) Prepare(ctx context.Context, a types.JobAssignment) error {
	rt, err := m.selectRuntime(a)
	if err != nil {
		return err
	}
	return rt.Prepare(ctx, a)
}

func (m *multiRuntime) Run(ctx context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	rt, err := m.selectRuntime(a)
	if err != nil {
		return nil, err
	}
	return rt.Run(ctx, a)
}

func (m *multiRuntime) Cleanup(force bool) error {
	if m.ollama != nil {
		m.ollama.Cleanup(force)
	}
	if m.dockerCustom != nil {
		m.dockerCustom.Cleanup(force)
	}
	return nil
}

// modelID normalizes a model name into a filesystem-safe cache key.
func modelID(name string) string {
	return strings.ReplaceAll(name, "/", "_")
}
