// Package dockermgr — security.go enforces container sandboxing rules.
//
// Threat model: a renter submits a malicious Docker image that tries to:
//   - Access the host filesystem outside allowed mounts
//   - Escalate privileges (--privileged, host PID/network)
//   - Run a crypto miner indefinitely
//   - Exfiltrate host data via network
//   - Mount sensitive host paths (/etc, /root, /var/run/docker.sock)
//
// Mitigations:
//  1. Image allowlist — only trusted registries (HuggingFace, our GHCR, Docker Hub official)
//  2. Container sandboxing — no --privileged, no host network, no host PID
//  3. Mount restrictions — only agent-controlled paths, never host system dirs
//  4. Resource limits — CPU, memory, timeout
//  5. Read-only root filesystem option
//  6. No new privileges (security-opt)
package dockermgr

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

// TrustedRegistries are the only registries we allow images from.
// Images from other registries are rejected unless the host opts in.
var TrustedRegistries = []string{
	"ghcr.io/tokenize/", // our pre-built model images
	"ghcr.io/ai-dock/",  // ComfyUI, A1111 community images
	"ghcr.io/clsferguson/comfyui-docker@sha256:b59dcaeece5585ac2040b76b40c1fd1b424f0c287fcaa2ebfb45af41a0b9f599", // pinned managed ComfyUI runtime
	"ghcr.io/invoke-ai/", // InvokeAI
	"registry.hf.space/", // HuggingFace Spaces
	"docker.io/library/", // Docker Hub official images
	"jupyter/",           // Jupyter official
	"pytorch/",           // PyTorch official
	"nvcr.io/nvidia/",    // NVIDIA NGC
	"atinoda/",           // text-generation-webui
}

// BlockedMountPaths are host paths that must NEVER be mounted into a container.
var BlockedMountPaths = []string{
	"/",
	"/etc",
	"/root",
	"/home",
	"/var/run/docker.sock",
	"/var/run",
	"/proc",
	"/sys",
	"/dev",
	"/boot",
	"/usr",
	"/bin",
	"/sbin",
	"/lib",
	"/tmp", // could contain agent config with API key
}

// SecurityPolicy controls what containers are allowed to do.
type SecurityPolicy struct {
	AllowAnyImage     bool     // if true, skip image allowlist (host opts in to risk)
	TrustedRegistries []string // additional trusted registries beyond defaults
	MaxMemoryGB       int      // container memory limit (0 = no limit)
	MaxCPUs           float64  // container CPU limit (0 = no limit)
	TimeoutMinutes    int      // max container runtime (0 = no limit)
	AllowHostNetwork  bool     // if true, allow --network host (dangerous)
}

// DefaultPolicy returns a secure default policy.
func DefaultPolicy() SecurityPolicy {
	return SecurityPolicy{
		AllowAnyImage:     false,
		TrustedRegistries: nil,
		MaxMemoryGB:       16,
		MaxCPUs:           2,
		TimeoutMinutes:    60,
		AllowHostNetwork:  false,
	}
}

// PolicyFromConfig maps the host's YAML security config onto a SecurityPolicy.
// The built-in TrustedRegistries always apply; config only adds to them.
func PolicyFromConfig(c types.SecurityConfig) SecurityPolicy {
	return SecurityPolicy{
		AllowAnyImage:     c.AllowAnyImage,
		TrustedRegistries: c.TrustedRegistries,
		MaxMemoryGB:       c.MaxMemoryGB,
		MaxCPUs:           c.MaxCPUs,
		AllowHostNetwork:  c.AllowHostNetwork,
	}
}

// ValidateImage checks if a Docker image is from a trusted registry.
func ValidateImage(image string, policy SecurityPolicy) error {
	if policy.AllowAnyImage {
		return nil
	}

	// Normalize: docker.io images may omit the registry prefix
	normalized := image
	if !strings.Contains(image, "/") {
		// Single-segment like "ubuntu" → "docker.io/library/ubuntu"
		normalized = "docker.io/library/" + image
	} else if !strings.Contains(strings.SplitN(image, "/", 2)[0], ".") {
		// Two-segment like "pytorch/pytorch" → "docker.io/pytorch/pytorch"
		normalized = "docker.io/" + image
	}

	allTrusted := append(TrustedRegistries, policy.TrustedRegistries...)
	for _, prefix := range allTrusted {
		if strings.HasPrefix(normalized, prefix) || strings.HasPrefix(image, prefix) {
			return nil
		}
	}

	return fmt.Errorf(
		"image %q is not from a trusted registry. Allowed: %s. "+
			"Host can set allow_any_image: true in config to override",
		image, strings.Join(allTrusted, ", "),
	)
}

// ValidateMounts checks that no mount path accesses sensitive host directories.
func ValidateMounts(mounts []string) error {
	for _, mount := range mounts {
		parts := strings.SplitN(mount, ":", 2)
		if len(parts) < 2 {
			continue
		}
		hostPath := filepath.Clean(parts[0])

		for _, blocked := range BlockedMountPaths {
			if hostPath == blocked || strings.HasPrefix(hostPath, blocked+"/") {
				// Exception: the agent's own directories live under ~/.tokenize
				// (which may itself sit under a blocked root like /home or /root).
				// Match a real ".tokenize" path *segment* — not a substring — so
				// paths like "/etc/cache" or "/root/staging" stay blocked.
				if isAgentControlledPath(hostPath) {
					continue
				}
				return fmt.Errorf("mount path %q is blocked for security (accesses %s)", hostPath, blocked)
			}
		}
	}
	return nil
}

// isAgentControlledPath reports whether hostPath sits inside the agent's own
// ~/.tokenize tree, by matching a ".tokenize" path *segment* (not a substring).
func isAgentControlledPath(hostPath string) bool {
	for _, seg := range strings.Split(filepath.ToSlash(hostPath), "/") {
		if seg == ".tokenize" {
			return true
		}
	}
	return false
}

// SandboxArgs returns Docker CLI arguments that sandbox the container.
// These are always applied — the renter cannot override them.
func SandboxArgs(policy SecurityPolicy) []string {
	args := []string{
		"--security-opt", "no-new-privileges", // prevent privilege escalation
		"--pids-limit", "4096", // prevent fork bombs
	}

	// Only opt into host networking when explicitly allowed; otherwise the
	// container uses its own network namespace (Docker's default bridge).
	if policy.AllowHostNetwork {
		args = append(args, "--network", "host")
	}

	if policy.MaxMemoryGB > 0 {
		args = append(args, "--memory", fmt.Sprintf("%dg", policy.MaxMemoryGB))
	}

	if policy.MaxCPUs > 0 {
		args = append(args, "--cpus", fmt.Sprintf("%.1f", policy.MaxCPUs))
	}

	return args
}
