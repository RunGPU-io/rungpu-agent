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
//  1. Image allowlist — official public bases, host extras, or the one image this job assigned
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

// TrustedRegistries are generic public bases only. Managed product images are
// not listed here — the coordinator assigns those per job (CoordinatorImage).
var TrustedRegistries = []string{
	"docker.io/library/",
	"jupyter/",
	"pytorch/",
	"nvcr.io/nvidia/",
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
	AllowHostNetwork  bool   // if true, allow --network host (dangerous)
	CoordinatorImage  string // the single image this authenticated job assigned
}

// WithAssignedImage allows one coordinator-assigned image for this job only.
func WithAssignedImage(policy SecurityPolicy, image string) SecurityPolicy {
	policy.CoordinatorImage = strings.TrimSpace(image)
	return policy
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
	if policy.CoordinatorImage != "" && (imageMatchesTrusted(image, policy.CoordinatorImage) || imageMatchesTrusted(normalizeImage(image), normalizeImage(policy.CoordinatorImage))) {
		return nil
	}

	// Normalize: docker.io images may omit the registry prefix
	normalized := normalizeImage(image)

	allTrusted := append(TrustedRegistries, policy.TrustedRegistries...)
	for _, prefix := range allTrusted {
		if imageMatchesTrusted(normalized, prefix) || imageMatchesTrusted(image, prefix) {
			return nil
		}
	}

	return fmt.Errorf(
		"image %q is not from a trusted registry. Allowed: %s. "+
			"Host can set allow_any_image: true in config to override",
		image, strings.Join(allTrusted, ", "),
	)
}

func normalizeImage(image string) string {
	if !strings.Contains(image, "/") {
		return "docker.io/library/" + image
	}
	if !strings.Contains(strings.SplitN(image, "/", 2)[0], ".") {
		return "docker.io/" + image
	}
	return image
}

// imageMatchesTrusted reports whether image is covered by a trusted entry.
// Registry prefixes end with "/". Digest pins contain "@sha256:". Bare image
// names must match exactly or be followed by a tag (":") or digest ("@").
func imageMatchesTrusted(image, trusted string) bool {
	if trusted == "" {
		return false
	}
	if strings.HasSuffix(trusted, "/") || strings.Contains(trusted, "@sha256:") {
		return strings.HasPrefix(image, trusted)
	}
	return image == trusted || strings.HasPrefix(image, trusted+":") || strings.HasPrefix(image, trusted+"@")
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
