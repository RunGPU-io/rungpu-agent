package dockermgr

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

var TrustedRegistries = []string{
	"docker.io/library/",
	"jupyter/",
	"pytorch/",
	"nvcr.io/nvidia/",
}

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
	"/tmp",
}

type SecurityPolicy struct {
	AllowAnyImage     bool
	TrustedRegistries []string
	MaxMemoryGB       int
	MaxCPUs           float64
	TimeoutMinutes    int
	AllowHostNetwork  bool
	CoordinatorImage  string
}

func WithAssignedImage(policy SecurityPolicy, image string) SecurityPolicy {
	policy.CoordinatorImage = strings.TrimSpace(image)
	return policy
}

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

func PolicyFromConfig(c types.SecurityConfig) SecurityPolicy {
	return SecurityPolicy{
		AllowAnyImage:     c.AllowAnyImage,
		TrustedRegistries: c.TrustedRegistries,
		MaxMemoryGB:       c.MaxMemoryGB,
		MaxCPUs:           c.MaxCPUs,
		AllowHostNetwork:  c.AllowHostNetwork,
	}
}

func ValidateImage(image string, policy SecurityPolicy) error {
	if policy.AllowAnyImage {
		return nil
	}
	if policy.CoordinatorImage != "" && (imageMatchesTrusted(image, policy.CoordinatorImage) || imageMatchesTrusted(normalizeImage(image), normalizeImage(policy.CoordinatorImage))) {
		return nil
	}

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

func imageMatchesTrusted(image, trusted string) bool {
	if trusted == "" {
		return false
	}
	if strings.HasSuffix(trusted, "/") || strings.Contains(trusted, "@sha256:") {
		return strings.HasPrefix(image, trusted)
	}
	return image == trusted || strings.HasPrefix(image, trusted+":") || strings.HasPrefix(image, trusted+"@")
}

func ValidateMounts(mounts []string) error {
	for _, mount := range mounts {
		parts := strings.SplitN(mount, ":", 2)
		if len(parts) < 2 {
			continue
		}
		hostPath := filepath.Clean(parts[0])

		for _, blocked := range BlockedMountPaths {
			if hostPath == blocked || strings.HasPrefix(hostPath, blocked+"/") {

				if isAgentControlledPath(hostPath) {
					continue
				}
				return fmt.Errorf("mount path %q is blocked for security (accesses %s)", hostPath, blocked)
			}
		}
	}
	return nil
}

func isAgentControlledPath(hostPath string) bool {
	for _, seg := range strings.Split(filepath.ToSlash(hostPath), "/") {
		if seg == ".tokenize" {
			return true
		}
	}
	return false
}

func SandboxArgs(policy SecurityPolicy) []string {
	args := []string{
		"--security-opt", "no-new-privileges",
		"--pids-limit", "4096",
	}

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
