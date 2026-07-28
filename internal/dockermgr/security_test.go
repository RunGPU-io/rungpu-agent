package dockermgr

import (
	"testing"

	"github.com/RunGPU-io/gpu-agent/internal/types"
)

func TestPolicyFromConfig(t *testing.T) {
	p := PolicyFromConfig(types.SecurityConfig{
		AllowAnyImage:     true,
		TrustedRegistries: []string{"mycorp.example.com/"},
		MaxMemoryGB:       16,
		MaxCPUs:           4,
		AllowHostNetwork:  true,
	})
	if !p.AllowAnyImage || !p.AllowHostNetwork || p.MaxMemoryGB != 16 || p.MaxCPUs != 4 {
		t.Fatalf("policy not mapped from config: %+v", p)
	}
	if len(p.TrustedRegistries) != 1 || p.TrustedRegistries[0] != "mycorp.example.com/" {
		t.Fatalf("trusted registries not mapped: %+v", p.TrustedRegistries)
	}

	// A config-added registry is honored by ValidateImage alongside the defaults.
	if err := ValidateImage("mycorp.example.com/model:v1", p); err != nil {
		t.Errorf("config trusted registry should pass: %v", err)
	}
	// Built-in trusted registries still work with a config-derived policy.
	if err := ValidateImage("ghcr.io/tokenize/ltx-video:latest", PolicyFromConfig(types.SecurityConfig{})); err != nil {
		t.Errorf("built-in trusted registry should still pass under config policy: %v", err)
	}
}

func TestValidateImageTrusted(t *testing.T) {
	policy := DefaultPolicy()

	// Trusted registries should pass
	trusted := []string{
		"ghcr.io/tokenize/ltx-video:latest",
		"ghcr.io/ai-dock/comfyui:latest",
		"ghcr.io/invoke-ai/invokeai:latest",
		"registry.hf.space/Lightricks-LTX-Video",
		"jupyter/pytorch-notebook:latest",
		"pytorch/pytorch:2.0-cuda11.8",
		"nvcr.io/nvidia/pytorch:latest",
	}
	for _, img := range trusted {
		if err := ValidateImage(img, policy); err != nil {
			t.Errorf("trusted image %q rejected: %v", img, err)
		}
	}
}

func TestValidateImageUntrusted(t *testing.T) {
	policy := DefaultPolicy()

	// Untrusted registries should be rejected
	untrusted := []string{
		"evil.io/cryptominer:latest",
		"randomuser/malicious-image:v1",
		"quay.io/someone/something",
		"gcr.io/attacker-project/backdoor",
	}
	for _, img := range untrusted {
		if err := ValidateImage(img, policy); err == nil {
			t.Errorf("untrusted image %q should be rejected", img)
		}
	}
}

func TestValidateImageAllowAny(t *testing.T) {
	policy := SecurityPolicy{AllowAnyImage: true}

	// With AllowAnyImage, everything passes
	if err := ValidateImage("evil.io/anything:latest", policy); err != nil {
		t.Errorf("AllowAnyImage should allow everything: %v", err)
	}
}

func TestValidateImageCustomTrusted(t *testing.T) {
	policy := SecurityPolicy{
		TrustedRegistries: []string{"mycompany.azurecr.io/"},
	}

	if err := ValidateImage("mycompany.azurecr.io/mymodel:v1", policy); err != nil {
		t.Errorf("custom trusted registry should pass: %v", err)
	}

	if err := ValidateImage("other.azurecr.io/model:v1", policy); err == nil {
		t.Error("non-custom registry should still be rejected")
	}
}

func TestValidateImageDockerHubShorthand(t *testing.T) {
	policy := DefaultPolicy()

	// "ubuntu" → "docker.io/library/ubuntu" → trusted
	if err := ValidateImage("ubuntu", policy); err != nil {
		t.Errorf("Docker Hub official image should be trusted: %v", err)
	}

	// "pytorch/pytorch" → "docker.io/pytorch/pytorch" → trusted
	if err := ValidateImage("pytorch/pytorch:2.0", policy); err != nil {
		t.Errorf("pytorch official should be trusted: %v", err)
	}
}

func TestValidateMountsBlocked(t *testing.T) {
	blocked := []string{
		"/etc:/etc",
		"/root:/root",
		"/var/run/docker.sock:/var/run/docker.sock",
		"/:/host",
		"/proc:/proc",
		"/home/user:/data",
	}
	for _, mount := range blocked {
		if err := ValidateMounts([]string{mount}); err == nil {
			t.Errorf("mount %q should be blocked", mount)
		}
	}
}

func TestValidateMountsAllowed(t *testing.T) {
	allowed := []string{
		"/home/user/.tokenize/cache:/cache",
		"/home/user/.tokenize/staging/job1:/custom:ro",
		"/home/user/.tokenize/output/job1:/output",
		"/home/user/.tokenize/workspaces/job1:/workspace",
	}
	for _, mount := range allowed {
		if err := ValidateMounts([]string{mount}); err != nil {
			t.Errorf("mount %q should be allowed: %v", mount, err)
		}
	}
}

func TestValidateMountsSubstringBypass(t *testing.T) {
	// Paths that merely CONTAIN "cache"/"staging"/etc. as a substring but are
	// NOT under the agent's ~/.tokenize tree must stay blocked.
	blocked := []string{
		"/etc/cache:/cache",
		"/root/staging:/data",
		"/usr/output:/out",
		"/proc/workspaces:/ws",
		"/home/tokenize/x:/x", // "tokenize" segment without the leading dot
	}
	for _, mount := range blocked {
		if err := ValidateMounts([]string{mount}); err == nil {
			t.Errorf("mount %q should be blocked (substring bypass)", mount)
		}
	}
}

func TestSandboxArgsHostNetwork(t *testing.T) {
	// Default policy must NOT use host networking.
	for i, a := range SandboxArgs(DefaultPolicy()) {
		if a == "--network" {
			t.Fatalf("default policy must not set --network (got %v)", SandboxArgs(DefaultPolicy())[i:])
		}
	}

	// Explicitly allowed → --network host present.
	args := SandboxArgs(SecurityPolicy{AllowHostNetwork: true})
	found := false
	for i, a := range args {
		if a == "--network" && i+1 < len(args) && args[i+1] == "host" {
			found = true
		}
	}
	if !found {
		t.Errorf("AllowHostNetwork should add --network host, got %v", args)
	}
}

func TestSandboxArgs(t *testing.T) {
	policy := DefaultPolicy()
	args := SandboxArgs(policy)

	// Must contain no-new-privileges
	found := false
	for i, a := range args {
		if a == "--security-opt" && i+1 < len(args) && args[i+1] == "no-new-privileges" {
			found = true
		}
	}
	if !found {
		t.Error("sandbox args must include --security-opt no-new-privileges")
	}

	// Must contain pids-limit
	foundPids := false
	for i, a := range args {
		if a == "--pids-limit" && i+1 < len(args) {
			foundPids = true
		}
	}
	if !foundPids {
		t.Error("sandbox args must include --pids-limit")
	}
}

func TestSandboxArgsWithLimits(t *testing.T) {
	policy := SecurityPolicy{
		MaxMemoryGB: 16,
		MaxCPUs:     4.0,
	}
	args := SandboxArgs(policy)

	hasMemory := false
	hasCPU := false
	for i, a := range args {
		if a == "--memory" && i+1 < len(args) && args[i+1] == "16g" {
			hasMemory = true
		}
		if a == "--cpus" && i+1 < len(args) && args[i+1] == "4.0" {
			hasCPU = true
		}
	}
	if !hasMemory {
		t.Error("should include --memory 16g")
	}
	if !hasCPU {
		t.Error("should include --cpus 4.0")
	}
}
