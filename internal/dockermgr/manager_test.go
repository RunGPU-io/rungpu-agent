package dockermgr

import (
	"context"
	"strings"
	"testing"
)

// argValue returns the argument immediately following the first occurrence of
// flag in args, or "" if the flag is absent / has no value.
func argValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func TestBuildRunArgsGPUScoping(t *testing.T) {
	m := New()

	// No GPU → no --gpus flag at all.
	if got := m.buildRunArgs(RunOptions{Image: "ubuntu", Name: "c", UseGPU: false}); hasFlag(got, "--gpus") {
		t.Errorf("UseGPU=false should not add --gpus, got %v", got)
	}

	// GPU with no device / "all" → --gpus all.
	for _, dev := range []string{"", "all", "ALL", "  "} {
		args := m.buildRunArgs(RunOptions{Image: "ubuntu", Name: "c", Network: "none", UseGPU: true, GPUDevice: dev})
		if v := argValue(args, "--gpus"); v != "all" {
			t.Errorf("GPUDevice=%q → --gpus %q, want all", dev, v)
		}
	}

	// GPU with a specific device → --gpus device=<id>.
	args := m.buildRunArgs(RunOptions{Image: "ubuntu", Name: "c", Network: "none", UseGPU: true, GPUDevice: "1"})
	if v := argValue(args, "--gpus"); v != "device=1" {
		t.Errorf("GPUDevice=1 → --gpus %q, want device=1", v)
	}
}

func TestBuildRunArgsComposition(t *testing.T) {
	m := New()
	args := m.buildRunArgs(RunOptions{
		Image:      "ghcr.io/tokenize/x:latest",
		Name:       "job-c",
		Network:    "none",
		Ports:      []string{"127.0.0.1:8188:8188"},
		Mounts:     []string{"/home/u/.tokenize/cache:/cache"},
		Volumes:    []string{"tokenize-vol:/data"},
		Env:        map[string]string{"FOO": "bar"},
		ShmSize:    "8g",
		Entrypoint: "python",
		Command:    []string{"serve"},
	})

	joined := strings.Join(args, " ")
	for _, want := range []string{
		"run -d --name job-c",
		"--network none",
		"--security-opt no-new-privileges",
		"--shm-size 8g",
		"-p 127.0.0.1:8188:8188",
		"-v /home/u/.tokenize/cache:/cache",
		"-v tokenize-vol:/data",
		"-e FOO=bar",
		"--entrypoint python",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args missing %q\n got: %s", want, joined)
		}
	}

	// Image must precede its command, and the command comes last.
	if args[len(args)-1] != "serve" {
		t.Errorf("command should be last arg, got %q", args[len(args)-1])
	}
	imgIdx, cmdIdx := -1, -1
	for i, a := range args {
		if a == "ghcr.io/tokenize/x:latest" {
			imgIdx = i
		}
		if a == "serve" {
			cmdIdx = i
		}
	}
	if imgIdx < 0 || cmdIdx < 0 || imgIdx > cmdIdx {
		t.Errorf("image (%d) must come before command (%d)", imgIdx, cmdIdx)
	}
}

func TestRunRejectsUnspecifiedNetwork(t *testing.T) {
	_, err := New().Run(context.Background(), RunOptions{Image: "ubuntu", Name: "job-c"})
	if err == nil || !strings.Contains(err.Error(), "network must be none or bridge") {
		t.Fatalf("expected unspecified network to fail closed, got %v", err)
	}
}
