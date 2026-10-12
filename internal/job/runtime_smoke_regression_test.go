package job

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestDockerSmokeInferenceArgs(t *testing.T) {
	if !regexp.MustCompile(`^ghcr\.io/ggml-org/llama\.cpp:light@sha256:[a-f0-9]{64}$`).MatchString(runtimeSmokeDockerImage) {
		t.Fatal("CPU smoke image must be pinned to an immutable digest")
	}
	args := dockerSmokeInferenceArgs("smoke", "/models with spaces/model.gguf")
	imageIndex := slices.Index(args, runtimeSmokeDockerImage)
	entrypointIndex := slices.Index(args, "--entrypoint")
	if entrypointIndex < 0 || imageIndex != entrypointIndex+2 || args[entrypointIndex+1] != "/app/llama-completion" {
		t.Fatalf("must override the interactive CLI before the image argument: %q", args)
	}
	for flag, value := range map[string]string{
		"--name": "smoke", "--volume": "/models with spaces/model.gguf:/models/model.gguf:ro",
		"--model": "/models/model.gguf", "--predict": "8", "--temp": "0",
	} {
		i := slices.Index(args, flag)
		if i < 0 || i+1 >= len(args) || args[i+1] != value {
			t.Errorf("%s must have value %q: %q", flag, value, args)
		}
	}
	for _, flag := range []string{"--rm", "--no-display-prompt", "--single-turn", "--simple-io"} {
		if !slices.Contains(args, flag) {
			t.Errorf("missing %s: %q", flag, args)
		}
	}
}

func TestDockerSmokeInferenceOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake Docker executable uses a POSIX shell")
	}
	dir := t.TempDir()
	script := "#!/bin/sh\nprintf '%s' \"$SMOKE_STDOUT\"\nprintf '%s' \"$SMOKE_STDERR\" >&2\nexit \"$SMOKE_EXIT\"\n"
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	for _, tc := range []struct {
		name, stdout, stderr, exit, wantError string
	}{
		{"answer", "4\n", "CPU backend loaded", "0", ""},
		{"sentence", "The answer is 4.\n", "", "0", ""},
		{"logs are not inference", "", "build 1234, 4 threads", "0", "unexpected Docker inference response"},
		{"wrong answer", "44\n", "", "0", "unexpected Docker inference response"},
		{"backend failure", "", "no backends are loaded", "1", "no backends are loaded"},
		{"failure with answer", "4\n", "server exited", "1", "run Docker llama.cpp inference"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("SMOKE_STDOUT", tc.stdout)
			t.Setenv("SMOKE_STDERR", tc.stderr)
			t.Setenv("SMOKE_EXIT", tc.exit)
			err := runDockerSmokeInference(context.Background(), "smoke", "/model.gguf")
			if tc.wantError == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("got %v, want error containing %q", err, tc.wantError)
			}
		})
	}
}
