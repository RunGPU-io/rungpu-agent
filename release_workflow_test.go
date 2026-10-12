package main

import (
	"os"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type releaseWorkflow struct {
	On   map[string]interface{} `yaml:"on"`
	Jobs map[string]struct {
		Uses            string
		If              string
		Needs           []string
		ContinueOnError interface{} `yaml:"continue-on-error"`
		Steps           []struct {
			Uses            string
			Run             string
			If              string
			ContinueOnError interface{} `yaml:"continue-on-error"`
			With            map[string]string
			Env             map[string]string
		}
	}
}

func readReleaseWorkflow(t *testing.T, name string) releaseWorkflow {
	t.Helper()
	data, err := os.ReadFile(".github/workflows/" + name + ".yml")
	if err != nil {
		t.Fatal(err)
	}
	var workflow releaseWorkflow
	if err := yaml.Unmarshal(data, &workflow); err != nil {
		t.Fatal(err)
	}
	return workflow
}

func TestReleaseRequiresRuntimeVerification(t *testing.T) {
	release := readReleaseWorkflow(t, "release")
	verify, ok := release.Jobs["verify"]
	if !ok || verify.Uses != "./.github/workflows/ci.yml" || verify.If != "" || verify.ContinueOnError != nil {
		t.Fatal("release must unconditionally reuse CI from the released commit")
	}
	publish, ok := release.Jobs["release"]
	if !ok || !slices.Contains(publish.Needs, "verify") || !slices.Contains(publish.Needs, "build") || publish.If != "" {
		t.Fatal("publishing must require successful verification and all builds, without a status override")
	}
	for _, step := range release.Jobs["build"].Steps {
		if strings.HasPrefix(step.Uses, "actions/checkout@") && step.With["ref"] != "${{ github.sha }}" {
			t.Fatal("release builds must check out the same immutable SHA as verification")
		}
	}
}

func TestReusableCIVerifiesReleaseRuntimePaths(t *testing.T) {
	ci := readReleaseWorkflow(t, "ci")
	if _, ok := ci.On["workflow_call"]; !ok {
		t.Fatal("CI must support workflow_call")
	}
	for jobName, requiredCommands := range map[string][]string{
		"build":               {"go build -v ./...", "go vet ./...", "go test -v -count=1 -short ./...", "node --test gui_browser_test.mjs"},
		"runtime-smoke":       {"docker version", "bash scripts/test-clean-ollama-linux.sh", "sudo bash scripts/install-ollama-linux.sh", "go test -v -count=1 -run '^TestRuntimeSmoke_' ./internal/job -timeout 20m"},
		"macos-runtime-smoke": {"go test -v -count=1 -run '^TestRuntimeSmoke_NativeOllamaInference$' ./internal/job -timeout 15m"},
	} {
		job, ok := ci.Jobs[jobName]
		if !ok || job.If != "" || job.ContinueOnError != nil {
			t.Fatalf("%s must run for reusable CI without failure suppression", jobName)
		}
		var commands []string
		checkedOutSHA := false
		nodeReady := false
		for _, step := range job.Steps {
			if step.If != "" || step.ContinueOnError != nil {
				t.Fatalf("%s must not skip or suppress verification steps", jobName)
			}
			if strings.HasPrefix(step.Uses, "actions/checkout@") {
				checkedOutSHA = step.With["ref"] == "${{ github.sha }}"
			}
			if strings.HasPrefix(step.Uses, "actions/setup-node@") {
				nodeReady = step.With["node-version"] == "22"
			}
			if step.Run == "node --test gui_browser_test.mjs" && !nodeReady {
				t.Fatal("browser regressions require Node 22 setup before execution")
			}
			commands = append(commands, step.Run)
			if strings.Contains(step.Run, "go test") && strings.Contains(step.Run, "TestRuntimeSmoke_") {
				if step.Env["RUNGPU_RUNTIME_SMOKE"] != "1" || strings.Contains(step.Run, "-short") {
					t.Fatalf("%s must execute real runtime tests, not skip them", jobName)
				}
			}
		}
		if !checkedOutSHA {
			t.Fatalf("%s must verify the immutable caller SHA", jobName)
		}
		for _, command := range requiredCommands {
			if !slices.Contains(commands, command) {
				t.Errorf("%s missing required verification command %q", jobName, command)
			}
		}
	}
}
