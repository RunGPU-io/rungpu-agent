package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestPowerShellDockerInferenceScript(t *testing.T) {
	shell, err := exec.LookPath("pwsh")
	if err != nil && runtime.GOOS == "windows" {
		shell, err = exec.LookPath("powershell")
	}
	if err != nil {
		t.Skip("PowerShell is not installed")
	}
	script, err := filepath.Abs("scripts/test-docker-inference.ps1")
	if err != nil {
		t.Fatal(err)
	}
	helperDir := t.TempDir()
	source := filepath.Join(helperDir, "command.go")
	const helper = `package main
	import ("fmt"; "os"; "path/filepath"; "strings")
	func main() {
	  mode := os.Getenv("SMOKE_MODE")
	  if strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") == "docker" {
	    if mode == "docker-unavailable" { fmt.Fprintln(os.Stderr, "daemon unavailable"); os.Exit(1) }
	    if os.Args[1] == "info" {
	      if mode == "windows-containers" { fmt.Println("windows") } else { fmt.Println("linux") }
	    } else { fmt.Println("Docker fixture") }
	    return
	  }
	  if os.Args[1] == "version" { fmt.Println("go version go1.22 fixture"); return }
	  wd, _ := os.Getwd()
	  call := append([]string{wd, os.Getenv("RUNGPU_RUNTIME_SMOKE"), os.Getenv("GOFLAGS")}, os.Args[1:]...)
	  if err := os.WriteFile(os.Getenv("SMOKE_CALL_LOG"), []byte(strings.Join(call, "\n")+"\n"), 0600); err != nil { panic(err) }
	  switch mode {
	  case "pass", "stderr-pass":
	    if mode == "stderr-pass" { fmt.Fprintln(os.Stderr, "backend diagnostics, not an error") }
	    fmt.Println("--- PASS: TestRuntimeSmoke_DockerInference (0.01s)\nPASS")
	  case "test-failure":
	    fmt.Println("--- PASS: TestRuntimeSmoke_DockerInference (0.01s)\nFAIL"); os.Exit(1)
	  case "skip": fmt.Println("--- SKIP: TestRuntimeSmoke_DockerInference (0.00s)\nPASS")
	  case "no-tests": fmt.Println("testing: warning: no tests to run\nPASS")
	  default: fmt.Fprintln(os.Stderr, "unexpected test invocation"); os.Exit(1)
	  }
	}`
	if err := os.WriteFile(source, []byte(helper), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(helperDir, "command")
	build := exec.Command("go", "build", "-o", binary, source)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build command fixture: %v\n%s", err, output)
	}
	binaryBytes, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"pass", "stderr-pass", "test-failure", "skip", "no-tests", "docker-unavailable", "windows-containers", "help", "bad-argument"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			bin := filepath.Join(dir, "fake bin")
			if err := os.Mkdir(bin, 0700); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"go", "docker"} {
				if runtime.GOOS == "windows" {
					name += ".exe"
				}
				if err := os.WriteFile(filepath.Join(bin, name), binaryBytes, 0700); err != nil {
					t.Fatal(err)
				}
			}
			callLog := filepath.Join(dir, "go-call")
			t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
			t.Setenv("TMPDIR", dir)
			t.Setenv("TEMP", dir)
			t.Setenv("TMP", dir)
			t.Setenv("SMOKE_MODE", mode)
			t.Setenv("SMOKE_CALL_LOG", callLog)
			t.Setenv("GOFLAGS", "-short")
			t.Setenv("RUNGPU_RUNTIME_SMOKE", "0")
			args := []string{"-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned", "-File", script}
			if mode == "help" {
				args = append(args, "-Help")
			} else if mode == "bad-argument" {
				args = append(args, "-UnknownOption")
			}
			cmd := exec.Command(shell, args...)
			cmd.Dir = dir
			output, runErr := cmd.CombinedOutput()
			wantSuccess := mode == "pass" || mode == "stderr-pass" || mode == "help"
			if (runErr == nil) != wantSuccess {
				t.Fatalf("error=%v, output=%s", runErr, output)
			}
			call, readErr := os.ReadFile(callLog)
			if mode == "help" || mode == "bad-argument" || mode == "docker-unavailable" || mode == "windows-containers" {
				if !os.IsNotExist(readErr) {
					t.Fatalf("preflight invoked inference: %s, %v", call, readErr)
				}
				return
			}
			want := strings.Join([]string{
				filepath.Dir(filepath.Dir(script)), "1", "", "test", "-v",
				"-count=1", "-short=false", "-run", "^TestRuntimeSmoke_DockerInference$",
				"./internal/job", "-timeout", "20m", "",
			}, "\n")
			if readErr != nil || string(call) != want {
				t.Fatalf("invocation=%q, want=%q, error=%v", call, want, readErr)
			}
			if !strings.Contains(string(output), "rungpu-docker-inference-") {
				t.Fatal("runner did not print the diagnostic log location")
			}
		})
	}
}
func TestLocalDockerInferenceScript(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Bash runner is for macOS/Linux/WSL")
	}
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Fatal(err)
	}
	script, err := filepath.Abs("scripts/test-docker-inference.sh")
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"pass", "test-failure", "skip", "no-tests", "docker-unavailable", "help", "bad-argument"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			bin := filepath.Join(dir, "fake bin")
			if err := os.Mkdir(bin, 0700); err != nil {
				t.Fatal(err)
			}
			docker := "#!/bin/sh\nif [ \"$SMOKE_MODE\" = docker-unavailable ]; then echo 'daemon unavailable' >&2; exit 1; fi\necho 'Docker fixture'\n"
			goCommand := `#!/bin/sh
if [ "$1" = version ]; then echo 'go version go1.22 fixture'; exit 0; fi
printf '%s\n' "$PWD" "$RUNGPU_RUNTIME_SMOKE" "$GOFLAGS" "$@" > "$SMOKE_CALL_LOG"
case "$SMOKE_MODE" in
  pass) printf '%s\n' '--- PASS: TestRuntimeSmoke_DockerInference (0.01s)' 'PASS' ;;
  test-failure) printf '%s\n' '--- PASS: TestRuntimeSmoke_DockerInference (0.01s)' 'FAIL'; exit 1 ;;
  skip) printf '%s\n' '--- SKIP: TestRuntimeSmoke_DockerInference (0.00s)' 'PASS' ;;
  no-tests) printf '%s\n' 'testing: warning: no tests to run' 'PASS' ;;
  *) echo 'unexpected test invocation' >&2; exit 1 ;;
esac
`
			for name, content := range map[string]string{"docker": docker, "go": goCommand} {
				if err := os.WriteFile(filepath.Join(bin, name), []byte(content), 0700); err != nil {
					t.Fatal(err)
				}
			}
			callLog := filepath.Join(dir, "go-call")
			t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
			t.Setenv("TMPDIR", dir)
			t.Setenv("SMOKE_MODE", mode)
			t.Setenv("SMOKE_CALL_LOG", callLog)
			t.Setenv("GOFLAGS", "-short")
			t.Setenv("RUNGPU_RUNTIME_SMOKE", "0")
			args := []string{script}
			if mode == "help" {
				args = append(args, "--help")
			} else if mode == "bad-argument" {
				args = append(args, "--short")
			}
			cmd := exec.Command(bash, args...)
			cmd.Dir = dir
			output, runErr := cmd.CombinedOutput()
			wantSuccess := mode == "pass" || mode == "help"
			if (runErr == nil) != wantSuccess {
				t.Fatalf("error=%v, output=%s", runErr, output)
			}
			call, readErr := os.ReadFile(callLog)
			if mode == "help" || mode == "bad-argument" || mode == "docker-unavailable" {
				if !os.IsNotExist(readErr) {
					t.Fatalf("preflight invoked the inference test: %s, %v", call, readErr)
				}
				return
			}
			want := strings.Join([]string{
				filepath.Dir(filepath.Dir(script)), "1", "", "test", "-v",
				"-count=1", "-short=false", "-run", "^TestRuntimeSmoke_DockerInference$",
				"./internal/job", "-timeout", "20m", "",
			}, "\n")
			if readErr != nil || string(call) != want {
				t.Fatalf("test invocation=%q, want=%q, error=%v", call, want, readErr)
			}
			logs, err := filepath.Glob(filepath.Join(dir, "rungpu-docker-inference.*"))
			if err != nil || len(logs) != 1 {
				t.Fatalf("expected one diagnostic log: %v, %v", logs, err)
			}
			if !strings.Contains(string(output), logs[0]) {
				t.Fatal("runner did not report the diagnostic log path")
			}
		})
	}
}
