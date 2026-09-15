package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestVersionHasBuildFallback(t *testing.T) {
	if strings.TrimSpace(version) == "" {
		t.Fatal("version must have a non-empty fallback for local builds")
	}
}

func TestBatchEnrollmentSendsMachineIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/fleet/batch/enroll" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["batch_token"] != "batch-secret" || body["installation_id"] != "install-123" || body["installation_secret"] != "install-secret" || body["hostname"] != "rack-a1" {
			t.Fatalf("unexpected identity payload: %#v", body)
		}
		if body["operating_system"] != runtime.GOOS || body["architecture"] != runtime.GOARCH {
			t.Fatalf("unexpected platform payload: %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"machine_id":"machine-1","agent_key":"agent-key","price_per_minute":0}`))
	}))
	defer server.Close()

	result, err := enrollMachineBatch(server.URL, "batch-secret", "install-123", "install-secret", "rack-a1")
	if err != nil {
		t.Fatalf("enrollMachineBatch: %v", err)
	}
	if result.MachineID != "machine-1" || result.AgentKey != "agent-key" {
		t.Fatalf("unexpected response: %#v", result)
	}
}

func TestReadBatchTokenFileRequiresPrivatePermissions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "batch-token")
	if err := os.WriteFile(path, []byte("batch-secret\n"), 0600); err != nil {
		t.Fatal(err)
	}
	contents, err := readBatchTokenFile(path)
	if err != nil || strings.TrimSpace(string(contents)) != "batch-secret" {
		t.Fatalf("read private token: contents=%q err=%v", contents, err)
	}
	if runtime.GOOS != "windows" {
		if err := os.Chmod(path, 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := readBatchTokenFile(path); err == nil {
			t.Fatal("expected readable-by-others token file to be rejected")
		}
	}
}

func TestStartCommand(t *testing.T) {
	tests := []struct {
		name       string
		goos       string
		executable string
		want       string
	}{
		{
			name:       "windows",
			goos:       "windows",
			executable: "rungpu-agent-windows-amd64.exe",
			want:       `.\rungpu-agent-windows-amd64.exe start`,
		},
		{
			name:       "macOS",
			goos:       "darwin",
			executable: "rungpu-agent-darwin-arm64",
			want:       "./rungpu-agent-darwin-arm64 start",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := startCommand(test.goos, test.executable); got != test.want {
				t.Fatalf("startCommand() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestSetupCommands(t *testing.T) {
	commands := setupCommandPlan("windows")
	if len(commands) != 2 || commands[0][0] != "winget" || commands[1][0] != "winget" {
		t.Fatalf("unexpected Windows setup commands: %#v", commands)
	}
	if got := setupCommand("windows", "rungpu-agent-windows-amd64.exe"); got != `.\rungpu-agent-windows-amd64.exe setup` {
		t.Fatalf("setupCommand() = %q", got)
	}

	if _, err := setupCommands("linux"); err == nil {
		t.Fatal("Linux setup should provide distribution-specific instructions")
	}
}

func TestSetupCommandMacOS(t *testing.T) {
	if got := setupCommand("darwin", "rungpu-agent-darwin-arm64"); got != "./rungpu-agent-darwin-arm64 setup" {
		t.Fatalf("setupCommand() = %q", got)
	}
}

func TestRuntimeReadiness(t *testing.T) {
	tests := []struct {
		name         string
		backend      string
		capabilities []string
		wantRuntime  string
		wantReady    bool
	}{
		{name: "CUDA with Docker", backend: "cuda", capabilities: []string{"docker", "workspace"}, wantRuntime: "docker", wantReady: true},
		{name: "CUDA without Docker", backend: "cuda", capabilities: []string{"ollama"}, wantRuntime: "docker", wantReady: false},
		{name: "Metal with Ollama", backend: "metal", capabilities: []string{"ollama"}, wantRuntime: "ollama", wantReady: true},
		{name: "CPU without Ollama", backend: "cpu", capabilities: nil, wantRuntime: "ollama", wantReady: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtimeName, ready := runtimeReadiness(test.backend, test.capabilities)
			if runtimeName != test.wantRuntime || ready != test.wantReady {
				t.Fatalf("runtimeReadiness() = (%q, %v), want (%q, %v)", runtimeName, ready, test.wantRuntime, test.wantReady)
			}
		})
	}
}
