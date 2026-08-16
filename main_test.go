package main

import "testing"

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
