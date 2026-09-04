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

func TestParseEnrollmentInput(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "token only", in: "enroll_abc123", want: "enroll_abc123"},
		{name: "quoted token", in: `"enroll_abc123"`, want: "enroll_abc123"},
		{name: "windows command", in: `.\rungpu-agent-windows-amd64.exe init --enrollment-token enroll_win`, want: "enroll_win"},
		{name: "mac command", in: "./rungpu-agent-darwin-arm64 init --enrollment-token enroll_mac", want: "enroll_mac"},
		{name: "equals form", in: "rungpu-agent init --enrollment-token=enroll_eq", want: "enroll_eq"},
		{name: "empty", in: "   ", want: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := parseEnrollmentInput(test.in); got != test.want {
				t.Fatalf("parseEnrollmentInput(%q) = %q, want %q", test.in, got, test.want)
			}
		})
	}
}

func TestGUIEnrollRejectsEmptyToken(t *testing.T) {
	ctrl := &guiController{}
	req := httptest.NewRequest(http.MethodPost, "/api/enroll", strings.NewReader(`{"token":""}`))
	rec := httptest.NewRecorder()
	ctrl.handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body["error"], "enrollment token") {
		t.Fatalf("error = %q", body["error"])
	}
}

func TestOnlyLoopbackRejectsRemote(t *testing.T) {
	handler := onlyLoopback(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.RemoteAddr = "203.0.113.9:443"
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", rec.Code)
	}
}

func TestSupportsDesktopGUIMatchesReleaseTargets(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "darwin", "linux":
		if !supportsDesktopGUI() {
			t.Fatal("Windows, macOS, and Linux should offer the GUI")
		}
	default:
		if supportsDesktopGUI() {
			t.Fatal("desktop GUI is only offered on Windows, macOS, and Linux")
		}
	}
}

func TestCmdUnenrollDeletesConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("api_key: test\nmachine_id: test-id\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := cmdUnenroll([]string{"--config", path}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("config still present: %v", err)
	}
	if err := cmdUnenroll([]string{"--config", path}); err != nil {
		t.Fatal(err)
	}
}
