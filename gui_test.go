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
	"time"
)

func newTestGUIController() *guiController {
	return &guiController{
		lastPing:   time.Now(),
		authority:  "127.0.0.1:32123",
		origin:     "http://127.0.0.1:32123",
		capability: "test-gui-capability",
	}
}

func serveGUI(ctrl *guiController, rec *httptest.ResponseRecorder, req *http.Request) {
	req.Host = ctrl.authority
	req.Header.Set("Origin", ctrl.origin)
	req.Header.Set("X-RunGPU-GUI-Token", ctrl.capability)
	ctrl.handler().ServeHTTP(rec, req)
}

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
		{name: "linux gui command", in: "rungpu-agent gui\nrungpu-agent init --enrollment-token enroll_linux", want: "enroll_linux"},
		{name: "multiline bootstrap", in: "rungpu-agent init --enrollment-token enroll_boot\nrungpu-agent setup\nrungpu-agent start", want: "enroll_boot"},
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
	ctrl := newTestGUIController()
	req := httptest.NewRequest(http.MethodPost, "/api/enroll", strings.NewReader(`{"token":""}`))
	rec := httptest.NewRecorder()
	serveGUI(ctrl, rec, req)
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

func TestGUIRejectsRequestsWithoutLaunchCapability(t *testing.T) {
	ctrl := newTestGUIController()
	req := httptest.NewRequest(http.MethodPost, "/api/stop", nil)
	req.Host = ctrl.authority
	req.Header.Set("Origin", ctrl.origin)
	rec := httptest.NewRecorder()

	ctrl.handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", rec.Code)
	}
}

func TestGUIRejectsCrossOriginAndWrongHost(t *testing.T) {
	ctrl := newTestGUIController()
	for _, test := range []struct {
		name   string
		host   string
		origin string
	}{
		{name: "cross origin", host: ctrl.authority, origin: "https://attacker.example"},
		{name: "wrong host", host: "attacker.example", origin: ctrl.origin},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/stop", nil)
			req.Host = test.host
			req.Header.Set("Origin", test.origin)
			req.Header.Set("X-RunGPU-GUI-Token", ctrl.capability)
			rec := httptest.NewRecorder()
			ctrl.handler().ServeHTTP(rec, req)
			if rec.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want 403", rec.Code)
			}
		})
	}
}

func TestGUIMutationsRequirePost(t *testing.T) {
	ctrl := newTestGUIController()
	req := httptest.NewRequest(http.MethodGet, "/api/unenroll", nil)
	rec := httptest.NewRecorder()
	serveGUI(ctrl, rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
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

func isolateAgentHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	return home
}

func writeAgentConfig(t *testing.T, home, body string) string {
	t.Helper()
	dir := filepath.Join(home, ".tokenize")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestGUIServesHTML(t *testing.T) {
	ctrl := newTestGUIController()
	req := httptest.NewRequest(http.MethodGet, "/?token="+ctrl.capability, nil)
	req.Host = ctrl.authority
	rec := httptest.NewRecorder()
	ctrl.handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{
		"RunGPU Agent",
		`id="pause"`,
		`id="resume"`,
		`id="enroll"`,
		"Connect this computer",
		"Open the host dashboard",
		"Paste the token or the full init command",
		"init --batch-token-file",
		`api("/api/ping").then(render)`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("html missing %q", want)
		}
	}
}

func TestGUIPingRefreshesLease(t *testing.T) {
	ctrl := newTestGUIController()
	ctrl.lastPing = time.Now().Add(-time.Minute)
	before := ctrl.lastPing
	rec := httptest.NewRecorder()

	serveGUI(ctrl, rec, httptest.NewRequest(http.MethodGet, "/api/ping", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !ctrl.lastPing.After(before) {
		t.Fatal("ping did not refresh the GUI lease")
	}
}

func TestGUIStateWhenUnenrolled(t *testing.T) {
	isolateAgentHome(t)
	ctrl := newTestGUIController()
	req := httptest.NewRequest(http.MethodGet, "/api/state", nil)
	rec := httptest.NewRecorder()
	serveGUI(ctrl, rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["enrolled"] != false {
		t.Fatalf("enrolled = %v", body["enrolled"])
	}
	if body["paused"] != false {
		t.Fatalf("paused = %v", body["paused"])
	}
}

func TestGUIPauseResumeRoundTrip(t *testing.T) {
	home := isolateAgentHome(t)
	writeAgentConfig(t, home, "api_key: test-key\nmachine_id: test-machine\n")
	ctrl := newTestGUIController()
	ctrl.running = true

	pause := httptest.NewRecorder()
	serveGUI(ctrl, pause, httptest.NewRequest(http.MethodPost, "/api/pause", nil))
	if pause.Code != http.StatusOK {
		t.Fatalf("pause status = %d body=%s", pause.Code, pause.Body.String())
	}
	var paused map[string]any
	if err := json.Unmarshal(pause.Body.Bytes(), &paused); err != nil {
		t.Fatal(err)
	}
	if paused["enrolled"] != true || paused["paused"] != true {
		t.Fatalf("after pause: %+v", paused)
	}

	resume := httptest.NewRecorder()
	serveGUI(ctrl, resume, httptest.NewRequest(http.MethodPost, "/api/resume", nil))
	if resume.Code != http.StatusOK {
		t.Fatalf("resume status = %d body=%s", resume.Code, resume.Body.String())
	}
	var resumed map[string]any
	if err := json.Unmarshal(resume.Body.Bytes(), &resumed); err != nil {
		t.Fatal(err)
	}
	if resumed["paused"] != false {
		t.Fatalf("paused should be false after resume: %+v", resumed)
	}
}

func TestGUIScheduleUpdatesConfig(t *testing.T) {
	home := isolateAgentHome(t)
	path := writeAgentConfig(t, home, "api_key: test-key\nmachine_id: test-machine\n")
	ctrl := newTestGUIController()
	ctrl.running = true
	req := httptest.NewRequest(http.MethodPost, "/api/schedule", strings.NewReader(`{"enabled":true,"timezone":"America/Los_Angeles","windows":[{"start_hour":9,"end_hour":17,"days":["mon","tue"]}]}`))
	rec := httptest.NewRecorder()
	serveGUI(ctrl, rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, "enabled: true") || !strings.Contains(text, "America/Los_Angeles") {
		t.Fatalf("schedule not saved: %s", text)
	}
}

func TestCmdPauseResume(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("api_key: test\nmachine_id: test-id\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := cmdPause([]string{"--config", path}); err != nil {
		t.Fatal(err)
	}
	paused, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(paused), "paused: true") {
		t.Fatalf("pause did not persist: %s", paused)
	}
	if err := cmdResume([]string{"--config", path}); err != nil {
		t.Fatal(err)
	}
	resumed, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(resumed), "paused: false") {
		t.Fatalf("resume did not persist: %s", resumed)
	}
}
