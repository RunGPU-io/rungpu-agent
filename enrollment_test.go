package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/config"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestSingleEnrollmentRecoveryAfterLostResponseAndSaveFailure(t *testing.T) {
	for _, failure := range []string{"response", "save"} {
		t.Run(failure, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			var requests []map[string]string
			var mu sync.Mutex
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api/v1/fleet/enroll" {
					t.Errorf("path = %s", r.URL.Path)
				}
				var body map[string]string
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				mu.Lock()
				requests = append(requests, body)
				first := len(requests) == 1
				mu.Unlock()
				if first && failure == "response" {

					w.Write([]byte(`{"machine_id":`))
					return
				}
				if first && failure == "save" {
					if err := os.Mkdir(path, 0700); err != nil {
						t.Error(err)
					}
				}
				w.Write([]byte(`{"machine_id":"machine-1","agent_key":"agent-stable","price_per_minute":0}`))
			}))
			defer server.Close()
			err := initializeEnrollment(server.URL, "enroll-token", "", path)
			if err == nil || !strings.Contains(err.Error(), "installation-secret") {
				t.Fatalf("expected actionable recovery error: %v", err)
			}
			if failure == "save" {
				if !strings.Contains(err.Error(), "server enrollment succeeded") {
					t.Fatal(err)
				}
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			}
			if err := initializeEnrollment(server.URL, "enroll-token", "", path); err != nil {
				t.Fatal(err)
			}
			cfg, err := config.Load(path)
			if err != nil || cfg.APIKey != "agent-stable" || cfg.MachineID != "machine-1" {
				t.Fatalf("config = %+v, err = %v", cfg, err)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(requests) != 2 || requests[0]["installation_id"] == "" ||
				len(requests[0]["installation_secret"]) != 43 ||
				requests[0]["installation_id"] != requests[1]["installation_id"] ||
				requests[0]["installation_secret"] != requests[1]["installation_secret"] {
				t.Fatal("retry did not retain installation proof")
			}
		})
	}
}

func TestSingleEnrollmentReplayPreservesSavedConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"machine_id":"machine-1","agent_key":"agent-stable","price_per_minute":0}`))
	}))
	defer server.Close()
	cfg := &types.Config{APIKey: "agent-stable", MachineID: "machine-1", PoolURL: server.URL, Paused: true, PricePerMinute: 0.7}
	if err := config.Save(cfg, path); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	if err := initializeEnrollment(server.URL, "token", "", path); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("replay overwrote saved settings")
	}
}

func TestSingleEnrollmentPreservesDifferentMachineIncludingBatchConfig(t *testing.T) {
	for _, credential := range []string{"agent-other-single", "agent-batch-issued"} {
		t.Run(credential, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(`{"machine_id":"recovered-machine","agent_key":"recovered-key","price_per_minute":0}`))
			}))
			defer server.Close()
			if err := config.Save(&types.Config{APIKey: credential, MachineID: "currently-enrolled-machine", PoolURL: server.URL, Paused: true}, path); err != nil {
				t.Fatal(err)
			}
			before, _ := os.ReadFile(path)
			err := initializeEnrollment(server.URL, "old-single-token", "", path)
			if err == nil || !strings.Contains(err.Error(), "belongs to machine") || !strings.Contains(err.Error(), "existing config was preserved") {
				t.Fatalf("expected actionable machine mismatch: %v", err)
			}
			after, _ := os.ReadFile(path)
			if string(before) != string(after) {
				t.Fatal("different machine's credential/config was overwritten")
			}
		})
	}
}

func TestSingleEnrollmentRejectsDifferentPoolBeforeHTTP(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(`{"machine_id":"machine-1","agent_key":"new-key"}`))
	}))
	defer server.Close()
	if err := config.Save(&types.Config{APIKey: "batch-key", MachineID: "machine-1", PoolURL: "https://original-pool.example"}, path); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	err := initializeEnrollment(server.URL, "token", "", path)
	if err == nil || !strings.Contains(err.Error(), "different pool") || !strings.Contains(err.Error(), "no enrollment request sent") {
		t.Fatalf("expected actionable pool mismatch: %v", err)
	}
	after, _ := os.ReadFile(path)
	if calls.Load() != 0 || string(before) != string(after) {
		t.Fatal("pool mismatch sent a request or changed config")
	}
}

func TestSingleEnrollmentSameMachineReplacementRotatesKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"machine_id":"machine-1","agent_key":"replacement-key","price_per_minute":0.04}`))
	}))
	defer server.Close()
	if err := config.Save(&types.Config{APIKey: "old-batch-key", MachineID: "machine-1", PoolURL: server.URL + "/", Paused: true, ModelCacheDir: "keep-models"}, path); err != nil {
		t.Fatal(err)
	}
	if err := initializeEnrollment(server.URL, "replacement-token", "", path); err != nil {
		t.Fatal(err)
	}
	saved, err := config.Load(path)
	if err != nil || saved.APIKey != "replacement-key" || saved.MachineID != "machine-1" ||
		saved.PricePerMinute != 0.04 || !saved.Paused || saved.ModelCacheDir != "keep-models" {
		t.Fatalf("same-machine replacement failed or reset local settings: %+v %v", saved, err)
	}
}

func TestSingleEnrollmentRechecksConfigAfterResponse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	changed := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		replacement := &types.Config{APIKey: "restored-key", MachineID: "other-machine", PoolURL: "http://" + r.Host}
		if err := config.Save(replacement, path); err != nil {
			t.Error(err)
		}
		data, _ := os.ReadFile(path)
		changed <- data
		w.Write([]byte(`{"machine_id":"recovered-machine","agent_key":"recovered-key"}`))
	}))
	defer server.Close()
	err := initializeEnrollment(server.URL, "token", "", path)
	if err == nil || !strings.Contains(err.Error(), "belongs to machine") {
		t.Fatalf("expected post-response mismatch: %v", err)
	}
	after, _ := os.ReadFile(path)
	if string(after) != string(<-changed) {
		t.Fatal("config restored during enrollment was overwritten")
	}
}

func TestSingleEnrollmentPreflightAndLockPreventNetwork(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(500)
	}))
	defer server.Close()
	for _, condition := range []string{"directory", "lock", "invalid-proof"} {
		t.Run(condition, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			switch condition {
			case "directory":
				os.Mkdir(path, 0700)
			case "lock":
				unlock, err := config.LockEnrollment(path)
				if err != nil {
					t.Fatal(err)
				}
				defer unlock()
			case "invalid-proof":
				os.WriteFile(filepath.Join(filepath.Dir(path), "installation-secret"), []byte("broken"), 0600)
			}
			if err := initializeEnrollment(server.URL, "token", "", path); err == nil {
				t.Fatal("expected local failure")
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("sent %d requests despite failed preflight", calls.Load())
	}
}

func TestSingleEnrollmentDoesNotFollowCredentialRedirect(t *testing.T) {
	var leaked atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { leaked.Add(1) }))
	defer target.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	if _, err := enrollMachine(server.URL, "token", "id", "secret"); err == nil {
		t.Fatal("expected redirect rejection")
	}
	if leaked.Load() != 0 {
		t.Fatal("installation proof followed redirect")
	}
}

func TestEnrollmentSuccessDoesNotClaimConnection(t *testing.T) {
	ctrl := &guiController{}
	if err := ctrl.withOutput(func() error {
		printEnrollmentSuccess("config.yaml", "rungpu-agent start")
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	ctrl.mu.Lock()
	output := ctrl.logs.String()
	ctrl.mu.Unlock()
	for _, want := range []string{"enrolled with RunGPU", "Credentials saved to: config.yaml", "Start the agent to connect", "rungpu-agent start"} {
		if !strings.Contains(output, want) {
			t.Fatalf("success output missing %q: %s", want, output)
		}
	}
	if strings.Contains(output, "is connected") {
		t.Fatal("enrollment claimed an active connection")
	}
	if !strings.Contains(string(guiHTML), `state.running ? "Agent process started" : "Not connected yet"`) ||
		!strings.Contains(string(guiHTML), "Enrolled — click Start to connect") {
		t.Fatal("GUI must distinguish enrollment/process state from a pool connection")
	}
}

func TestGUIEnrollmentUsesPersistencePreflight(t *testing.T) {
	home := isolateAgentHome(t)
	path := filepath.Join(home, ".tokenize", "config.yaml")
	if err := os.MkdirAll(path, 0700); err != nil {
		t.Fatal(err)
	}
	ctrl := newTestGUIController()
	req := httptest.NewRequest(http.MethodPost, "/api/enroll", strings.NewReader(`{"token":"enroll-token"}`))
	rec := httptest.NewRecorder()
	serveGUI(ctrl, rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "no enrollment request sent") {
		t.Fatalf("GUI enrollment did not return preflight error: %d %s", rec.Code, rec.Body.String())
	}
}
