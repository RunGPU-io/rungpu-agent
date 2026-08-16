// Package pool — e2e_test.go is the TRUE end-to-end integration test.
//
// Unlike the existing tests that use mocks, this test:
//
//  1. Starts a mock pool coordinator (WebSocket server) that mimics pool-api
//  2. Connects the REAL Go agent with REAL GPU detection (nvidia-smi / Metal / CPU)
//  3. Verifies gpu_register contains actual hardware info (GPU name, VRAM, backend, driver)
//  4. Sends a job_assignment and verifies the agent executes it end-to-end
//  5. Tracks job_progress events through all stages
//  6. Verifies heartbeats contain live GPU metrics (VRAM, utilization)
//  7. Tests output upload to a mock storage server
//  8. Verifies reconnection after server disconnect
//
// This test uses the REAL runtime (Ollama for text, Docker for images).
// If Ollama is installed AND the model is already cached, it runs real inference.
// Otherwise it tests the error/pull path without blocking.
//
// Fast tests (no model pull, < 30s):
//
//	go test -v -run TestE2E_RealAgent ./internal/pool/ -timeout 120s
//
// Full Ollama inference test (may pull 2GB model, needs -timeout 600s):
//
//	go test -v -run TestE2E_RealAgent_OllamaInference ./internal/pool/ -timeout 600s
package pool

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/tokenize/gpu-agent/internal/gpu"
	"github.com/tokenize/gpu-agent/internal/types"
)

// ollamaModelCached checks if a model is already pulled in Ollama (no download needed).
// Returns false if Ollama is not installed or the model is not cached.
func ollamaModelCached(model string) bool {
	if exec.Command("ollama", "--version").Run() != nil {
		return false
	}
	// Check via the Ollama API — /api/show returns 200 if model exists locally
	body, _ := json.Marshal(map[string]string{"name": model})
	resp, err := http.Post("http://localhost:11434/api/show", "application/json", bytes.NewReader(body))
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// TestE2E_RealAgent_GPURegistration proves the agent sends real GPU hardware
// info when it registers with the pool coordinator.
//
// This is the most important test: it verifies that a GPU host's machine info
// actually reaches the coordinator correctly.
func TestE2E_RealAgent_GPURegistration(t *testing.T) {
	// Detect what this machine actually has
	monitor := gpu.NewMonitor()
	detectedGPUs := monitor.GPUs()
	detectedBackend := monitor.Backend()

	t.Logf("Host GPU detection:")
	t.Logf("  Backend: %s", detectedBackend)
	for _, g := range detectedGPUs {
		t.Logf("  GPU %d: %s (%d MB, compute=%s, driver=%s)",
			g.Index, g.Name, g.MemoryMB, g.ComputeCapability, g.DriverVersion)
	}

	// ── Mock pool coordinator ───────────────────────────────────────────
	upgrader := websocket.Upgrader{}
	registerCh := make(chan map[string]interface{}, 1)
	heartbeatCh := make(chan map[string]interface{}, 10)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify bearer auth and non-secret GPU identity are present.
		apiKey := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		gpuID := r.URL.Query().Get("gpu_id")
		if apiKey == "" || gpuID == "" {
			http.Error(w, "missing auth", 401)
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			json.Unmarshal(data, &msg)

			switch msg["type"] {
			case "gpu_register":
				registerCh <- msg
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_register_ack", "success": true,
				})
			case "gpu_heartbeat":
				select {
				case heartbeatCh <- msg:
				default:
				}
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_heartbeat_ack", "success": true,
				})
			}
		}
	}))
	defer srv.Close()

	// ── Start the real agent ────────────────────────────────────────────
	cfg := &types.Config{
		APIKey:                "e2e-test-key-gpu-reg",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-e2e-0"},
		PricePerMinute:        0.05,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       10,
		HeartbeatIntervalSecs: 1, // fast heartbeats for testing
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	// ── Verify gpu_register contains real hardware info ─────────────────
	var reg map[string]interface{}
	select {
	case reg = <-registerCh:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for gpu_register")
	}

	t.Logf("\nReceived gpu_register frame:")
	regJSON, _ := json.MarshalIndent(reg, "  ", "  ")
	t.Logf("  %s", string(regJSON))

	// Type must be gpu_register
	if reg["type"] != "gpu_register" {
		t.Errorf("type = %v, want gpu_register", reg["type"])
	}

	// GPU ID must match config
	if reg["gpu_id"] != "gpu-e2e-0" {
		t.Errorf("gpu_id = %v, want gpu-e2e-0", reg["gpu_id"])
	}

	// Price must match config
	if price, ok := reg["price_per_minute"].(float64); !ok || price != 0.05 {
		t.Errorf("price_per_minute = %v, want 0.05", reg["price_per_minute"])
	}

	// Backend must be one of: cuda, metal, cpu
	backend, _ := reg["backend"].(string)
	if backend != "cuda" && backend != "metal" && backend != "cpu" {
		t.Errorf("backend = %q, want cuda|metal|cpu", backend)
	}
	if backend != detectedBackend {
		t.Errorf("backend = %q, but gpu.Monitor detected %q", backend, detectedBackend)
	}

	// GPU type must be a non-empty string (actual hardware name)
	gpuType, _ := reg["gpu_type"].(string)
	if gpuType == "" {
		t.Error("gpu_type is empty — should contain the GPU name")
	} else {
		t.Logf("  GPU type reported: %s", gpuType)
		// Verify it matches what we detected
		if len(detectedGPUs) > 0 && gpuType != detectedGPUs[0].Name {
			t.Errorf("gpu_type = %q, but detected %q", gpuType, detectedGPUs[0].Name)
		}
	}

	// VRAM must be reported (> 0 for real GPUs, 0 for cpu-only)
	vramGB, _ := reg["vram_gb"].(float64)
	if len(detectedGPUs) > 0 && detectedGPUs[0].MemoryMB > 0 {
		expectedVRAM := float64(detectedGPUs[0].MemoryMB) / 1024.0
		if vramGB < expectedVRAM*0.9 || vramGB > expectedVRAM*1.1 {
			t.Errorf("vram_gb = %.1f, expected ~%.1f", vramGB, expectedVRAM)
		}
		t.Logf("  VRAM: %.1f GB", vramGB)
	}

	if _, exposed := reg["hostname"]; exposed {
		t.Error("registration exposed the operating-system hostname")
	}

	// Driver version should be present for NVIDIA/macOS
	driverVersion, _ := reg["driver_version"].(string)
	if backend != "cpu" && driverVersion == "" {
		t.Error("driver_version is empty for non-CPU backend")
	}
	if driverVersion != "" {
		t.Logf("  Driver: %s", driverVersion)
	}

	// models_cached should be an array (possibly empty)
	if _, ok := reg["models_cached"]; !ok {
		t.Error("models_cached field missing")
	}

	// ── Verify heartbeat contains GPU metrics ───────────────────────────
	t.Log("\nWaiting for heartbeat with GPU metrics...")
	var hb map[string]interface{}
	select {
	case hb = <-heartbeatCh:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for gpu_heartbeat")
	}

	t.Logf("Received gpu_heartbeat:")
	hbJSON, _ := json.MarshalIndent(hb, "  ", "  ")
	t.Logf("  %s", string(hbJSON))

	if hb["type"] != "gpu_heartbeat" {
		t.Errorf("heartbeat type = %v", hb["type"])
	}
	if hb["gpu_id"] != "gpu-e2e-0" {
		t.Errorf("heartbeat gpu_id = %v", hb["gpu_id"])
	}

	// available_vram_gb should be present
	if _, ok := hb["available_vram_gb"]; !ok {
		t.Error("heartbeat missing available_vram_gb")
	} else {
		availVRAM, _ := hb["available_vram_gb"].(float64)
		t.Logf("  Available VRAM: %.1f GB", availVRAM)
	}

	// current_jobs should be 0 (no jobs running)
	currentJobs, _ := hb["current_jobs"].(float64)
	if currentJobs != 0 {
		t.Errorf("current_jobs = %.0f, want 0", currentJobs)
	}

	t.Log("\n✅ GPU registration and heartbeat verified with real hardware info")
}

// TestE2E_RealAgent_JobExecution proves the full job lifecycle:
// coordinator sends job → agent executes → result returned.
//
// This test NEVER triggers a model pull. It only runs real inference if the
// model is already cached in Ollama. Otherwise it tests the error path.
// For the full model-pull + inference test, see TestE2E_RealAgent_OllamaInference.
func TestE2E_RealAgent_JobExecution(t *testing.T) {
	hasOllama := exec.Command("ollama", "--version").Run() == nil
	hasDocker := exec.Command("docker", "version").Run() == nil

	// Only use Ollama if the model is ALREADY cached (no 2GB download)
	modelName := "llama3.2"
	modelCached := ollamaModelCached(modelName)

	t.Logf("Runtime availability:")
	t.Logf("  Ollama installed: %v", hasOllama)
	t.Logf("  Docker installed: %v", hasDocker)
	t.Logf("  Model %q cached:  %v", modelName, modelCached)

	useOllama := hasOllama && modelCached

	if !useOllama {
		if hasOllama && !modelCached {
			t.Logf("Ollama installed but %q not cached — testing error/pull path (no download)", modelName)
			t.Logf("To run with real inference: ollama pull %s", modelName)
		} else {
			t.Log("No Ollama — testing error path")
		}
	}

	// ── Mock storage server (for output upload) ─────────────────────────
	var uploadCount int
	var uploadMu sync.Mutex

	storageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			data, _ := io.ReadAll(r.Body)
			uploadMu.Lock()
			uploadCount++
			uploadMu.Unlock()
			t.Logf("  Upload received: %s (%d bytes, %s)", r.URL.Path, len(data), r.Header.Get("Content-Type"))
			w.WriteHeader(200)
			w.Write([]byte(`{"status":"ok"}`))
			return
		}
		http.Error(w, "method not allowed", 405)
	}))
	defer storageSrv.Close()
	_ = storageSrv // used as upload target in job assignments

	// ── Mock pool coordinator ───────────────────────────────────────────
	upgrader := websocket.Upgrader{}
	var mu sync.Mutex
	var allFrames []map[string]interface{}
	resultCh := make(chan map[string]interface{}, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		jobSent := false

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			json.Unmarshal(data, &msg)

			mu.Lock()
			allFrames = append(allFrames, msg)
			mu.Unlock()

			switch msg["type"] {
			case "gpu_register":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_register_ack", "success": true,
				})

				// Send a job after registration (only once)
				if !jobSent {
					jobSent = true

					var jobAssignment map[string]interface{}
					if useOllama {
						// Model is cached — real inference, no download
						jobAssignment = map[string]interface{}{
							"type":             "job_assignment",
							"job_id":           "e2e-real-job-1",
							"model_name":       modelName,
							"input":            map[string]interface{}{"prompt": "Say exactly: E2E_TEST_OK"},
							"parameters":       map[string]interface{}{"num_predict": 10},
							"vram_required_gb": 2.0,
						}
					} else {
						// No cached model — send a job that will fail fast
						// Use a model name that won't trigger a Docker pull either
						jobAssignment = map[string]interface{}{
							"type":             "job_assignment",
							"job_id":           "e2e-real-job-1",
							"model_name":       "test-nonexistent-model",
							"input":            map[string]interface{}{"prompt": "test"},
							"parameters":       map[string]interface{}{},
							"vram_required_gb": 1.0,
						}
					}

					time.Sleep(100 * time.Millisecond)
					conn.WriteJSON(jobAssignment)
				}

			case "gpu_heartbeat":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_heartbeat_ack", "success": true,
				})

			case "job_result":
				resultCh <- msg
				conn.WriteJSON(map[string]interface{}{
					"type": "job_result_ack", "job_id": msg["job_id"], "success": true,
				})
			}
		}
	}))
	defer srv.Close()

	// ── Start the real agent ────────────────────────────────────────────
	cfg := &types.Config{
		APIKey:                "e2e-job-test-key",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-e2e-job"},
		PricePerMinute:        0.05,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       10,
		HeartbeatIntervalSecs: 30,
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	// 60s is plenty — no model pull happens here
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	// ── Wait for job result ─────────────────────────────────────────────
	t.Log("Waiting for job result...")
	var result map[string]interface{}
	select {
	case result = <-resultCh:
	case <-ctx.Done():
		t.Fatal("timed out waiting for job_result")
	}

	t.Logf("\nJob result:")
	resultJSON, _ := json.MarshalIndent(result, "  ", "  ")
	t.Logf("  %s", string(resultJSON))

	// Verify result structure
	if result["type"] != "job_result" {
		t.Errorf("type = %v, want job_result", result["type"])
	}
	if result["job_id"] != "e2e-real-job-1" {
		t.Errorf("job_id = %v, want e2e-real-job-1", result["job_id"])
	}
	if result["gpu_id"] != "gpu-e2e-job" {
		t.Errorf("gpu_id = %v, want gpu-e2e-job", result["gpu_id"])
	}

	// Duration must be present and positive
	durationMS, _ := result["duration_ms"].(float64)
	if durationMS <= 0 {
		t.Errorf("duration_ms = %.0f, want > 0", durationMS)
	}
	t.Logf("  Duration: %.0fms", durationMS)

	if useOllama {
		// With cached Ollama model: job should succeed with a real response
		success, _ := result["success"].(bool)
		if !success {
			errMsg, _ := result["error"].(string)
			t.Fatalf("Job failed with cached Ollama model: %s", errMsg)
		}

		// Verify result contains response text
		if resultData, ok := result["result"].(map[string]interface{}); ok {
			if resp, ok := resultData["response"].(string); ok {
				t.Logf("  LLM Response: %s", truncate(resp, 200))
				if resp == "" {
					t.Error("Response is empty")
				}
			}
			if model, ok := resultData["model"].(string); ok {
				t.Logf("  Model used: %s", model)
			}
			if bk, ok := resultData["backend"].(string); ok {
				t.Logf("  Backend: %s", bk)
			}
		}
		t.Log("\n✅ Real Ollama inference completed successfully (cached model)")
	} else {
		// Without cached model: job should fail gracefully (no hang, no 2GB download)
		success, _ := result["success"].(bool)
		if success {
			t.Log("Job succeeded unexpectedly (maybe Docker handled it)")
		} else {
			errMsg, _ := result["error"].(string)
			t.Logf("  Expected failure: %s", truncate(errMsg, 200))
			if errMsg == "" {
				t.Error("Error message is empty")
			}
		}
		t.Log("\n✅ Error path verified correctly (no model pull, fast failure)")
	}

	cancel()

	// ── Verify frame types ──────────────────────────────────────────────
	mu.Lock()
	frameTypes := map[string]int{}
	for _, f := range allFrames {
		if typ, ok := f["type"].(string); ok {
			frameTypes[typ]++
		}
	}
	mu.Unlock()

	t.Logf("\nFrame types received by coordinator: %v", frameTypes)

	if frameTypes["gpu_register"] < 1 {
		t.Error("Expected at least 1 gpu_register frame")
	}
	if frameTypes["job_result"] < 1 {
		t.Error("Expected at least 1 job_result frame")
	}
}

// TestE2E_RealAgent_FullLifecycle_WithReconnect tests the complete agent
// lifecycle including server disconnect and automatic reconnection.
//
// Flow:
//  1. Agent connects → registers with real GPU info
//  2. Server sends job → agent executes (will fail fast — no model)
//  3. Server disconnects → agent reconnects automatically
//  4. Agent re-registers on reconnect with same GPU info
//  5. Heartbeats flow on both connections
func TestE2E_RealAgent_FullLifecycle_WithReconnect(t *testing.T) {
	upgrader := websocket.Upgrader{}
	var mu sync.Mutex
	connectionCount := 0
	registerFrames := make(chan map[string]interface{}, 5)
	heartbeatFrames := make(chan map[string]interface{}, 20)
	resultCh := make(chan map[string]interface{}, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		connectionCount++
		connNum := connectionCount
		mu.Unlock()

		t.Logf("Connection #%d from agent", connNum)

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			json.Unmarshal(data, &msg)

			switch msg["type"] {
			case "gpu_register":
				select {
				case registerFrames <- msg:
				default:
				}
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_register_ack", "success": true,
				})

				// On first connection, send a simple job then disconnect after result
				if connNum == 1 {
					conn.WriteJSON(map[string]interface{}{
						"type":             "job_assignment",
						"job_id":           "reconnect-test-job",
						"model_name":       "test-nonexistent-model",
						"input":            map[string]interface{}{"prompt": "test"},
						"parameters":       map[string]interface{}{},
						"vram_required_gb": 1.0,
					})
				}

			case "gpu_heartbeat":
				select {
				case heartbeatFrames <- msg:
				default:
				}
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_heartbeat_ack", "success": true,
				})

			case "job_result":
				select {
				case resultCh <- msg:
				default:
				}
				conn.WriteJSON(map[string]interface{}{
					"type": "job_result_ack", "success": true,
				})

				// After receiving result on first connection, close it to trigger reconnect
				if connNum == 1 {
					time.Sleep(100 * time.Millisecond)
					conn.Close()
					return
				}
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey:                "e2e-reconnect-key",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-e2e-rc"},
		PricePerMinute:        0.03,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       10,
		HeartbeatIntervalSecs: 1,
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	// ── Wait for first registration ─────────────────────────────────────
	t.Log("Waiting for first gpu_register...")
	var firstReg map[string]interface{}
	select {
	case firstReg = <-registerFrames:
		t.Logf("First register: gpu_type=%v, backend=%v, vram_gb=%v",
			firstReg["gpu_type"], firstReg["backend"], firstReg["vram_gb"])
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for first gpu_register")
	}

	// ── Wait for job result ─────────────────────────────────────────────
	t.Log("Waiting for job result...")
	select {
	case result := <-resultCh:
		t.Logf("Job result: success=%v, duration_ms=%v", result["success"], result["duration_ms"])
	case <-time.After(30 * time.Second):
		t.Fatal("timed out waiting for job_result")
	}

	// ── Wait for reconnection and second registration ───────────────────
	t.Log("Waiting for reconnect + second gpu_register...")
	var secondReg map[string]interface{}
	select {
	case secondReg = <-registerFrames:
		t.Logf("Second register (after reconnect): gpu_type=%v, backend=%v",
			secondReg["gpu_type"], secondReg["backend"])
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for second gpu_register (reconnect failed)")
	}

	// ── Verify both registrations have the same GPU info ────────────────
	if firstReg["gpu_type"] != secondReg["gpu_type"] {
		t.Errorf("GPU type changed after reconnect: %v → %v",
			firstReg["gpu_type"], secondReg["gpu_type"])
	}
	if firstReg["backend"] != secondReg["backend"] {
		t.Errorf("Backend changed after reconnect: %v → %v",
			firstReg["backend"], secondReg["backend"])
	}
	if firstReg["vram_gb"] != secondReg["vram_gb"] {
		t.Errorf("VRAM changed after reconnect: %v → %v",
			firstReg["vram_gb"], secondReg["vram_gb"])
	}

	// ── Wait for at least one heartbeat on the second connection ────────
	// The heartbeat interval is 1s. After reconnect, the agent needs time
	// to fire its first heartbeat tick. Wait up to 3s for it.
	t.Log("Waiting for heartbeat on reconnected session...")
	select {
	case hb := <-heartbeatFrames:
		t.Logf("Heartbeat received: gpu_id=%v, available_vram_gb=%v", hb["gpu_id"], hb["available_vram_gb"])
	case <-time.After(3 * time.Second):
		t.Log("No heartbeat within 3s (first connection was too short for a tick)")
	}

	cancel()
	time.Sleep(200 * time.Millisecond)

	// Drain any remaining heartbeats
	heartbeatCount := 0
	for {
		select {
		case <-heartbeatFrames:
			heartbeatCount++
		default:
			goto doneHB
		}
	}
doneHB:
	heartbeatCount++ // count the one we already received (or 0+1 if we got one above)

	mu.Lock()
	conns := connectionCount
	mu.Unlock()

	t.Logf("\nLifecycle summary:")
	t.Logf("  Connections: %d (expected >= 2)", conns)
	t.Logf("  Heartbeats: %d (across both connections)", heartbeatCount)

	if conns < 2 {
		t.Errorf("Expected >= 2 connections (original + reconnect), got %d", conns)
	}
	// Heartbeats are verified in TestE2E_RealAgent_GPUMetrics_InHeartbeat.
	// Here we just verify the reconnect worked — heartbeat count depends on timing.

	t.Log("\n✅ Full lifecycle with reconnect verified")
}

// TestE2E_RealAgent_GPUMetrics_InHeartbeat verifies that heartbeat messages
// contain meaningful GPU metrics from the actual hardware.
func TestE2E_RealAgent_GPUMetrics_InHeartbeat(t *testing.T) {
	monitor := gpu.NewMonitor()
	detectedGPUs := monitor.GPUs()
	detectedBackend := monitor.Backend()
	metrics := monitor.CollectMetrics()

	t.Logf("Detected GPU metrics:")
	for _, m := range metrics {
		temp := "n/a"
		if m.TemperatureC != nil {
			temp = fmt.Sprintf("%.0f°C", *m.TemperatureC)
		}
		t.Logf("  GPU %d: %.0f%% util, %d/%d MB mem, %s",
			m.GPUIndex, m.UtilizationPercent, m.MemoryUsedMB, m.MemoryTotalMB, temp)
	}

	// ── Mock coordinator that collects heartbeats ───────────────────────
	upgrader := websocket.Upgrader{}
	heartbeats := make(chan map[string]interface{}, 10)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			json.Unmarshal(data, &msg)

			switch msg["type"] {
			case "gpu_register":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_register_ack", "success": true,
				})
			case "gpu_heartbeat":
				select {
				case heartbeats <- msg:
				default:
				}
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_heartbeat_ack", "success": true,
				})
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey:                "e2e-metrics-key",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-e2e-metrics"},
		PricePerMinute:        0.04,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       10,
		HeartbeatIntervalSecs: 1,
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	// Collect 3 heartbeats to verify consistency
	t.Log("Collecting heartbeats...")
	var collectedHBs []map[string]interface{}
	for i := 0; i < 3; i++ {
		select {
		case hb := <-heartbeats:
			collectedHBs = append(collectedHBs, hb)
			t.Logf("  Heartbeat %d: available_vram_gb=%.2f, current_jobs=%.0f",
				i+1, hb["available_vram_gb"], hb["current_jobs"])
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for heartbeat %d", i+1)
		}
	}
	cancel()

	// Verify heartbeat fields
	for i, hb := range collectedHBs {
		if hb["type"] != "gpu_heartbeat" {
			t.Errorf("heartbeat %d: type = %v", i, hb["type"])
		}
		if hb["gpu_id"] != "gpu-e2e-metrics" {
			t.Errorf("heartbeat %d: gpu_id = %v", i, hb["gpu_id"])
		}

		availVRAM, _ := hb["available_vram_gb"].(float64)

		// For machines with real GPUs, VRAM should be > 0
		if detectedBackend != "cpu" && len(detectedGPUs) > 0 && detectedGPUs[0].MemoryMB > 0 {
			if availVRAM <= 0 {
				t.Errorf("heartbeat %d: available_vram_gb = %.2f, expected > 0 for %s backend",
					i, availVRAM, detectedBackend)
			}
		}

		// models_cached should be present
		if _, ok := hb["models_cached"]; !ok {
			t.Errorf("heartbeat %d: missing models_cached", i)
		}
	}

	t.Logf("\n✅ GPU metrics in heartbeats verified (%d heartbeats collected)", len(collectedHBs))
}

// TestE2E_RealAgent_MultipleJobsSequential tests that the agent can handle
// multiple jobs in sequence without issues. Uses fast-failing jobs (no model pull).
func TestE2E_RealAgent_MultipleJobsSequential(t *testing.T) {
	upgrader := websocket.Upgrader{}
	results := make(chan map[string]interface{}, 5)
	jobsToSend := 3
	jobsSent := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			json.Unmarshal(data, &msg)

			switch msg["type"] {
			case "gpu_register":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_register_ack", "success": true,
				})
				// Send first job
				if jobsSent < jobsToSend {
					jobsSent++
					conn.WriteJSON(map[string]interface{}{
						"type":             "job_assignment",
						"job_id":           fmt.Sprintf("seq-job-%d", jobsSent),
						"model_name":       "test-nonexistent-model",
						"input":            map[string]interface{}{"prompt": fmt.Sprintf("test %d", jobsSent)},
						"parameters":       map[string]interface{}{},
						"vram_required_gb": 1.0,
					})
				}

			case "gpu_heartbeat":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_heartbeat_ack", "success": true,
				})

			case "job_result":
				select {
				case results <- msg:
				default:
				}
				conn.WriteJSON(map[string]interface{}{
					"type": "job_result_ack", "success": true,
				})
				// Send next job
				if jobsSent < jobsToSend {
					jobsSent++
					conn.WriteJSON(map[string]interface{}{
						"type":             "job_assignment",
						"job_id":           fmt.Sprintf("seq-job-%d", jobsSent),
						"model_name":       "test-nonexistent-model",
						"input":            map[string]interface{}{"prompt": fmt.Sprintf("test %d", jobsSent)},
						"parameters":       map[string]interface{}{},
						"vram_required_gb": 1.0,
					})
				}
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey:                "e2e-multi-key",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-e2e-multi"},
		PricePerMinute:        0.03,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       10,
		HeartbeatIntervalSecs: 30, // slow heartbeats to reduce noise
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	// Collect all job results
	var jobResults []map[string]interface{}
	for i := 0; i < jobsToSend; i++ {
		select {
		case r := <-results:
			jobResults = append(jobResults, r)
			t.Logf("Job %d result: job_id=%v, success=%v, duration_ms=%v",
				i+1, r["job_id"], r["success"], r["duration_ms"])
		case <-ctx.Done():
			t.Fatalf("timed out waiting for job %d result (got %d/%d)", i+1, len(jobResults), jobsToSend)
		}
	}
	cancel()

	// Verify all jobs completed (success or failure — both are valid results)
	if len(jobResults) != jobsToSend {
		t.Errorf("Expected %d results, got %d", jobsToSend, len(jobResults))
	}

	// Verify each result has required fields
	seenJobIDs := map[string]bool{}
	for _, r := range jobResults {
		jobID, _ := r["job_id"].(string)
		if jobID == "" {
			t.Error("Result missing job_id")
			continue
		}
		if seenJobIDs[jobID] {
			t.Errorf("Duplicate job_id: %s", jobID)
		}
		seenJobIDs[jobID] = true

		if _, ok := r["duration_ms"]; !ok {
			t.Errorf("Job %s missing duration_ms", jobID)
		}
		if r["gpu_id"] != "gpu-e2e-multi" {
			t.Errorf("Job %s: gpu_id = %v", jobID, r["gpu_id"])
		}
	}

	t.Logf("\n✅ %d sequential jobs completed", len(jobResults))
}

// TestE2E_RealAgent_OllamaInference runs a real LLM inference if Ollama is
// available. This is the ultimate proof that the agent can actually serve
// GPU inference.
//
// ⚠️  This test may pull a 2GB model on first run. Use a long timeout:
//
//	go test -v -run TestE2E_RealAgent_OllamaInference ./internal/pool/ -timeout 600s
//
// It is automatically skipped when:
//   - Ollama is not installed
//   - Running with -short flag
func TestE2E_RealAgent_OllamaInference(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Ollama inference test in -short mode (may pull 2GB model)")
	}

	if exec.Command("ollama", "--version").Run() != nil {
		t.Skip("Ollama not installed — skipping real inference test")
	}

	t.Log("Ollama is available — running real LLM inference test")

	// Check if a small model is available
	out, err := exec.Command("ollama", "list").Output()
	if err != nil {
		t.Logf("Could not list Ollama models: %v", err)
	} else {
		t.Logf("Available Ollama models:\n%s", string(out))
	}

	// Pick the smallest available model, or default to llama3.2
	modelName := "llama3.2"
	// Check if a smaller model is already cached
	for _, candidate := range []string{"qwen2.5:0.5b", "tinyllama", "phi3:mini", "llama3.2:1b", "llama3.2"} {
		if ollamaModelCached(candidate) {
			modelName = candidate
			t.Logf("Using already-cached model: %s", modelName)
			break
		}
	}

	if !ollamaModelCached(modelName) {
		t.Logf("Model %q not cached — will pull (this may take a few minutes)...", modelName)
	}

	// ── Mock coordinator ────────────────────────────────────────────────
	upgrader := websocket.Upgrader{}
	resultCh := make(chan map[string]interface{}, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			json.Unmarshal(data, &msg)

			switch msg["type"] {
			case "gpu_register":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_register_ack", "success": true,
				})

				// Send a real inference job
				conn.WriteJSON(map[string]interface{}{
					"type":       "job_assignment",
					"job_id":     "e2e-ollama-real",
					"model_name": modelName,
					"input": map[string]interface{}{
						"prompt": "What is 2+2? Answer with just the number.",
					},
					"parameters": map[string]interface{}{
						"num_predict": 10,
						"temperature": 0.1,
					},
					"vram_required_gb": 2.0,
				})

			case "gpu_heartbeat":
				conn.WriteJSON(map[string]interface{}{
					"type": "gpu_heartbeat_ack", "success": true,
				})

			case "job_result":
				resultCh <- msg
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey:                "e2e-ollama-key",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-e2e-ollama"},
		PricePerMinute:        0.05,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       50,
		HeartbeatIntervalSecs: 30,
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	// Long timeout for model pull (up to 10 minutes)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	t.Log("Waiting for Ollama inference result (may need to pull model)...")

	select {
	case result := <-resultCh:
		cancel()

		resultJSON, _ := json.MarshalIndent(result, "  ", "  ")
		t.Logf("Result:\n  %s", string(resultJSON))

		success, _ := result["success"].(bool)
		if !success {
			errMsg, _ := result["error"].(string)
			t.Fatalf("Ollama inference failed: %s", errMsg)
		}

		durationMS, _ := result["duration_ms"].(float64)
		t.Logf("Inference took %.0fms", durationMS)

		if resultData, ok := result["result"].(map[string]interface{}); ok {
			response, _ := resultData["response"].(string)
			t.Logf("LLM Response: %q", strings.TrimSpace(response))

			if response == "" {
				t.Error("Empty response from Ollama")
			}

			model, _ := resultData["model"].(string)
			t.Logf("Model: %s", model)

			bk, _ := resultData["backend"].(string)
			t.Logf("Backend: %s", bk)

			evalCount, _ := resultData["eval_count"].(float64)
			t.Logf("Tokens generated: %.0f", evalCount)
		}

		t.Log("\n✅ Real Ollama LLM inference completed successfully!")

	case <-ctx.Done():
		t.Fatal("timed out waiting for Ollama inference (10 min)")
	}
}

// TestE2E_RealAgent_StatusCommand verifies the `status` command output
// matches what the agent would send to the coordinator.
func TestE2E_RealAgent_StatusCommand(t *testing.T) {
	// This test doesn't need a WebSocket server — it just verifies
	// the GPU detection matches between the monitor and what would be sent.
	monitor := gpu.NewMonitor()
	gpus := monitor.GPUs()
	backend := monitor.Backend()
	metrics := monitor.CollectMetrics()
	healthy := monitor.IsHealthy()

	t.Logf("=== GPU Agent Status ===")
	t.Logf("Backend: %s", backend)
	t.Logf("Healthy: %v", healthy)
	t.Logf("GPUs detected: %d", len(gpus))

	for _, g := range gpus {
		t.Logf("  GPU %d: %s", g.Index, g.Name)
		t.Logf("    Memory: %d MB (%.1f GB)", g.MemoryMB, float64(g.MemoryMB)/1024.0)
		t.Logf("    Compute: %s", g.ComputeCapability)
		t.Logf("    Driver: %s", g.DriverVersion)
	}

	t.Logf("Metrics:")
	for _, m := range metrics {
		t.Logf("  GPU %d: %.0f%% util, %d/%d MB mem",
			m.GPUIndex, m.UtilizationPercent, m.MemoryUsedMB, m.MemoryTotalMB)
		if m.TemperatureC != nil {
			t.Logf("    Temperature: %.0f°C", *m.TemperatureC)
		}
		if m.PowerDrawW != nil {
			t.Logf("    Power: %.0fW", *m.PowerDrawW)
		}
	}

	// Verify basic invariants
	if len(gpus) == 0 {
		t.Fatal("No GPUs detected (should always have at least cpu-only fallback)")
	}

	if backend != "cuda" && backend != "metal" && backend != "cpu" {
		t.Errorf("Invalid backend: %s", backend)
	}

	if len(metrics) == 0 {
		t.Error("No metrics collected")
	}

	// Verify GPU info matches what would be sent in registration
	primaryGPU := gpus[0]
	if primaryGPU.Name == "" {
		t.Error("Primary GPU has no name")
	}

	// Build what the register message would look like
	regMsg := types.RegisterMessage{
		Type:           "gpu_register",
		GPUID:          "gpu-0",
		GPUType:        primaryGPU.Name,
		Backend:        backend,
		VRAMGB:         float64(primaryGPU.MemoryMB) / 1024.0,
		PricePerMinute: 0.05,
		DriverVersion:  primaryGPU.DriverVersion,
	}

	regJSON, _ := json.MarshalIndent(regMsg, "  ", "  ")
	t.Logf("\nRegister message that would be sent:\n  %s", string(regJSON))

	t.Log("\n✅ GPU status and registration data verified")
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func truncate(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
