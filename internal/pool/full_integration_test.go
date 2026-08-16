package pool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/tokenize/gpu-agent/internal/types"
)

// TestFullLifecycle tests the complete agent lifecycle in one test:
//  1. Agent connects to mock server
//  2. Agent sends gpu_register
//  3. Server sends job_assignment
//  4. Agent executes job and sends job_result
//  5. Agent sends heartbeats throughout
//  6. Server disconnects → agent reconnects
//
// This is the closest we can get to a real end-to-end test without
// running the actual pool-api server.
func TestFullLifecycle(t *testing.T) {
	upgrader := websocket.Upgrader{}
	var mu sync.Mutex
	var allFrames []map[string]interface{}
	resultCh := make(chan map[string]interface{}, 1)
	connectionCount := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		connectionCount++
		connNum := connectionCount
		mu.Unlock()

		// Verify auth on every connection
		apiKey := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		gpuID := r.URL.Query().Get("gpu_id")
		if apiKey != "lifecycle-key" || gpuID != "gpu-lc" {
			http.Error(w, "unauthorized", 401)
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		if connNum == 1 {
			// First connection: read register, send job, read result, then close
			// Read register
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var reg map[string]interface{}
			json.Unmarshal(data, &reg)
			mu.Lock()
			allFrames = append(allFrames, reg)
			mu.Unlock()

			// Send register ack
			conn.WriteJSON(map[string]interface{}{
				"type": "gpu_register_ack", "success": true,
			})

			// Send a job
			conn.WriteJSON(map[string]interface{}{
				"type":             "job_assignment",
				"job_id":           "lifecycle-job",
				"model_name":       "test-model",
				"input":            map[string]interface{}{"prompt": "hello"},
				"parameters":       map[string]interface{}{},
				"vram_required_gb": 1.0,
			})

			// Read frames until we get the job_result
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

				if msg["type"] == "job_result" {
					resultCh <- msg
					// Close connection to test reconnect
					time.Sleep(50 * time.Millisecond)
					conn.Close()
					return
				}
			}
		} else {
			// Second connection (after reconnect): just stay open
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
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey:                "lifecycle-key",
		PoolURL:               srv.URL + "/",
		GPUIDs:                []string{"gpu-lc"},
		PricePerMinute:        0.02,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       1,
		HeartbeatIntervalSecs: 1,
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	// Wait for job result
	select {
	case result := <-resultCh:
		// Verify job result
		if result["type"] != "job_result" {
			t.Errorf("type = %v", result["type"])
		}
		if result["job_id"] != "lifecycle-job" {
			t.Errorf("job_id = %v", result["job_id"])
		}
		if result["gpu_id"] != "gpu-lc" {
			t.Errorf("gpu_id = %v", result["gpu_id"])
		}
		// Job will fail (no ollama/docker) but should still return a result
		if _, ok := result["duration_ms"]; !ok {
			t.Error("missing duration_ms")
		}
		t.Logf("Job result: success=%v, error=%v, duration=%v",
			result["success"], result["error"], result["duration_ms"])
	case <-time.After(30 * time.Second):
		t.Fatal("timed out waiting for job_result")
	}

	// Wait for reconnect + second register
	time.Sleep(4 * time.Second)
	cancel()

	mu.Lock()
	defer mu.Unlock()

	// Verify we got at least 2 connections (original + reconnect)
	if connectionCount < 2 {
		t.Errorf("expected >= 2 connections, got %d", connectionCount)
	}

	// Verify frame types we received
	frameTypes := map[string]int{}
	for _, f := range allFrames {
		if t, ok := f["type"].(string); ok {
			frameTypes[t]++
		}
	}

	t.Logf("Frame types received: %v", frameTypes)

	if frameTypes["gpu_register"] < 2 {
		t.Errorf("expected >= 2 gpu_register frames (initial + reconnect), got %d", frameTypes["gpu_register"])
	}
	if frameTypes["job_result"] < 1 {
		t.Errorf("expected >= 1 job_result, got %d", frameTypes["job_result"])
	}
	if frameTypes["gpu_heartbeat"] < 1 {
		t.Errorf("expected >= 1 gpu_heartbeat, got %d", frameTypes["gpu_heartbeat"])
	}
}
