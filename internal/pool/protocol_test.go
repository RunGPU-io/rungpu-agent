package pool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestHeartbeatSent(t *testing.T) {
	upgrader := websocket.Upgrader{}
	var mu sync.Mutex
	var frames []map[string]interface{}

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
			if json.Unmarshal(data, &msg) == nil {
				mu.Lock()
				frames = append(frames, msg)
				mu.Unlock()
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey: "hb-key", PoolURL: srv.URL + "/",
		GPUIDs: []string{"gpu-hb"}, PricePerMinute: 0.01,
		ModelCacheDir: t.TempDir(), MaxModelCacheGB: 1,
		HeartbeatIntervalSecs: 1,
	}
	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = client.Run(ctx) }()
	time.Sleep(3500 * time.Millisecond)
	cancel()

	mu.Lock()
	defer mu.Unlock()
	reg, hb := 0, 0
	for _, f := range frames {
		switch f["type"] {
		case "gpu_register":
			reg++
		case "gpu_heartbeat":
			hb++
		}
	}
	if reg < 1 {
		t.Errorf("expected >= 1 gpu_register, got %d", reg)
	}
	if hb < 2 {
		t.Errorf("expected >= 2 gpu_heartbeat, got %d", hb)
	}
}

func TestJobDispatchAndResult(t *testing.T) {
	upgrader := websocket.Upgrader{}
	resultCh := make(chan map[string]interface{}, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		conn.ReadMessage() // register
		conn.WriteJSON(map[string]interface{}{
			"type": "job_assignment", "job_id": "j1",
			"model_name": "test", "input": map[string]interface{}{"prompt": "hi"},
			"parameters": map[string]interface{}{}, "vram_required_gb": 1.0,
		})
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]interface{}
			if json.Unmarshal(data, &msg) == nil && msg["type"] == "job_result" {
				resultCh <- msg
				return
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey: "j-key", PoolURL: srv.URL + "/",
		GPUIDs: []string{"gpu-j"}, PricePerMinute: 0.01,
		ModelCacheDir: t.TempDir(), MaxModelCacheGB: 1,
		HeartbeatIntervalSecs: 60,
	}
	client, _ := NewClient(cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	select {
	case r := <-resultCh:
		if r["type"] != "job_result" {
			t.Errorf("type = %v", r["type"])
		}
		if r["job_id"] != "j1" {
			t.Errorf("job_id = %v", r["job_id"])
		}
		if r["gpu_id"] != "gpu-j" {
			t.Errorf("gpu_id = %v", r["gpu_id"])
		}
		if _, ok := r["duration_ms"]; !ok {
			t.Error("missing duration_ms")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("timeout waiting for job_result")
	}
}

func TestReconnectOnDisconnect(t *testing.T) {
	upgrader := websocket.Upgrader{}
	var mu sync.Mutex
	n := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		cur := n
		mu.Unlock()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		if cur == 1 {
			conn.ReadMessage()
			conn.Close()
			return
		}
		defer conn.Close()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	defer srv.Close()

	cfg := &types.Config{
		APIKey: "rc-key", PoolURL: srv.URL + "/",
		GPUIDs: []string{"gpu-rc"}, PricePerMinute: 0.01,
		ModelCacheDir: t.TempDir(), MaxModelCacheGB: 1,
		HeartbeatIntervalSecs: 60,
	}
	client, _ := NewClient(cfg)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = client.Run(ctx) }()
	time.Sleep(5 * time.Second)
	cancel()

	mu.Lock()
	defer mu.Unlock()
	if n < 2 {
		t.Errorf("expected >= 2 connections, got %d", n)
	}
}

func TestRegisterMessageFormat(t *testing.T) {
	cfg := &types.Config{
		APIKey: "f-key", PoolURL: "https://pool.rungpu.io",
		GPUIDs: []string{"gpu-f"}, PricePerMinute: 0.05,
		ModelCacheDir: t.TempDir(), MaxModelCacheGB: 10,
	}
	client, _ := NewClient(cfg)
	msg := client.registerMessage()

	if msg.Type != "gpu_register" {
		t.Errorf("Type = %q", msg.Type)
	}
	if msg.GPUID != "gpu-f" {
		t.Errorf("GPUID = %q", msg.GPUID)
	}
	if msg.Backend == "" {
		t.Error("Backend empty")
	}
	data, _ := json.Marshal(msg)
	var raw map[string]interface{}
	json.Unmarshal(data, &raw)
	for _, k := range []string{"type", "gpu_id", "gpu_type", "backend", "vram_gb", "price_per_minute", "detected_device_count", "capabilities"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("missing field %q", k)
		}
	}
	if _, ok := raw["hostname"]; ok {
		t.Error("registration must not expose the operating-system hostname")
	}
}

func TestHeartbeatMessageFormat(t *testing.T) {
	cfg := &types.Config{
		APIKey: "hf-key", PoolURL: "https://pool.rungpu.io",
		GPUIDs: []string{"gpu-hf"}, PricePerMinute: 0.01,
		ModelCacheDir: t.TempDir(), MaxModelCacheGB: 1,
	}
	client, _ := NewClient(cfg)
	msg := client.heartbeatMessage()

	if msg.Type != "gpu_heartbeat" {
		t.Errorf("Type = %q", msg.Type)
	}

	data, _ := json.Marshal(msg)
	var raw map[string]interface{}
	json.Unmarshal(data, &raw)
	for _, k := range []string{"type", "gpu_id", "available_vram_gb", "current_jobs", "models_cached"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("missing field %q", k)
		}
	}
}
