package pool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
	"github.com/gorilla/websocket"
)

func TestRegisterHandshake(t *testing.T) {
	upgrader := websocket.Upgrader{}
	received := make(chan map[string]interface{}, 1)
	authParams := make(chan map[string]string, 1)

	mux := http.NewServeMux()
	mux.HandleFunc("/agent", func(w http.ResponseWriter, r *http.Request) {
		authParams <- map[string]string{
			"authorization": r.Header.Get("Authorization"),
			"gpu_id":        r.URL.Query().Get("gpu_id"),
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade failed: %v", err)
			return
		}
		defer conn.Close()
		_, data, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var msg map[string]interface{}
		if err := json.Unmarshal(data, &msg); err != nil {
			t.Errorf("bad frame: %v", err)
			return
		}
		received <- msg

		time.Sleep(100 * time.Millisecond)
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	cfg := &types.Config{
		APIKey:                "test-key",
		PoolURL:               srv.URL,
		GPUIDs:                []string{"gpu-test"},
		PricePerMinute:        0.03,
		ModelCacheDir:         t.TempDir(),
		MaxModelCacheGB:       10,
		HeartbeatIntervalSecs: 1,
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = client.Run(ctx) }()

	select {
	case ap := <-authParams:
		if ap["authorization"] != "Bearer test-key" {
			t.Errorf("authorization = %q, want Bearer test-key", ap["authorization"])
		}
		if ap["gpu_id"] != "gpu-test" {
			t.Errorf("gpu_id = %q, want gpu-test", ap["gpu_id"])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for connection")
	}

	select {
	case msg := <-received:
		if msg["type"] != "gpu_register" {
			t.Errorf("type = %v, want gpu_register", msg["type"])
		}
		if msg["gpu_id"] != "gpu-test" {
			t.Errorf("gpu_id = %v, want gpu-test", msg["gpu_id"])
		}
		if msg["price_per_minute"].(float64) != 0.03 {
			t.Errorf("price_per_minute = %v, want 0.03", msg["price_per_minute"])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for gpu_register frame")
	}
}
