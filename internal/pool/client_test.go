package pool

import (
	"context"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestJobResultAckOnlyRemovesAcceptedResult(t *testing.T) {
	client := &Client{outboxDir: t.TempDir()}
	resultPath := filepath.Join(client.outboxDir, "job-1.json")
	if err := os.WriteFile(resultPath, []byte(`{"job_id":"job-1"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	client.dispatch(context.Background(), make(chan interface{}, 1), []byte(`{"type":"job_result_ack","job_id":"job-1","success":false}`))
	if _, err := os.Stat(resultPath); err != nil {
		t.Fatalf("rejected result ACK removed durable result: %v", err)
	}

	client.dispatch(context.Background(), make(chan interface{}, 1), []byte(`{"type":"job_result_ack","job_id":"job-1","success":true}`))
	if _, err := os.Stat(resultPath); !os.IsNotExist(err) {
		t.Fatalf("accepted result ACK did not remove durable result: %v", err)
	}
}

func TestEndpoint(t *testing.T) {
	cases := []struct {
		name       string
		poolURL    string
		gpuID      string
		wantScheme string
		wantPath   string
	}{
		{"https to wss", "https://pool.rungpu.io", "gpu-0", "wss", "/agent"},
		{"http to ws", "http://localhost:3001", "gpu-1", "ws", "/agent"},
		{"wss stays wss", "wss://pool.rungpu.io", "gpu-0", "wss", "/agent"},
		{"trailing slash", "http://localhost:3001/", "gpu-0", "ws", "/agent"},
		{"base path preserved", "https://host/api", "gpu-2", "wss", "/api/agent"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &Client{
				cfg:   &types.Config{PoolURL: tc.poolURL, APIKey: "secret-key"},
				gpuID: tc.gpuID,
			}
			got, err := c.endpoint()
			if err != nil {
				t.Fatalf("endpoint() error: %v", err)
			}
			u, err := url.Parse(got)
			if err != nil {
				t.Fatalf("result not a valid URL %q: %v", got, err)
			}
			if u.Scheme != tc.wantScheme {
				t.Errorf("scheme = %q, want %q", u.Scheme, tc.wantScheme)
			}
			if u.Path != tc.wantPath {
				t.Errorf("path = %q, want %q", u.Path, tc.wantPath)
			}
			q := u.Query()
			if q.Get("api_key") != "" {
				t.Errorf("endpoint leaked api_key = %q", q.Get("api_key"))
			}
			if q.Get("gpu_id") != tc.gpuID {
				t.Errorf("gpu_id = %q, want %q", q.Get("gpu_id"), tc.gpuID)
			}
		})
	}
}

func TestEndpointRejectsRemotePlaintext(t *testing.T) {
	c := &Client{cfg: &types.Config{PoolURL: "http://pool.example.com"}, gpuID: "gpu"}
	if _, err := c.endpoint(); err == nil {
		t.Fatal("expected remote plaintext WebSocket URL to be rejected")
	}
}

func TestDeterministicGPUID(t *testing.T) {
	first := deterministicGPUID("49c6e0e1-75c7-4fa2-a5cb-bb55ebcec900", 0)
	if first != deterministicGPUID("49c6e0e1-75c7-4fa2-a5cb-bb55ebcec900", 0) {
		t.Fatal("same machine and device must produce a stable GPU ID")
	}
	if first == deterministicGPUID("49c6e0e1-75c7-4fa2-a5cb-bb55ebcec900", 1) {
		t.Fatal("different devices must produce different GPU IDs")
	}
}
