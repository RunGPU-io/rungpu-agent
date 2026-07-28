package pool

import (
	"net/url"
	"testing"

	"github.com/RunGPU-io/gpu-agent/internal/types"
)

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
			if q.Get("api_key") != "secret-key" {
				t.Errorf("api_key = %q, want %q", q.Get("api_key"), "secret-key")
			}
			if q.Get("gpu_id") != tc.gpuID {
				t.Errorf("gpu_id = %q, want %q", q.Get("gpu_id"), tc.gpuID)
			}
		})
	}
}
