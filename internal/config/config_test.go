package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/RunGPU-io/gpu-agent/internal/types"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	original := &types.Config{
		APIKey:                "key-abc",
		PoolURL:               "http://localhost:3001",
		GPUIDs:                []string{"gpu-0", "gpu-1"},
		PricePerMinute:        0.05,
		ModelCacheDir:         "/tmp/models",
		MaxModelCacheGB:       50,
		CleanupIntervalHours:  12,
		HeartbeatIntervalSecs: 15,
		Metrics:               types.MetricsConfig{EnableGPUMonitoring: true, MonitoringIntervalSecs: 7},
	}

	if err := Save(original, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if loaded.APIKey != original.APIKey ||
		loaded.PoolURL != original.PoolURL ||
		loaded.PricePerMinute != original.PricePerMinute ||
		loaded.MaxModelCacheGB != original.MaxModelCacheGB ||
		len(loaded.GPUIDs) != 2 ||
		loaded.GPUIDs[1] != "gpu-1" ||
		loaded.Metrics.MonitoringIntervalSecs != 7 {
		t.Errorf("round-trip mismatch: %+v", loaded)
	}
}

func TestValidate(t *testing.T) {
	good := &types.Config{APIKey: "k", PoolURL: "u", GPUIDs: []string{"g"}, MaxModelCacheGB: 1}
	if err := Validate(good); err != nil {
		t.Errorf("expected valid config, got %v", err)
	}

	cases := map[string]*types.Config{
		"missing api_key":      {PoolURL: "u", GPUIDs: []string{"g"}, MaxModelCacheGB: 1},
		"missing pool_url":     {APIKey: "k", GPUIDs: []string{"g"}, MaxModelCacheGB: 1},
		"no gpus":              {APIKey: "k", PoolURL: "u", MaxModelCacheGB: 1},
		"zero cache":           {APIKey: "k", PoolURL: "u", GPUIDs: []string{"g"}},
	}
	for name, cfg := range cases {
		if err := Validate(cfg); err == nil {
			t.Errorf("%s: expected validation error, got nil", name)
		}
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "nope.yaml")); err == nil {
		t.Error("expected error loading missing file")
	}
}

func TestSaveCreatesFileWith0600(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "secure.yaml")

	cfg := &types.Config{
		APIKey:          "secret-key-123",
		PoolURL:         "https://pool.rungpu.io",
		GPUIDs:          []string{"gpu-0"},
		MaxModelCacheGB: 10,
	}

	if err := Save(cfg, path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}

	perm := info.Mode().Perm()
	// On Unix, should be 0600 (owner read/write only).
	// On Windows, file permissions work differently — skip the check.
	if perm&0o077 != 0 {
		t.Errorf("config file permissions = %o, want 0600 (no group/other access)", perm)
	}
}

func TestNewAutoDetectsGPUs(t *testing.T) {
	cfg, err := New("test-api-key")
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if cfg.APIKey != "test-api-key" {
		t.Errorf("APIKey = %q, want test-api-key", cfg.APIKey)
	}
	if cfg.PoolURL == "" {
		t.Error("PoolURL should have a default")
	}
	if len(cfg.GPUIDs) == 0 {
		t.Error("GPUIDs should have at least one entry (fallback)")
	}
	if cfg.HeartbeatIntervalSecs <= 0 {
		t.Error("HeartbeatIntervalSecs should be positive")
	}
	if cfg.ModelCacheDir == "" {
		t.Error("ModelCacheDir should have a default")
	}
}

func TestValidateRejectsEmptyAPIKey(t *testing.T) {
	cfg := &types.Config{
		APIKey:          "",
		PoolURL:         "https://pool.rungpu.io",
		GPUIDs:          []string{"gpu-0"},
		MaxModelCacheGB: 10,
	}
	if err := Validate(cfg); err == nil {
		t.Error("expected error for empty API key")
	}
}
