package pool

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tokenize/gpu-agent/internal/types"
)

func TestPruneCustomAssetsRequiresIdleMachine(t *testing.T) {
	cacheDir := t.TempDir()
	asset := filepath.Join(cacheDir, "assets", "expired.safetensors")
	if err := os.MkdirAll(filepath.Dir(asset), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(asset, []byte("asset"), 0o600); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	old := now.Add(-8 * 24 * time.Hour)
	if err := os.Chtimes(asset, old, old); err != nil {
		t.Fatal(err)
	}

	cfg := &types.Config{ModelCacheDir: cacheDir, CustomAssetTTLDays: 7}
	client := &Client{}
	atomic.StoreInt64(&client.activeJobs, 1)
	if _, _, skipped, err := pruneCustomAssetsIfIdle(cfg, []*Client{client}, now); err != nil || !skipped {
		t.Fatalf("busy cleanup: skipped=%v err=%v", skipped, err)
	}
	if _, err := os.Stat(asset); err != nil {
		t.Fatalf("busy cleanup removed asset: %v", err)
	}

	atomic.StoreInt64(&client.activeJobs, 0)
	files, _, skipped, err := pruneCustomAssetsIfIdle(cfg, []*Client{client}, now)
	if err != nil || skipped || files != 1 {
		t.Fatalf("idle cleanup: files=%d skipped=%v err=%v", files, skipped, err)
	}
	if _, err := os.Stat(asset); !os.IsNotExist(err) {
		t.Fatalf("expired asset still exists: %v", err)
	}
}
