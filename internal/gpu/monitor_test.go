package gpu

import (
	"testing"

	"github.com/tokenize/gpu-agent/internal/types"
)

func TestSplitCSV(t *testing.T) {
	got := splitCSV("0, NVIDIA GeForce RTX 4090, 24564, 535.154.05")
	want := []string{"0", "NVIDIA GeForce RTX 4090", "24564", "535.154.05"}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("field %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestDetectAlwaysReturnsOne(t *testing.T) {
	// Detect must never return empty, even with no NVIDIA GPU present.
	gpus := Detect()
	if len(gpus) == 0 {
		t.Fatal("Detect() returned no GPUs; expected at least a fallback entry")
	}
}

func TestIsHealthyWithNoMetrics(t *testing.T) {
	// A monitor with a CPU-only / zero-memory device must report healthy
	// (no division-by-zero, no false negatives).
	m := &Monitor{
		gpus:      []types.GPUInfo{{Index: 0, Name: "cpu-only", MemoryMB: 0}},
		hasNvidia: false,
	}
	if !m.IsHealthy() {
		t.Error("CPU-only monitor should report healthy")
	}
}
