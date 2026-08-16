package job

import "testing"

func TestParseOutput(t *testing.T) {
	t.Run("valid output line among logs", func(t *testing.T) {
		logs := "[Job x] starting\n[Job x] loading\nOUTPUT:{\"status\":\"completed\",\"n\":3}\n"
		out := parseOutput(logs)
		if out == nil {
			t.Fatal("expected parsed output, got nil")
		}
		if out["status"] != "completed" {
			t.Errorf("status = %v, want completed", out["status"])
		}
		if out["n"].(float64) != 3 {
			t.Errorf("n = %v, want 3", out["n"])
		}
	})

	t.Run("uses last OUTPUT line", func(t *testing.T) {
		logs := "OUTPUT:{\"v\":1}\nnoise\nOUTPUT:{\"v\":2}\n"
		out := parseOutput(logs)
		if out == nil || out["v"].(float64) != 2 {
			t.Fatalf("expected last output v=2, got %v", out)
		}
	})

	t.Run("no output marker", func(t *testing.T) {
		if out := parseOutput("just logs, no marker\n"); out != nil {
			t.Errorf("expected nil, got %v", out)
		}
	})

	t.Run("malformed json", func(t *testing.T) {
		if out := parseOutput("OUTPUT:{not json}\n"); out != nil {
			t.Errorf("expected nil for bad json, got %v", out)
		}
	})
}
