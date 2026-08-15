package job

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestCleanupRemovesOnlySelectedFixedDirectories(t *testing.T) {
	cacheDir := t.TempDir()
	write := func(path string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil { t.Fatal(err) }
		if err := os.WriteFile(path, []byte("asset"), 0o600); err != nil { t.Fatal(err) }
	}
	write(filepath.Join(cacheDir, "assets", "lora.safetensors"))
	write(filepath.Join(cacheDir, "staging", "job-1", "workflow.json"))
	write(filepath.Join(cacheDir, "output", "job-1", "result.png"))
	write(filepath.Join(cacheDir, "outbox", "gpu-1", "result.json"))
	write(filepath.Join(cacheDir, "config.yaml"))

	executor := &Executor{cacheDir: cacheDir, inflight: map[string]context.CancelFunc{}}
	preview, err := executor.PreviewCleanup([]string{"custom_assets", "job_files"})
	if err != nil { t.Fatal(err) }
	if preview.TotalBytes == 0 || preview.Categories["custom_assets"].Items != 1 {
		t.Fatalf("unexpected preview: %#v", preview)
	}
	if _, err := executor.ExecuteCleanup(context.Background(), []string{"custom_assets", "job_files"}); err != nil {
		t.Fatal(err)
	}
	for _, removed := range []string{"assets", "staging", "output"} {
		if _, err := os.Stat(filepath.Join(cacheDir, removed)); !os.IsNotExist(err) {
			t.Fatalf("%s was not removed", removed)
		}
	}
	for _, preserved := range []string{filepath.Join("outbox", "gpu-1", "result.json"), "config.yaml"} {
		if _, err := os.Stat(filepath.Join(cacheDir, preserved)); err != nil {
			t.Fatalf("preserved file %s missing: %v", preserved, err)
		}
	}
}

func TestCleanupRejectsUnknownCategoryAndActiveJobs(t *testing.T) {
	executor := &Executor{cacheDir: t.TempDir(), inflight: map[string]context.CancelFunc{}}
	if _, err := executor.PreviewCleanup([]string{"arbitrary_path"}); err == nil {
		t.Fatal("unknown category should fail")
	}
	executor.inflight["job-1"] = func() {}
	if _, err := executor.ExecuteCleanup(context.Background(), []string{"job_files"}); err == nil {
		t.Fatal("cleanup should be blocked while a job is active")
	}
}