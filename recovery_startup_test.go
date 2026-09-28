package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/config"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestExecutionRecoveryStartupPinsStateAndRejectsLostOrReplacedJournal(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "agent.yaml")
	cfg := &types.Config{ModelCacheDir: filepath.Join(dir, "cache")}
	if err := prepareExecutionRecovery(cfg, configPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := config.Load(configPath)
	if err != nil || loaded.ExecutionJournalID == "" || loaded.ExecutionJournalID != cfg.ExecutionJournalID {
		t.Fatalf("journal identity was not pinned outside the cache: %+v %v", loaded, err)
	}
	if err := prepareExecutionRecovery(loaded, configPath); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(cfg.ModelCacheDir, "executions")); err != nil {
		t.Fatal(err)
	}
	if err := prepareExecutionRecovery(loaded, configPath); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("lost execution state silently became a fresh agent: %v", err)
	}
	replacement := &types.Config{ModelCacheDir: cfg.ModelCacheDir}
	if err := prepareExecutionRecovery(replacement, filepath.Join(dir, "replacement.yaml")); err != nil {
		t.Fatal(err)
	}
	if err := prepareExecutionRecovery(loaded, configPath); err == nil {
		t.Fatal("a different execution-state root was accepted for the old agent config")
	}
}

func TestExecutionRecoveryStartupRequiresDurableConfigPin(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "actual.yaml")
	link := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(target, []byte("unchanged"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	cfg := &types.Config{ModelCacheDir: filepath.Join(dir, "cache")}
	if err := prepareExecutionRecovery(cfg, link); err == nil {
		t.Fatal("startup proceeded without durably pinning the journal identity")
	}
	if cfg.ExecutionJournalID != "" {
		t.Fatal("failed config write left an in-memory pin that could bypass persistence on retry")
	}
	data, err := os.ReadFile(target)
	if err != nil || string(data) != "unchanged" {
		t.Fatal("failed journal preflight damaged an unrelated config file")
	}
}
