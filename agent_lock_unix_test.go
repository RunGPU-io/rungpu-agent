//go:build linux || darwin

package main

import (
	"path/filepath"
	"testing"
)

func TestAgentFileLockPreventsConcurrentStartAndReleases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agent.lock")
	releaseFirst, err := acquireAgentFileLock(path)
	if err != nil {
		t.Fatal(err)
	}
	if releaseSecond, err := acquireAgentFileLock(path); err == nil {
		releaseSecond()
		releaseFirst()
		t.Fatal("second agent process lock should fail while the first is held")
	}
	releaseFirst()

	releaseThird, err := acquireAgentFileLock(path)
	if err != nil {
		t.Fatalf("lock should be reusable after release: %v", err)
	}
	releaseThird()
}
