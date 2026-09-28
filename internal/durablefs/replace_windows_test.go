//go:build windows

package durablefs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWindowsWriteThroughReplacementAndDirectoryCommit(t *testing.T) {
	dir := t.TempDir()
	source, destination := filepath.Join(dir, "source"), filepath.Join(dir, "destination")
	if err := os.WriteFile(source, []byte("new"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := Replace(source, destination); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(destination); err != nil || string(data) != "new" {
		t.Fatalf("replacement failed: %q %v", data, err)
	}
	stage, committed := filepath.Join(dir, "stage"), filepath.Join(dir, "committed")
	if err := os.Mkdir(stage, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := MoveNew(stage, committed); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(stage, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := MoveNew(stage, committed); err == nil {
		t.Fatal("directory commit replaced an existing directory")
	}
	if err := Replace(filepath.Join(dir, "missing"), destination); err == nil {
		t.Fatal("write-through move failure was ignored")
	}
}
