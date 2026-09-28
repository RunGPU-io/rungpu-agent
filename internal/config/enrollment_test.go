package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestInstallationConcurrentCreationKeepsOnePrivateProof(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	var wg sync.WaitGroup
	ids, secrets := make(chan string, 16), make(chan string, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id, err := LoadOrCreateInstallationID(path)
			if err != nil {
				t.Error(err)
				return
			}
			secret, err := LoadOrCreateInstallationSecret(path)
			if err != nil {
				t.Error(err)
				return
			}
			ids <- id
			secrets <- secret
		}()
	}
	wg.Wait()
	close(ids)
	close(secrets)
	for _, values := range []chan string{ids, secrets} {
		first := ""
		for value := range values {
			if first == "" {
				first = value
			}
			if first != value {
				t.Fatal("competing creators replaced installation proof")
			}
		}
	}
}

func TestInstallationDoesNotReplaceDamagedOrInsecureProof(t *testing.T) {
	for _, condition := range []string{"empty", "invalid", "directory", "insecure"} {
		t.Run(condition, func(t *testing.T) {
			if condition == "insecure" && runtime.GOOS == "windows" {
				t.Skip("Unix mode check")
			}
			path := filepath.Join(t.TempDir(), "config.yaml")
			secretPath := filepath.Join(filepath.Dir(path), "installation-secret")
			content := ""
			if condition == "invalid" {
				content = "broken"
			}
			if condition == "directory" {
				os.Mkdir(secretPath, 0700)
			} else {
				mode := os.FileMode(0600)
				if condition == "insecure" {
					content, mode = strings.Repeat("a", 43), 0644
				}
				os.WriteFile(secretPath, []byte(content), mode)
			}
			if _, err := LoadOrCreateInstallationSecret(path); err == nil {
				t.Fatal("invalid existing proof should not be regenerated")
			}
			if condition != "directory" {
				after, _ := os.ReadFile(secretPath)
				if string(after) != content {
					t.Fatal("existing proof was changed")
				}
			}
		})
	}
}

func TestAtomicSaveReplacesInsecureModeAndPreflightPreservesConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	os.WriteFile(path, []byte("old config"), 0644)
	if err := PreflightSave(path); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	if string(before) != "old config" {
		t.Fatal("preflight changed existing credentials")
	}
	if err := Save(&types.Config{APIKey: "private"}, path); err != nil {
		t.Fatal(err)
	}
	info, _ := os.Stat(path)
	if configPermissionsNeedWarning(runtime.GOOS, info.Mode()) {
		t.Fatal("replacement retained insecure permissions")
	}
	files, _ := filepath.Glob(filepath.Join(filepath.Dir(path), ".enrollment-write-*"))
	if len(files) != 0 {
		t.Fatal("staged credential files were not cleaned up")
	}
}

func TestAtomicSaveRejectsSymlinkWithoutDamagingTarget(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink privileges vary on Windows")
	}
	dir := t.TempDir()
	target, link := filepath.Join(dir, "original"), filepath.Join(dir, "config.yaml")
	os.WriteFile(target, []byte("original"), 0600)
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	if err := Save(&types.Config{APIKey: "new"}, link); err == nil {
		t.Fatal("symlink should be rejected")
	}
	data, _ := os.ReadFile(target)
	if string(data) != "original" {
		t.Fatal("original credential was damaged")
	}
}

func TestAtomicSaveReadersNeverObservePartialCredentials(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	first, second := &types.Config{APIKey: "first"}, &types.Config{APIKey: "second"}
	one, two := generateCommentedConfig(first), generateCommentedConfig(second)
	if err := Save(first, path); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	var readers sync.WaitGroup
	readers.Add(1)
	go func() {
		defer readers.Done()
		for {
			select {
			case <-done:
				return
			default:
				data, err := os.ReadFile(path)
				if err != nil || (string(data) != one && string(data) != two) {
					t.Errorf("observed incomplete credential file: %v", err)
					return
				}
			}
		}
	}()
	for i := 0; i < 30; i++ {
		cfg := first
		if i%2 == 0 {
			cfg = second
		}
		if err := Save(cfg, path); err != nil {
			t.Error(err)
			break
		}
	}
	close(done)
	readers.Wait()
}
