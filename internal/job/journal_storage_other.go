//go:build !windows

package job

import (
	"fmt"
	"os"
	"path/filepath"
)

func privateDirectory(path string) error {
	var created []string
	for current := path; ; current = filepath.Dir(current) {
		if _, err := os.Lstat(current); !os.IsNotExist(err) {
			break
		}
		created = append(created, current)
	}
	if err := os.MkdirAll(path, 0o700); err != nil {
		return err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode().Perm() != 0o700 {
		return fmt.Errorf("journal directory must be a private, nonsymlink directory: %s", path)
	}
	for i := len(created) - 1; i >= 0; i-- {
		parent, err := os.Open(filepath.Dir(created[i]))
		if err != nil {
			return err
		}
		err = parent.Sync()
		parent.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func validatePrivateJournalFile(path string, info os.FileInfo) error {
	if !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 {
		return fmt.Errorf("journal file has unsafe type or permissions: %s", path)
	}
	return nil
}

func openJournalLockFile(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
}

func createJournalStage(dir string) (*os.File, error) { return os.CreateTemp(dir, ".stage-") }
