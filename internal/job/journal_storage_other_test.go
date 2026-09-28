//go:build !windows

package job

import "os"

func setJournalDirectoryWritable(path string, writable bool) error {
	mode := os.FileMode(0o500)
	if writable {
		mode = 0o700
	}
	return os.Chmod(path, mode)
}

func makeJournalFilePublic(path string) error { return os.Chmod(path, 0o644) }
