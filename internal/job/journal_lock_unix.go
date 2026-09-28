//go:build darwin || linux || freebsd || openbsd || netbsd || dragonfly

package job

import (
	"os"
	"syscall"
)

func lockJournalFile(file *os.File) error {
	return lockMachineFile(file, false)
}

func lockRootFile(file *os.File) error { return syscall.Flock(int(file.Fd()), syscall.LOCK_EX) }

func lockMachineFile(file *os.File, shared bool) error {
	mode := syscall.LOCK_EX
	if shared {
		mode = syscall.LOCK_SH
	}
	return syscall.Flock(int(file.Fd()), mode|syscall.LOCK_NB)
}
