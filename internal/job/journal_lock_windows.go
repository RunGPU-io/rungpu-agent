//go:build windows

package job

import (
	"os"
	"syscall"
	"unsafe"
)

func lockJournalFile(file *os.File) error {
	return lockMachineFile(file, false)
}

func lockRootFile(file *os.File) error {
	var overlapped syscall.Overlapped
	lock := syscall.NewLazyDLL("kernel32.dll").NewProc("LockFileEx")
	ok, _, err := lock.Call(file.Fd(), 2, 0, 1, 0, uintptr(unsafe.Pointer(&overlapped)))
	if ok == 0 {
		return err
	}
	return nil
}

func lockMachineFile(file *os.File, shared bool) error {
	var overlapped syscall.Overlapped
	lock := syscall.NewLazyDLL("kernel32.dll").NewProc("LockFileEx")
	flags := uintptr(3)
	if shared {
		flags = 1
	}
	ok, _, err := lock.Call(file.Fd(), flags, 0, 1, 0, uintptr(unsafe.Pointer(&overlapped)))
	if ok == 0 {
		return err
	}
	return nil
}
