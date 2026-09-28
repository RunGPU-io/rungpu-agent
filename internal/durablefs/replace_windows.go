//go:build windows

package durablefs

import (
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

var moveFileEx = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")

func Replace(source, destination string) error { return move(source, destination, 1|8) }

func MoveNew(source, destination string) error { return move(source, destination, 8) }

func move(source, destination string, flags uintptr) error {
	from, err := WindowsPath(source)
	if err != nil {
		return err
	}
	to, err := WindowsPath(destination)
	if err != nil {
		return err
	}
	ok, _, err := moveFileEx.Call(uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), flags)
	if ok == 0 {
		return err
	}
	return nil
}

func WindowsPath(path string) (*uint16, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if !strings.HasPrefix(absolute, `\\?\`) {
		if strings.HasPrefix(absolute, `\\`) {
			absolute = `\\?\UNC\` + absolute[2:]
		} else {
			absolute = `\\?\` + absolute
		}
	}
	return syscall.UTF16PtrFromString(absolute)
}
