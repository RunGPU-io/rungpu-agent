//go:build windows

package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	kernel32CreateMutex = syscall.NewLazyDLL("kernel32.dll").NewProc("CreateMutexW")
	kernel32CloseHandle = syscall.NewLazyDLL("kernel32.dll").NewProc("CloseHandle")
)

const errorAlreadyExists syscall.Errno = 183

func acquireAgentProcessLock() (func(), error) {
	name, err := syscall.UTF16PtrFromString(`Local\RunGPUAgent`)
	if err != nil {
		return nil, fmt.Errorf("create agent mutex name: %w", err)
	}
	handle, _, callErr := kernel32CreateMutex.Call(0, 0, uintptr(unsafe.Pointer(name)))
	if handle == 0 {
		return nil, fmt.Errorf("create agent process mutex: %w", callErr)
	}
	if errno, ok := callErr.(syscall.Errno); ok && errno == errorAlreadyExists {
		_, _, _ = kernel32CloseHandle.Call(handle)
		return nil, fmt.Errorf("another RunGPU agent is already running")
	}
	return func() {
		_, _, _ = kernel32CloseHandle.Call(handle)
	}, nil
}
