//go:build !windows

package main

import "os"

func hideOwnConsole() {}

func launchedWithoutSharedConsole() bool {
	info, err := os.Stdout.Stat()
	if err != nil {
		return true
	}
	return info.Mode()&os.ModeCharDevice == 0
}
