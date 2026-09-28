//go:build !darwin && !linux && !freebsd && !openbsd && !netbsd && !dragonfly && !windows

package job

import (
	"fmt"
	"os"
)

func lockJournalFile(*os.File) error {
	return fmt.Errorf("durable execution ownership locking is unsupported on this operating system")
}

func lockMachineFile(file *os.File, _ bool) error { return lockJournalFile(file) }
func lockRootFile(file *os.File) error            { return lockJournalFile(file) }
