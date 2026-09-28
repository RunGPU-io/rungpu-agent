//go:build windows

package job

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/RunGPU-io/rungpu-agent/internal/durablefs"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func setTestJournalDACL(t *testing.T, path, sddl string, protected bool) {
	t.Helper()
	if err := changeJournalTestDACL(path, sddl, protected); err != nil {
		t.Fatal(err)
	}
}

func changeJournalTestDACL(path, sddl string, protected bool) error {
	text, err := syscall.UTF16PtrFromString(sddl)
	if err != nil {
		return err
	}
	var descriptor unsafe.Pointer
	ok, _, err := convertJournalSD.Call(uintptr(unsafe.Pointer(text)), 1, uintptr(unsafe.Pointer(&descriptor)), 0)
	if ok == 0 {
		return err
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(descriptor)))
	acl, err := journalDescriptorDACL(descriptor)
	if err != nil {
		return err
	}
	name, err := durablefs.WindowsPath(path)
	if err != nil {
		return err
	}
	flags := uintptr(4 | 0x80000000)
	if !protected {
		flags = 4 | 0x20000000
	}
	code, _, _ := journalAdvapi.NewProc("SetNamedSecurityInfoW").Call(
		uintptr(unsafe.Pointer(name)), 1, flags, 0, 0, uintptr(acl), 0)
	if code != 0 {
		return syscall.Errno(code)
	}
	return nil
}

func setJournalDirectoryWritable(path string, writable bool) error {
	sid, err := journalUserSID()
	if err != nil {
		return err
	}
	rights := "FR"
	if writable {
		rights = "FA"
	}
	return changeJournalTestDACL(path, "D:P(A;;"+rights+";;;"+sid+")", true)
}

func makeJournalFilePublic(path string) error {
	sid, err := journalUserSID()
	if err != nil {
		return err
	}
	return changeJournalTestDACL(path, "D:P(A;;FA;;;"+sid+")(A;;FR;;;WD)", true)
}

func TestWindowsJournalPrivateStorageRoundTrip(t *testing.T) {
	cache := filepath.Join(t.TempDir(), strings.Repeat("a", 80), strings.Repeat("b", 80))
	j, err := OpenAttemptJournal(cache, "gpu-windows")
	if err != nil {
		t.Fatal(err)
	}
	a := types.JobAssignment{JobID: "job", DispatchToken: "A", Runtime: "workspace"}
	if _, err := j.Admit(a, "cuda"); err != nil {
		t.Fatal(err)
	}
	if _, err := j.RememberCancellation(a.JobID, a.DispatchToken, true); err != nil {
		t.Fatal(err)
	}
	for _, directory := range []string{filepath.Dir(j.dir), j.dir} {
		if err := validateWindowsJournalPath(directory, true); err != nil {
			t.Fatal(err)
		}
		entries, err := os.ReadDir(directory)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				if err := validateWindowsJournalPath(filepath.Join(directory, entry.Name()), false); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	j.Close()
	again, err := OpenAttemptJournal(cache, "gpu-windows")
	if err != nil {
		t.Fatal(err)
	}
	defer again.Close()
	record, exists := again.Find(a.JobID, a.DispatchToken)
	if !exists || !record.KnownUnstarted || !record.CancelRequested {
		t.Fatal("Windows durable journal lost cancellation proof")
	}
}

func TestWindowsJournalRejectsUnsafeDACLs(t *testing.T) {
	for _, scenario := range []string{"public-file", "public-directory", "inherited", "read-only-owner"} {
		t.Run(scenario, func(t *testing.T) {
			cache := t.TempDir()
			j, err := OpenAttemptJournal(cache, "gpu")
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(j.dir, "identity.json")
			if scenario == "public-directory" {
				path = j.dir
			}
			j.Close()
			sid, err := journalUserSID()
			if err != nil {
				t.Fatal(err)
			}
			private := "D:P(A;;FA;;;" + sid + ")"
			unsafeDACL := private + "(A;;FR;;;WD)"
			if scenario == "inherited" {
				unsafeDACL = private
			} else if scenario == "read-only-owner" {
				unsafeDACL = "D:P(A;;FR;;;" + sid + ")"
			}
			setTestJournalDACL(t, path, unsafeDACL, scenario != "inherited")
			defer setTestJournalDACL(t, path, private, true)
			if opened, err := OpenAttemptJournal(cache, "gpu"); err == nil {
				opened.Close()
				t.Fatal("unsafe Windows journal DACL admitted recovery")
			}
		})
	}
}

func TestWindowsJournalRefusesInheritedDirectoryWithoutRepair(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	sid, err := journalUserSID()
	if err != nil {
		t.Fatal(err)
	}
	setTestJournalDACL(t, path, "D:(A;;FA;;;"+sid+")(A;;FR;;;WD)", false)
	if err := privateDirectory(path); err == nil {
		t.Fatal("existing inherited/public directory was silently trusted")
	}
	if err := validateWindowsJournalPath(path, true); err == nil {
		t.Fatal("existing unsafe directory was silently repaired")
	}
}
