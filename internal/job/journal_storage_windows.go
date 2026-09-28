//go:build windows

package job

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"github.com/RunGPU-io/rungpu-agent/internal/durablefs"
)

var (
	journalAdvapi        = syscall.NewLazyDLL("advapi32.dll")
	journalKernel        = syscall.NewLazyDLL("kernel32.dll")
	convertJournalSD     = journalAdvapi.NewProc("ConvertStringSecurityDescriptorToSecurityDescriptorW")
	getJournalSecurity   = journalAdvapi.NewProc("GetSecurityInfo")
	getJournalDACL       = journalAdvapi.NewProc("GetSecurityDescriptorDacl")
	getJournalControl    = journalAdvapi.NewProc("GetSecurityDescriptorControl")
	getJournalACE        = journalAdvapi.NewProc("GetAce")
	createJournalDir     = journalKernel.NewProc("CreateDirectoryW")
	getJournalDriveType  = journalKernel.NewProc("GetDriveTypeW")
	getJournalVolumeInfo = journalKernel.NewProc("GetVolumeInformationW")
)

type journalACL struct {
	revision byte
	reserved byte
	size     uint16
	count    uint16
	padding  uint16
}

func journalUserSID() (string, error) {
	process, err := syscall.GetCurrentProcess()
	if err != nil {
		return "", err
	}
	var token syscall.Token
	if err := syscall.OpenProcessToken(process, syscall.TOKEN_QUERY, &token); err != nil {
		return "", err
	}
	defer token.Close()
	user, err := token.GetTokenUser()
	if err != nil {
		return "", err
	}
	return user.User.Sid.String()
}

func journalSecurityDescriptor() (unsafe.Pointer, string, error) {
	sid, err := journalUserSID()
	if err != nil {
		return nil, "", err
	}

	sddl, err := syscall.UTF16PtrFromString("O:" + sid + "D:P(A;;FA;;;" + sid + ")")
	if err != nil {
		return nil, "", err
	}
	var descriptor unsafe.Pointer
	ok, _, err := convertJournalSD.Call(uintptr(unsafe.Pointer(sddl)), 1, uintptr(unsafe.Pointer(&descriptor)), 0)
	if ok == 0 {
		return nil, "", err
	}
	return descriptor, sid, nil
}

func journalDescriptorDACL(descriptor unsafe.Pointer) (unsafe.Pointer, error) {
	var present, defaulted uint32
	var acl unsafe.Pointer
	ok, _, err := getJournalDACL.Call(uintptr(descriptor), uintptr(unsafe.Pointer(&present)),
		uintptr(unsafe.Pointer(&acl)), uintptr(unsafe.Pointer(&defaulted)))
	if ok == 0 {
		return nil, err
	}
	if present == 0 || acl == nil {
		return nil, fmt.Errorf("journal requires a non-null DACL")
	}
	return acl, nil
}

func journalSingleACE(acl unsafe.Pointer) ([]byte, error) {
	if acl == nil || (*journalACL)(acl).count != 1 {
		return nil, fmt.Errorf("journal DACL must grant access only to its owner")
	}
	var ace unsafe.Pointer
	ok, _, err := getJournalACE.Call(uintptr(acl), 0, uintptr(unsafe.Pointer(&ace)))
	if ok == 0 {
		return nil, err
	}
	size := *(*uint16)(unsafe.Add(ace, 2))
	if size < 8 || size > (*journalACL)(acl).size {
		return nil, fmt.Errorf("invalid journal access-control entry")
	}
	return unsafe.Slice((*byte)(ace), int(size)), nil
}

func validateWindowsJournalPath(path string, directory bool) error {
	name, err := durablefs.WindowsPath(path)
	if err != nil {
		return err
	}
	handle, err := syscall.CreateFile(name, 0x20000, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.OPEN_EXISTING, 0x02000000|0x00200000, 0)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(handle)
	var info syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(handle, &info); err != nil {
		return err
	}
	if info.FileAttributes&0x400 != 0 || (info.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0) != directory {
		return fmt.Errorf("journal path has unsafe type or is a reparse point: %s", path)
	}
	var owner, acl, descriptor unsafe.Pointer
	code, _, _ := getJournalSecurity.Call(uintptr(handle), 1, 1|4,
		uintptr(unsafe.Pointer(&owner)), 0, uintptr(unsafe.Pointer(&acl)), 0, uintptr(unsafe.Pointer(&descriptor)))
	if code != 0 {
		return syscall.Errno(code)
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(descriptor)))
	expected, sid, err := journalSecurityDescriptor()
	if err != nil {
		return err
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(expected)))
	if owner == nil {
		return fmt.Errorf("journal has no owner")
	}
	actualOwner, err := (*syscall.SID)(owner).String()
	if err != nil || actualOwner != sid {
		return fmt.Errorf("journal belongs to another Windows account: %s", path)
	}
	var control uint16
	var revision uint32
	ok, _, err := getJournalControl.Call(uintptr(descriptor), uintptr(unsafe.Pointer(&control)), uintptr(unsafe.Pointer(&revision)))
	if ok == 0 {
		return err
	}
	if control&0x1000 == 0 {
		return fmt.Errorf("journal DACL must be protected from inheritance: %s", path)
	}
	expectedACL, err := journalDescriptorDACL(expected)
	if err != nil {
		return err
	}
	expectedACE, err := journalSingleACE(expectedACL)
	if err != nil {
		return err
	}
	actualACE, err := journalSingleACE(acl)
	if err != nil {
		return err
	}
	if !bytes.Equal(actualACE, expectedACE) {
		return fmt.Errorf("journal DACL is not private to its owner: %s", path)
	}
	return nil
}

func validateJournalVolume(path string) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	volume := strings.TrimPrefix(filepath.VolumeName(absolute), `\\?\`)
	if len(volume) != 2 || volume[1] != ':' {
		return fmt.Errorf("execution journal requires a local Windows volume")
	}
	var ancestors []string
	for current := absolute; ; current = filepath.Dir(current) {
		ancestors = append(ancestors, current)
		if current == filepath.Dir(current) {
			break
		}
	}
	for i := len(ancestors) - 1; i >= 0; i-- {
		current := ancestors[i]
		name, err := durablefs.WindowsPath(current)
		if err != nil {
			return err
		}
		attributes, err := syscall.GetFileAttributes(name)
		if err == nil && attributes&0x400 != 0 {
			return fmt.Errorf("execution journal path contains a reparse point: %s", current)
		}
		if err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	root, err := syscall.UTF16PtrFromString(volume + `\`)
	if err != nil {
		return err
	}
	kind, _, _ := getJournalDriveType.Call(uintptr(unsafe.Pointer(root)))
	if kind != 2 && kind != 3 {
		return fmt.Errorf("execution journal requires a local fixed or removable Windows drive")
	}
	var flags uint32
	ok, _, err := getJournalVolumeInfo.Call(uintptr(unsafe.Pointer(root)), 0, 0, 0, 0,
		uintptr(unsafe.Pointer(&flags)), 0, 0)
	if ok == 0 {
		return err
	}
	if flags&8 == 0 {
		return fmt.Errorf("execution journal volume does not support persistent Windows ACLs")
	}
	return nil
}

func privateDirectory(path string) error {
	if err := validateJournalVolume(path); err != nil {
		return err
	}
	if _, err := os.Lstat(path); err == nil {
		return validateWindowsJournalPath(path, true)
	} else if !os.IsNotExist(err) {
		return err
	}
	parent := filepath.Dir(path)
	if _, err := os.Lstat(parent); os.IsNotExist(err) {
		if err := privateDirectory(parent); err != nil {
			return err
		}
	} else if err != nil {
		return err
	}
	descriptor, _, err := journalSecurityDescriptor()
	if err != nil {
		return err
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(descriptor)))
	attributes := syscall.SecurityAttributes{Length: uint32(unsafe.Sizeof(syscall.SecurityAttributes{})), SecurityDescriptor: uintptr(descriptor)}
	var random [16]byte
	if _, err := rand.Read(random[:]); err != nil {
		return err
	}
	staged := filepath.Join(parent, fmt.Sprintf(".stage-dir-%x", random))
	name, err := durablefs.WindowsPath(staged)
	if err != nil {
		return err
	}
	ok, _, err := createJournalDir.Call(uintptr(unsafe.Pointer(name)), uintptr(unsafe.Pointer(&attributes)))
	if ok == 0 {
		return err
	}
	defer os.Remove(staged)
	if err := validateWindowsJournalPath(staged, true); err != nil {
		return err
	}
	if err := durablefs.MoveNew(staged, path); err != nil {
		if !os.IsExist(err) {
			return err
		}
	}
	return validateWindowsJournalPath(path, true)
}

func validatePrivateJournalFile(path string, info os.FileInfo) error {
	if !info.Mode().IsRegular() {
		return fmt.Errorf("journal file has unsafe type: %s", path)
	}
	return validateWindowsJournalPath(path, false)
}

func createPrivateJournalFile(path string, disposition uint32) (*os.File, error) {
	descriptor, _, err := journalSecurityDescriptor()
	if err != nil {
		return nil, err
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(descriptor)))
	attributes := syscall.SecurityAttributes{Length: uint32(unsafe.Sizeof(syscall.SecurityAttributes{})), SecurityDescriptor: uintptr(descriptor)}
	name, err := durablefs.WindowsPath(path)
	if err != nil {
		return nil, err
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ|syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE, &attributes, disposition, syscall.FILE_ATTRIBUTE_NORMAL|0x00200000, 0)
	if err != nil {
		return nil, err
	}
	file := os.NewFile(uintptr(handle), path)
	if err := validateWindowsJournalPath(path, false); err != nil {
		file.Close()
		return nil, err
	}
	return file, nil
}

func openJournalLockFile(path string) (*os.File, error) {
	return createPrivateJournalFile(path, syscall.OPEN_ALWAYS)
}

func createJournalStage(dir string) (*os.File, error) {
	var random [16]byte
	if _, err := rand.Read(random[:]); err != nil {
		return nil, err
	}
	return createPrivateJournalFile(filepath.Join(dir, fmt.Sprintf(".stage-%x", random)), syscall.CREATE_NEW)
}
