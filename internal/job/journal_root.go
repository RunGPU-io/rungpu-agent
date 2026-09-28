package job

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type journalRoot struct {
	Version int               `json:"version"`
	ID      string            `json:"id"`
	Scopes  map[string]string `json:"scopes"`
}

func PrepareJournalRoot(cacheDir, expected string) (string, error) {
	root, release, err := openJournalRoot(cacheDir, expected)
	if err != nil {
		return "", err
	}
	defer release()
	return root.ID, nil
}

func openJournalRoot(cacheDir, expected string) (journalRoot, func(), error) {
	var root journalRoot
	if strings.TrimSpace(cacheDir) == "" {
		return root, nil, fmt.Errorf("a model cache directory is required for durable execution ownership")
	}
	dir := filepath.Join(cacheDir, "executions")
	path := filepath.Join(dir, "machine.json")
	if expected != "" {
		if _, err := os.Lstat(path); err != nil {
			return root, nil, fmt.Errorf("pinned execution journal is missing: %w", err)
		}
	}
	if err := privateDirectory(dir); err != nil {
		return root, nil, err
	}
	file, err := openJournalLockFile(filepath.Join(dir, ".identity.lock"))
	if err != nil {
		return root, nil, err
	}
	release := func() { _ = file.Close() }
	info, statErr := os.Lstat(filepath.Join(dir, ".identity.lock"))
	opened, openErr := file.Stat()
	if statErr != nil || openErr != nil || validatePrivateJournalFile(filepath.Join(dir, ".identity.lock"), info) != nil || !os.SameFile(info, opened) {
		release()
		return root, nil, fmt.Errorf("journal root lock has unsafe ownership")
	}
	if err := lockRootFile(file); err != nil {
		release()
		return root, nil, err
	}
	err = readCheckedJSON(path, &root)
	if os.IsNotExist(err) && expected == "" {
		var id [32]byte
		if _, err = rand.Read(id[:]); err == nil {
			root = journalRoot{Version: 1, ID: hex.EncodeToString(id[:]), Scopes: make(map[string]string)}
			err = atomicCheckedJSON(dir, "machine.json", root)
		}
	}
	if err != nil {
		release()
		return root, nil, err
	}
	decoded, decodeErr := hex.DecodeString(root.ID)
	if root.Version != 1 || decodeErr != nil || len(decoded) != 32 || root.Scopes == nil ||
		(expected != "" && expected != root.ID) {
		release()
		return root, nil, fmt.Errorf("execution journal root identity does not match the agent configuration")
	}
	for key, owner := range root.Scopes {
		scopeKey, keyErr := hex.DecodeString(key)
		scopeOwner, ownerErr := hex.DecodeString(owner)
		var identity journalIdentity
		if keyErr != nil || ownerErr != nil || len(scopeKey) != 32 || len(scopeOwner) != 32 {
			release()
			return root, nil, fmt.Errorf("invalid registered journal scope")
		}
		if err := readCheckedJSON(filepath.Join(dir, key, "identity.json"), &identity); err != nil ||
			identity.Owner != owner || attemptKey(identity.GPUID, "") != key || !identity.Initialized {
			release()
			return root, nil, fmt.Errorf("registered GPU journal is missing or has changed ownership: %v", err)
		}
	}
	return root, release, nil
}
