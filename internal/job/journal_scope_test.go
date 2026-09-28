package job

import (
	"os"
	"sync"
	"testing"
)

func TestJournalRegisteredGPUDirectoryCannotDisappearSilently(t *testing.T) {
	cache, docker := t.TempDir(), &journalDocker{}
	original := newJournalExecutor(t, cache, "gpu-old", &sync.RWMutex{}, docker)
	dir := original.journal.dir
	original.journal.Close()
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := NewExecutorWithOptions(ExecutorOptions{
		CacheDir: cache, GPUID: "gpu-other", Backend: "cpu", Runtime: docker, OwnedTeardown: docker,
		ExecutionGate: &sync.RWMutex{},
	}); err == nil {
		t.Fatal("a sibling admitted work after a registered GPU journal disappeared")
	}
}
