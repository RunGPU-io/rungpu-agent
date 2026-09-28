package dockermgr

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRemoveAndVerifyRequiresDaemonConfirmedAbsence(t *testing.T) {
	for _, mode := range []string{"absent", "remove", "list-error", "remove-error", "remains", "verify-error", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			script := `#!/bin/sh
case "$1" in
ps)
  [ "$FAKE_MODE" = "timeout" ] && exec sleep 5
  [ "$FAKE_MODE" = "list-error" ] && exit 1
  [ "$FAKE_MODE" = "absent" ] && exit 0
  if [ -f "$FAKE_STATE" ]; then
    [ "$FAKE_MODE" = "verify-error" ] && exit 1
    [ "$FAKE_MODE" != "remains" ] && exit 0
  fi
  printf 'target\n'
  ;;
rm)
  [ "$FAKE_MODE" = "remove-error" ] && exit 1
  : > "$FAKE_STATE"
  ;;
*) exit 1 ;;
esac
`
			if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
			t.Setenv("FAKE_MODE", mode)
			t.Setenv("FAKE_STATE", filepath.Join(dir, "removed"))
			timeout := 5 * time.Second
			if mode == "timeout" {
				timeout = 100 * time.Millisecond
			}
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			err := New().RemoveAndVerify(ctx, "target")
			wantSuccess := mode == "absent" || mode == "remove"
			if (err == nil) != wantSuccess {
				t.Fatalf("mode %s: unexpected verification result %v", mode, err)
			}
			if mode == "remains" && !strings.Contains(err.Error(), "still exists") {
				t.Fatalf("remaining container not reported: %v", err)
			}
		})
	}
}
