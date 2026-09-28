package dockermgr

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOwnedRemovalRequiresExactIdentityAndUsesContainerID(t *testing.T) {
	for _, scenario := range []string{"owned", "absent", "foreign", "daemon-error", "malformed", "remaining", "wrong-id", "wrong-daemon", "renamed", "duplicate"} {
		t.Run(scenario, func(t *testing.T) {
			dir := t.TempDir()
			removed := filepath.Join(dir, "removed")
			id := strings.Repeat("a", 64)
			owner := OwnedResource{Name: "tokenize-attempt-owned", Labels: map[string]string{"ai.rungpu.owner": "expected"}, DaemonID: "fixture-daemon"}
			labels := map[string]string{"ai.rungpu.owner": "expected"}
			if scenario == "foreign" {
				labels["ai.rungpu.owner"] = "another-agent"
			}
			inspectID := id
			name := owner.Name
			if scenario == "renamed" {
				name = "renamed-owned-resource"
			}
			if scenario == "wrong-id" {
				inspectID = strings.Repeat("b", 64)
			}
			inspect, _ := json.Marshal([]interface{}{map[string]interface{}{
				"Id": inspectID, "Name": "/" + name, "Config": map[string]interface{}{"Labels": labels},
				"State": map[string]interface{}{"Running": true},
			}})
			t.Setenv("FIXTURE_CASE", scenario)
			t.Setenv("FIXTURE_ID", id)
			t.Setenv("FIXTURE_NAME", name)
			t.Setenv("FIXTURE_INSPECT", string(inspect))
			t.Setenv("FIXTURE_REMOVED", removed)
			script := `#!/bin/sh
case "$1" in
info)
  if [ "$FIXTURE_CASE" = wrong-daemon ]; then printf 'other-daemon\n'; else printf 'fixture-daemon\n'; fi
  ;;
ps)
  if [ "$FIXTURE_CASE" = daemon-error ]; then exit 1; fi
  if [ "$FIXTURE_CASE" = malformed ]; then printf 'invalid inventory\n'; exit 0; fi
  if [ "$FIXTURE_CASE" = absent ]; then exit 0; fi
  if [ -f "$FIXTURE_REMOVED" ] && [ "$FIXTURE_CASE" != remaining ]; then exit 0; fi
  printf '%s %s\n' "$FIXTURE_ID" "$FIXTURE_NAME"
  if [ "$FIXTURE_CASE" = duplicate ]; then printf '%s duplicate\n' "$FIXTURE_ID"; fi
  ;;
inspect) printf '%s\n' "$FIXTURE_INSPECT";;
rm) printf '%s' "$3" > "$FIXTURE_REMOVED";;
*) exit 2;;
esac
`
			if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir)
			err := New().RemoveOwnedAndVerify(context.Background(), owner)
			wantSuccess := scenario == "owned" || scenario == "absent" || scenario == "renamed"
			if (err == nil) != wantSuccess {
				t.Fatalf("unexpected ownership verification outcome: %v", err)
			}
			data, readErr := os.ReadFile(removed)
			if scenario == "owned" || scenario == "remaining" || scenario == "renamed" {
				if readErr != nil || string(data) != id {
					t.Fatalf("removal did not use verified immutable ID: %q %v", data, readErr)
				}
			} else if !os.IsNotExist(readErr) {
				t.Fatal("unowned/unverified resource was touched")
			}
		})
	}
}

func TestOwnedRunArgumentsCarryOwnershipLabels(t *testing.T) {
	args := New().buildRunArgs(RunOptions{
		Image: "worker", Name: "owned", Network: "none",
		Labels: map[string]string{"ai.rungpu.owner": "owner", "ai.rungpu.attempt": "attempt"},
	})
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--label ai.rungpu.owner=owner") ||
		!strings.Contains(joined, "--label ai.rungpu.attempt=attempt") {
		t.Fatalf("ownership labels missing from Docker creation: %v", args)
	}
}
