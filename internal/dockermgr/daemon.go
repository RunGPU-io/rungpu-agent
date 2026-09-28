package dockermgr

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

func (*Manager) DaemonID(ctx context.Context) (string, error) {
	out, err := exec.CommandContext(ctx, "docker", "info", "--format", "{{.ID}}").Output()
	id := strings.TrimSpace(string(out))
	if err != nil || id == "" || len(id) > 256 || strings.ContainsAny(id, "\r\n\t ") {
		return "", fmt.Errorf("Docker daemon identity is unavailable: %v", err)
	}
	return id, nil
}
