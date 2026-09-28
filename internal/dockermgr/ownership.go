package dockermgr

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
)

type OwnedResource struct {
	Name     string            `json:"name"`
	Labels   map[string]string `json:"labels"`
	DaemonID string            `json:"daemon_id,omitempty"`
}

type OwnedResourceRemover interface {
	RemoveOwnedAndVerify(context.Context, OwnedResource) error
	InspectOwned(context.Context, OwnedResource) (bool, error)
	DaemonID(context.Context) (string, error)
}

var fullContainerID = regexp.MustCompile(`^[a-f0-9]{64}$`)

func ownedContainer(ctx context.Context, owner OwnedResource) (string, string, error) {
	daemon, err := New().DaemonID(ctx)
	if err != nil {
		return "", "", fmt.Errorf("Docker daemon ownership cannot be verified: %w", err)
	}
	if owner.DaemonID == "" || daemon != owner.DaemonID {
		return "", "", fmt.Errorf("Docker daemon identity differs from the durable execution owner")
	}
	args := []string{"ps", "-a", "--no-trunc"}
	for key, value := range owner.Labels {
		args = append(args, "--filter", "label="+key+"="+value)
	}
	args = append(args, "--format", "{{.ID}} {{.Names}}")
	out, err := exec.CommandContext(ctx, "docker", args...).Output()
	if err != nil {
		return "", "", fmt.Errorf("inventory owned container: %w", err)
	}
	var id, name string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 || !fullContainerID.MatchString(fields[0]) {
			return "", "", fmt.Errorf("invalid container inventory")
		}
		if id != "" {
			return "", "", fmt.Errorf("multiple containers claim the same execution ownership")
		}
		id, name = fields[0], fields[1]
	}
	return id, name, nil
}

func inspectOwned(ctx context.Context, owner OwnedResource) (string, bool, error) {
	if owner.Name == "" || len(owner.Labels) == 0 {
		return "", false, fmt.Errorf("container ownership is required")
	}
	id, name, err := ownedContainer(ctx, owner)
	if err != nil || id == "" {
		return id, false, err
	}
	out, err := exec.CommandContext(ctx, "docker", "inspect", id).Output()
	if err != nil {
		return "", false, fmt.Errorf("inspect owned container: %w", err)
	}
	var containers []struct {
		ID     string `json:"Id"`
		Name   string `json:"Name"`
		Config struct {
			Labels map[string]string `json:"Labels"`
		} `json:"Config"`
		State *struct {
			Running *bool `json:"Running"`
		} `json:"State"`
	}
	if json.Unmarshal(out, &containers) != nil || len(containers) != 1 ||
		containers[0].ID != id || strings.TrimPrefix(containers[0].Name, "/") != name ||
		containers[0].State == nil || containers[0].State.Running == nil {
		return "", false, fmt.Errorf("container ownership inspection is inconsistent")
	}
	for key, value := range owner.Labels {
		if containers[0].Config.Labels[key] != value {
			return "", false, fmt.Errorf("container ownership mismatch for %s", owner.Name)
		}
	}
	return id, *containers[0].State.Running, nil
}

func (*Manager) InspectOwned(ctx context.Context, owner OwnedResource) (bool, error) {
	_, running, err := inspectOwned(ctx, owner)
	return running, err
}

func (m *Manager) RemoveOwnedAndVerify(ctx context.Context, owner OwnedResource) error {
	id, _, err := inspectOwned(ctx, owner)
	if err != nil || id == "" {
		return err
	}
	if err := m.Remove(ctx, id); err != nil {
		return fmt.Errorf("remove owned container: %w", err)
	}
	if remaining, _, err := ownedContainer(ctx, owner); err != nil {
		return err
	} else if remaining != "" {
		return fmt.Errorf("owned container name remains occupied after removal")
	}
	return nil
}
