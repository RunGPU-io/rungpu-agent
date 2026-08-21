package job

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func (r *customDockerRuntime) execPythonClient(ctx context.Context, name, outputDir, filename, script string, timeout time.Duration) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create speech output dir: %w", err)
	}
	clientPath := filepath.Join(outputDir, filename)
	if err := os.WriteFile(clientPath, []byte(script), 0o644); err != nil {
		return fmt.Errorf("write speech client: %w", err)
	}
	defer os.Remove(clientPath)

	readyTimeout := 3 * time.Minute
	if timeout > 0 && timeout < readyTimeout {
		readyTimeout = timeout
	}
	speechTimeout := timeout - readyTimeout
	if speechTimeout < 30*time.Second {
		speechTimeout = 30 * time.Second
	}

	execCtx := ctx
	cancel := func() {}
	if timeout > 0 {
		execCtx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()

	args := []string{
		"python", "/output/" + filename,
		strconv.Itoa(int(readyTimeout.Seconds())),
		strconv.Itoa(int(speechTimeout.Seconds())),
	}
	if _, err := r.docker.Exec(execCtx, name, args); err != nil {
		if _, fallbackErr := r.docker.Exec(execCtx, name, append([]string{"python3"}, args[1:]...)); fallbackErr != nil {
			logs, _ := r.docker.Logs(ctx, name, 200)
			return fmt.Errorf("speech runtime failed: %v\n%s", err, strings.TrimSpace(logs))
		}
	}
	return nil
}

func (r *customDockerRuntime) collectSpeechOutput(img, outputDir string) (map[string]interface{}, error) {
	parsed := map[string]interface{}{
		"status":  "completed",
		"backend": "docker-custom",
		"image":   img,
	}
	if outputFile := findOutputFile(outputDir); outputFile != "" {
		parsed["output_file"] = outputFile
		return parsed, nil
	}
	return nil, fmt.Errorf("speech runtime produced no audio under /output")
}
