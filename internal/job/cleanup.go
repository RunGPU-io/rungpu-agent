package job

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var allowedCleanupCategories = map[string]bool{
	"ollama_models":    true,
	"docker_images":    true,
	"container_models": true,
	"custom_assets":    true,
	"job_files":        true,
}

type CleanupPreview struct {
	Categories map[string]CleanupCategory `json:"categories"`
	TotalBytes int64                      `json:"total_bytes"`
	ActiveJobs int                        `json:"active_jobs"`
}

type CleanupCategory struct {
	Items int   `json:"items"`
	Bytes int64 `json:"bytes"`
}

func managedMarkerPath(cacheDir, category, value string) string {
	digest := sha256.Sum256([]byte(value))
	return filepath.Join(cacheDir, "managed", category, fmt.Sprintf("%x.json", digest))
}

func trackManagedAsset(cacheDir, category, value string) error {
	path := managedMarkerPath(cacheDir, category, value)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, _ := json.Marshal(map[string]string{"value": value, "tracked_at": time.Now().UTC().Format(time.RFC3339)})
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, data, 0o600); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func trackedAssets(cacheDir, category string) []string {
	entries, err := os.ReadDir(filepath.Join(cacheDir, "managed", category))
	if err != nil {
		return nil
	}
	values := []string{}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(cacheDir, "managed", category, entry.Name()))
		var marker struct {
			Value string `json:"value"`
		}
		if readErr == nil && json.Unmarshal(data, &marker) == nil && marker.Value != "" {
			values = append(values, marker.Value)
		}
	}
	return values
}

func fixedDirectorySize(path string) int64 {
	var total int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info.Mode().IsRegular() {
			total += info.Size()
		}
		return nil
	})
	return total
}

func normalizeCleanupCategories(categories []string) ([]string, error) {
	seen := map[string]bool{}
	result := []string{}
	for _, category := range categories {
		category = strings.TrimSpace(category)
		if !allowedCleanupCategories[category] {
			return nil, fmt.Errorf("unsupported cleanup category %q", category)
		}
		if !seen[category] {
			seen[category] = true
			result = append(result, category)
		}
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("at least one cleanup category is required")
	}
	return result, nil
}

func (e *Executor) PreviewCleanup(categories []string) (CleanupPreview, error) {
	categories, err := normalizeCleanupCategories(categories)
	if err != nil {
		return CleanupPreview{}, err
	}
	e.mu.Lock()
	activeJobs := len(e.inflight)
	e.mu.Unlock()
	preview := CleanupPreview{Categories: map[string]CleanupCategory{}, ActiveJobs: activeJobs}
	for _, category := range categories {
		item := CleanupCategory{}
		switch category {
		case "ollama_models":
			item.Items = len(trackedAssets(e.cacheDir, "ollama"))
		case "docker_images":
			item.Items = len(trackedAssets(e.cacheDir, "docker"))
		case "container_models":
			if exec.Command("docker", "volume", "inspect", "tokenize-model-cache").Run() == nil {
				item.Items = 1
			}
		case "custom_assets":
			item.Bytes = fixedDirectorySize(filepath.Join(e.cacheDir, "assets"))
			item.Items = countFiles(filepath.Join(e.cacheDir, "assets"))
		case "job_files":
			for _, name := range []string{"staging", "output"} {
				path := filepath.Join(e.cacheDir, name)
				item.Bytes += fixedDirectorySize(path)
				item.Items += countFiles(path)
			}
		}
		preview.Categories[category] = item
		preview.TotalBytes += item.Bytes
	}
	return preview, nil
}

func countFiles(path string) int {
	count := 0
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info.Mode().IsRegular() {
			count++
		}
		return nil
	})
	return count
}

func (e *Executor) ExecuteCleanup(ctx context.Context, categories []string) (CleanupPreview, error) {
	preview, err := e.PreviewCleanup(categories)
	if err != nil {
		return CleanupPreview{}, err
	}
	if preview.ActiveJobs > 0 {
		return CleanupPreview{}, fmt.Errorf("cleanup blocked while jobs are active")
	}
	for _, category := range categories {
		switch category {
		case "ollama_models":
			for _, model := range trackedAssets(e.cacheDir, "ollama") {
				if output, removeErr := exec.CommandContext(ctx, "ollama", "rm", model).CombinedOutput(); removeErr != nil {
					return CleanupPreview{}, fmt.Errorf("remove Ollama model %s: %v: %s", model, removeErr, strings.TrimSpace(string(output)))
				}
				_ = os.Remove(managedMarkerPath(e.cacheDir, "ollama", model))
			}
		case "docker_images":
			for _, image := range trackedAssets(e.cacheDir, "docker") {
				if output, removeErr := exec.CommandContext(ctx, "docker", "image", "rm", image).CombinedOutput(); removeErr != nil {
					return CleanupPreview{}, fmt.Errorf("remove Docker image %s: %v: %s", image, removeErr, strings.TrimSpace(string(output)))
				}
				_ = os.Remove(managedMarkerPath(e.cacheDir, "docker", image))
			}
		case "container_models":
			if exec.CommandContext(ctx, "docker", "volume", "inspect", "tokenize-model-cache").Run() == nil {
				if output, removeErr := exec.CommandContext(ctx, "docker", "volume", "rm", "tokenize-model-cache").CombinedOutput(); removeErr != nil {
					return CleanupPreview{}, fmt.Errorf("remove batch model cache volume: %v: %s", removeErr, strings.TrimSpace(string(output)))
				}
			}
		case "custom_assets":
			if err := os.RemoveAll(filepath.Join(e.cacheDir, "assets")); err != nil {
				return CleanupPreview{}, err
			}
		case "job_files":
			for _, name := range []string{"staging", "output"} {
				if err := os.RemoveAll(filepath.Join(e.cacheDir, name)); err != nil {
					return CleanupPreview{}, err
				}
			}
		}
	}
	return preview, nil
}
