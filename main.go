// Command rungpu-agent is the cross-platform GPU agent that contributes a
// host's GPU(s) to the RunGPU pool. It connects outbound over WebSocket,
// registers, heartbeats, and runs inference jobs in Docker.
//
// Usage:
//
//	rungpu-agent init --api-key KEY [--config PATH]
//	rungpu-agent start [--config PATH]
//	rungpu-agent status [--config PATH]
//	rungpu-agent cleanup [--all] [--containers] [--volumes] [--images] [--cache] [--ollama] [--config-file] [--service] [--dry-run]
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/RunGPU-io/gpu-agent/internal/config"
	"github.com/RunGPU-io/gpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/gpu-agent/internal/gpu"
	"github.com/RunGPU-io/gpu-agent/internal/pool"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	var err error
	switch cmd {
	case "init":
		err = cmdInit(args)
	case "start":
		err = cmdStart(args)
	case "status":
		err = cmdStatus(args)
	case "cleanup":
		err = cmdCleanup(args)
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", cmd)
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Println(`rungpu-agent — RunGPU GPU pool agent

Commands:
  init     Create config (auto-detects GPUs)
  start    Connect to the pool and serve jobs
  status   Show detected GPUs and live metrics
  cleanup  Remove containers, volumes, images, caches, config, and services

Run "tokenize-gpu-agent <command> -h" for command flags.`)
}

// ═══════════════════════════════════════════════════════════════════════════════
// init
// ═══════════════════════════════════════════════════════════════════════════════

func cmdInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ExitOnError)
	apiKey := fs.String("api-key", "", "API key from your RunGPU account (required)")
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)

	if *apiKey == "" {
		return fmt.Errorf("--api-key is required\n\nGet your API key from: https://www.rungpu.io/marketplace/host")
	}

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║           RunGPU Agent — Setup & Diagnostics            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ── Step 1: Check prerequisites ─────────────────────────────────────
	fmt.Println("Step 1/4 — Checking prerequisites...")
	fmt.Println()

	issues := 0

	// Docker
	dockerOk := exec.Command("docker", "version").Run() == nil
	if dockerOk {
		fmt.Println("  ✅ Docker            installed")
	} else {
		fmt.Println("  ❌ Docker            not found")
		fmt.Println("     → Install: https://docs.docker.com/get-docker/")
		issues++
	}

	// NVIDIA GPU / nvidia-smi
	nvidiaOk := false
	if out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output(); err == nil {
		gpuName := strings.TrimSpace(string(out))
		if gpuName != "" {
			fmt.Printf("  ✅ NVIDIA GPU        %s\n", strings.Split(gpuName, "\n")[0])
			nvidiaOk = true
		}
	}
	if !nvidiaOk && runtime.GOOS == "darwin" {
		fmt.Println("  ℹ️  NVIDIA GPU        not available (Apple Silicon — Ollama will be used)")
	} else if !nvidiaOk {
		fmt.Println("  ⚠️  NVIDIA GPU        not detected (nvidia-smi not found)")
		fmt.Println("     → Install NVIDIA drivers: https://www.nvidia.com/drivers")
	}

	// NVIDIA Container Toolkit (Linux only)
	if runtime.GOOS == "linux" && dockerOk {
		nctOk := exec.Command("docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.0.0-base-ubuntu22.04", "nvidia-smi").Run() == nil
		if nctOk {
			fmt.Println("  ✅ NVIDIA Toolkit    Docker GPU access works")
		} else if nvidiaOk {
			fmt.Println("  ⚠️  NVIDIA Toolkit    Docker can't access GPU")
			fmt.Println("     → Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
		}
	}

	// Ollama
	ollamaOk := exec.Command("ollama", "--version").Run() == nil
	if ollamaOk {
		fmt.Println("  ✅ Ollama            installed (LLM inference)")
	} else {
		fmt.Println("  ℹ️  Ollama            not installed (optional — for LLM jobs)")
		fmt.Println("     → Install: https://ollama.com")
	}

	// Disk space
	home, _ := os.UserHomeDir()
	if stat, err := os.Stat(home); err == nil && stat.IsDir() {
		// Simple check: can we write to the cache dir?
		cacheDir := filepath.Join(home, ".tokenize", "models")
		os.MkdirAll(cacheDir, 0755)
		testFile := filepath.Join(cacheDir, ".write-test")
		if err := os.WriteFile(testFile, []byte("ok"), 0644); err == nil {
			os.Remove(testFile)
			fmt.Println("  ✅ Disk              writable")
		} else {
			fmt.Println("  ❌ Disk              cannot write to ~/.tokenize/")
			issues++
		}
	}

	fmt.Println()

	if issues > 0 {
		fmt.Printf("  ⚠️  %d issue(s) found. The agent may not work correctly.\n", issues)
		fmt.Println("     Fix the issues above, then re-run init.")
		fmt.Println()
	}

	// ── Step 2: Detect GPUs ─────────────────────────────────────────────
	fmt.Println("Step 2/4 — Detecting GPUs...")
	fmt.Println()

	monitor := gpu.NewMonitor()
	gpus := monitor.GPUs()
	backend := monitor.Backend()

	fmt.Printf("  Backend: %s\n", backend)
	fmt.Printf("  GPUs found: %d\n", len(gpus))
	for _, g := range gpus {
		vramGB := float64(g.MemoryMB) / 1024.0
		fmt.Printf("    GPU %d: %s (%.0f GB, %s)\n", g.Index, g.Name, vramGB, g.ComputeCapability)
	}
	fmt.Println()

	// ── Step 3: Create config ───────────────────────────────────────────
	fmt.Println("Step 3/4 — Creating configuration...")
	fmt.Println()

	cfg, err := config.New(*apiKey)
	if err != nil {
		return err
	}
	if err := config.Save(cfg, *cfgPath); err != nil {
		return err
	}
	fmt.Printf("  ✅ Config saved to: %s\n", *cfgPath)
	fmt.Printf("  ✅ Pool URL: %s\n", cfg.PoolURL)
	fmt.Printf("  ✅ API key: %s...%s\n", (*apiKey)[:6], (*apiKey)[len(*apiKey)-4:])
	fmt.Println()

	// ── Step 4: Next steps ──────────────────────────────────────────────
	fmt.Println("Step 4/4 — Ready!")
	fmt.Println()
	fmt.Println("  ┌─────────────────────────────────────────────────────┐")
	fmt.Println("  │  Your GPU agent is configured. Start it with:      │")
	fmt.Println("  │                                                     │")
	fmt.Println("  │    rungpu-agent start                               │")
	fmt.Println("  │                                                     │")
	fmt.Println("  │  The agent will connect to the pool, register your  │")
	fmt.Println("  │  GPU, and start accepting jobs automatically.       │")
	fmt.Println("  │                                                     │")
	fmt.Println("  │  Other commands:                                    │")
	fmt.Println("  │    rungpu-agent status    — check GPU & metrics     │")
	fmt.Println("  │    rungpu-agent cleanup   — remove all agent data   │")
	fmt.Println("  │                                                     │")
	fmt.Println("  │  Dashboard: https://www.rungpu.io/marketplace/host  │")
	fmt.Println("  └─────────────────────────────────────────────────────┘")
	fmt.Println()

	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// start
// ═══════════════════════════════════════════════════════════════════════════════

func cmdStart(args []string) error {
	fs := flag.NewFlagSet("start", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	if err := config.Validate(cfg); err != nil {
		return err
	}

	client, err := pool.NewClient(cfg)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Printf("Starting agent (pool: %s)...\n", cfg.PoolURL)
	if err := client.Run(ctx); err != nil && err != context.Canceled {
		return err
	}
	fmt.Println("Agent stopped.")
	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// status
// ═══════════════════════════════════════════════════════════════════════════════

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	_ = fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)

	monitor := gpu.NewMonitor()
	gpus := monitor.GPUs()
	backend := monitor.Backend()

	fmt.Println("\n=== RunGPU Agent Status ===")
	fmt.Printf("Backend: %s\n", backend)
	if backend != "cuda" {
		fmt.Println("  (non-NVIDIA host: jobs run natively via Ollama; install from https://ollama.com)")
	}
	fmt.Printf("Detected GPUs: %d\n", len(gpus))
	for _, g := range gpus {
		fmt.Printf("  GPU %d: %s (%d MB, %s)\n", g.Index, g.Name, g.MemoryMB, g.ComputeCapability)
	}

	fmt.Println("\nMetrics:")
	for _, m := range monitor.CollectMetrics() {
		temp := "n/a"
		if m.TemperatureC != nil {
			temp = fmt.Sprintf("%.0fC", *m.TemperatureC)
		}
		fmt.Printf("  GPU %d: %.0f%% util, %d/%d MB mem, %s\n",
			m.GPUIndex, m.UtilizationPercent, m.MemoryUsedMB, m.MemoryTotalMB, temp)
	}

	healthy := "Healthy"
	if !monitor.IsHealthy() {
		healthy = "Issues detected"
	}
	fmt.Printf("\nHealth: %s\n\n", healthy)
	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// cleanup — comprehensive cleanup for host agent users
//
// Cleans up everything the agent creates on the host machine:
//   - Docker containers (tokenize-*)
//   - Docker named volumes (tokenize-* model caches)
//   - Docker images (workspace/model images)
//   - Local file cache (staging, output, models)
//   - Ollama models
//   - Config file (~/.tokenize/)
//   - System service (launchd / systemd)
//   - Log files
// ═══════════════════════════════════════════════════════════════════════════════

func cmdCleanup(args []string) error {
	fs := flag.NewFlagSet("cleanup", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Println(`Usage: tokenize-gpu-agent cleanup [flags]

Removes resources created by the agent on this machine.
By default (no flags), shows what would be cleaned up without removing anything.

Flags:`)
		fs.PrintDefaults()
		fmt.Println(`
Examples:
  tokenize-gpu-agent cleanup                    # dry-run: show what exists
  tokenize-gpu-agent cleanup --containers       # remove stopped tokenize-* containers
  tokenize-gpu-agent cleanup --volumes          # remove tokenize-* Docker volumes (model caches)
  tokenize-gpu-agent cleanup --cache            # remove local file cache (~/.tokenize/models)
  tokenize-gpu-agent cleanup --all              # remove everything (full uninstall)
  tokenize-gpu-agent cleanup --all --dry-run    # show what --all would remove`)
	}

	all := fs.Bool("all", false, "remove everything (containers, volumes, images, cache, ollama, config, service, logs)")
	containers := fs.Bool("containers", false, "stop and remove all tokenize-* Docker containers")
	volumes := fs.Bool("volumes", false, "remove all tokenize-* Docker named volumes (model caches)")
	images := fs.Bool("images", false, "remove Docker images pulled by the agent")
	cache := fs.Bool("cache", false, "remove local file cache (staging, output, downloaded models)")
	ollama := fs.Bool("ollama", false, "remove Ollama models pulled by the agent")
	configFile := fs.Bool("config-file", false, "remove config directory (~/.tokenize/)")
	service := fs.Bool("service", false, "uninstall the system service (launchd/systemd)")
	logs := fs.Bool("logs", false, "remove agent log files")
	dryRun := fs.Bool("dry-run", false, "show what would be removed without actually removing")
	_ = fs.Parse(args)

	// If no specific flags, default to dry-run overview
	nothingSelected := !*all && !*containers && !*volumes && !*images && !*cache && !*ollama && !*configFile && !*service && !*logs
	if nothingSelected {
		*dryRun = true
		// Show everything
		*all = true
	}

	if *all {
		*containers = true
		*volumes = true
		*images = true
		*cache = true
		*ollama = true
		*configFile = true
		*service = true
		*logs = true
	}

	ctx := context.Background()
	docker := dockermgr.New()
	hasDocker := docker.Available(ctx)

	if *dryRun && nothingSelected {
		fmt.Println("\n=== RunGPU Agent — Cleanup Overview ===")
		fmt.Println("Showing what exists on this machine. Pass flags to remove.")
	} else if *dryRun {
		fmt.Println("\n=== Dry Run — nothing will be removed ===")
	} else {
		fmt.Println("\n=== RunGPU Agent — Cleanup ===")
	}

	totalCleaned := 0

	// ── 1. Docker containers ────────────────────────────────────────────
	if *containers && hasDocker {
		cleaned, err := cleanupContainers(ctx, docker, *dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: container cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 2. Docker named volumes ─────────────────────────────────────────
	if *volumes && hasDocker {
		cleaned, err := cleanupVolumes(ctx, docker, *dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: volume cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 3. Docker images ────────────────────────────────────────────────
	if *images && hasDocker {
		cleaned, err := cleanupImages(ctx, docker, *dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: image cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 4. Local file cache ─────────────────────────────────────────────
	if *cache {
		cleaned, err := cleanupCache(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: cache cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 5. Ollama models ────────────────────────────────────────────────
	if *ollama {
		cleaned, err := cleanupOllama(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: ollama cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 6. Config file ──────────────────────────────────────────────────
	if *configFile {
		cleaned, err := cleanupConfig(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: config cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 7. System service ───────────────────────────────────────────────
	if *service {
		cleaned, err := cleanupService(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: service cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── 8. Log files ────────────────────────────────────────────────────
	if *logs {
		cleaned, err := cleanupLogs(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: log cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	// ── Summary ─────────────────────────────────────────────────────────
	fmt.Println()
	if *dryRun && nothingSelected {
		fmt.Println("To remove specific resources:")
		fmt.Println("  tokenize-gpu-agent cleanup --containers       # Docker containers")
		fmt.Println("  tokenize-gpu-agent cleanup --volumes          # Docker volumes (model caches)")
		fmt.Println("  tokenize-gpu-agent cleanup --images           # Docker images")
		fmt.Println("  tokenize-gpu-agent cleanup --cache            # Local file cache")
		fmt.Println("  tokenize-gpu-agent cleanup --ollama           # Ollama models")
		fmt.Println("  tokenize-gpu-agent cleanup --config-file      # Config (~/.tokenize/)")
		fmt.Println("  tokenize-gpu-agent cleanup --service          # System service")
		fmt.Println("  tokenize-gpu-agent cleanup --logs             # Log files")
		fmt.Println("  tokenize-gpu-agent cleanup --all              # Everything (full uninstall)")
	} else if *dryRun {
		fmt.Printf("Dry run complete. %d item(s) would be removed.\n", totalCleaned)
		fmt.Println("Run without --dry-run to actually remove.")
	} else {
		fmt.Printf("Cleanup complete. %d item(s) removed.\n", totalCleaned)
	}
	fmt.Println()

	return nil
}

// ── Cleanup: Docker containers ──────────────────────────────────────────────

func cleanupContainers(ctx context.Context, docker *dockermgr.Manager, dryRun bool) (int, error) {
	containers, err := docker.ListTokenizeContainers(ctx)
	if err != nil {
		return 0, err
	}

	fmt.Printf("Docker containers (tokenize-*):\n")
	if len(containers) == 0 {
		fmt.Println("  (none)")
		return 0, nil
	}

	for _, c := range containers {
		id := c.ID
		if len(id) > 12 {
			id = id[:12]
		}
		fmt.Printf("  %s  %-40s  %s\n", id, c.Name, c.Status)
	}

	if dryRun {
		fmt.Printf("  → %d container(s) would be removed\n", len(containers))
		return len(containers), nil
	}

	removed, err := docker.RemoveAllTokenizeContainers(ctx)
	fmt.Printf("  → Removed %d container(s)\n", removed)
	return removed, err
}

// ── Cleanup: Docker named volumes ─────────��─────────────────────────────────

func cleanupVolumes(ctx context.Context, docker *dockermgr.Manager, dryRun bool) (int, error) {
	volumes, err := docker.ListTokenizeVolumes(ctx)
	if err != nil {
		return 0, err
	}

	fmt.Printf("Docker volumes (tokenize-*):\n")
	if len(volumes) == 0 {
		fmt.Println("  (none)")
		return 0, nil
	}

	for _, v := range volumes {
		fmt.Printf("  %s\n", v.Name)
	}

	if dryRun {
		fmt.Printf("  → %d volume(s) would be removed\n", len(volumes))
		return len(volumes), nil
	}

	// Must remove containers first before volumes
	docker.RemoveAllTokenizeContainers(ctx)

	removed := 0
	for _, v := range volumes {
		if err := docker.RemoveVolume(ctx, v.Name); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: could not remove volume %s: %v\n", v.Name, err)
		} else {
			removed++
		}
	}
	fmt.Printf("  → Removed %d volume(s)\n", removed)
	return removed, nil
}

// ── Cleanup: Docker images ──────────────────────────────────────────────────

func cleanupImages(ctx context.Context, docker *dockermgr.Manager, dryRun bool) (int, error) {
	images, err := docker.ListTokenizeImages(ctx)
	if err != nil {
		return 0, err
	}

	fmt.Printf("Docker images (agent-related):\n")
	if len(images) == 0 {
		fmt.Println("  (none)")
		return 0, nil
	}

	for _, img := range images {
		fmt.Printf("  %-60s  %s\n", img.Repository, img.Size)
	}

	if dryRun {
		fmt.Printf("  → %d image(s) would be removed\n", len(images))
		return len(images), nil
	}

	removed := 0
	for _, img := range images {
		if err := docker.RemoveImage(ctx, img.ID); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: could not remove image %s: %v\n", img.Repository, err)
		} else {
			removed++
		}
	}
	fmt.Printf("  → Removed %d image(s)\n", removed)
	return removed, nil
}

// ── Cleanup: Local file cache ───────────────────────────────────────────────

func cleanupCache(dryRun bool) (int, error) {
	home, _ := os.UserHomeDir()
	cacheDir := filepath.Join(home, ".tokenize", "models")
	stagingDir := filepath.Join(home, ".tokenize", "staging")
	outputDir := filepath.Join(home, ".tokenize", "output")
	workspacesDir := filepath.Join(home, ".tokenize", "workspaces")
	tmpDir := filepath.Join(os.TempDir(), "tokenize-localtest")

	dirs := []struct {
		path string
		desc string
	}{
		{cacheDir, "model cache"},
		{stagingDir, "job staging files"},
		{outputDir, "job output files"},
		{workspacesDir, "workspace data"},
		{tmpDir, "localtest temp files"},
	}

	fmt.Printf("Local file cache:\n")
	found := 0
	for _, d := range dirs {
		size, count := dirStats(d.path)
		if count > 0 {
			fmt.Printf("  %-50s  %d file(s), %s\n", d.path, count, humanSize(size))
			found++
		}
	}

	if found == 0 {
		fmt.Println("  (none)")
		return 0, nil
	}

	if dryRun {
		fmt.Printf("  → %d director(ies) would be removed\n", found)
		return found, nil
	}

	removed := 0
	for _, d := range dirs {
		if _, count := dirStats(d.path); count > 0 {
			if err := os.RemoveAll(d.path); err != nil {
				fmt.Fprintf(os.Stderr, "  Warning: could not remove %s: %v\n", d.path, err)
			} else {
				removed++
			}
		}
	}
	fmt.Printf("  → Removed %d director(ies)\n", removed)
	return removed, nil
}

// ── Cleanup: Ollama models ──────────────────────────────────────────────────

func cleanupOllama(dryRun bool) (int, error) {
	fmt.Printf("Ollama models:\n")

	if exec.Command("ollama", "--version").Run() != nil {
		fmt.Println("  (ollama not installed)")
		return 0, nil
	}

	out, err := exec.Command("ollama", "list").Output()
	if err != nil {
		fmt.Println("  (could not list models — is ollama running?)")
		return 0, nil
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	// First line is header
	models := []string{}
	for i, line := range lines {
		if i == 0 {
			continue // skip header
		}
		fields := strings.Fields(line)
		if len(fields) > 0 {
			models = append(models, fields[0])
		}
	}

	if len(models) == 0 {
		fmt.Println("  (none)")
		return 0, nil
	}

	for _, m := range models {
		fmt.Printf("  %s\n", m)
	}

	if dryRun {
		fmt.Printf("  → %d model(s) would be removed\n", len(models))
		return len(models), nil
	}

	removed := 0
	for _, m := range models {
		cmd := exec.Command("ollama", "rm", m)
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: could not remove model %s: %v\n", m, err)
		} else {
			removed++
		}
	}
	fmt.Printf("  → Removed %d model(s)\n", removed)
	return removed, nil
}

// ── Cleanup: Config file ────────────────────────────────────────────────────

func cleanupConfig(dryRun bool) (int, error) {
	home, _ := os.UserHomeDir()
	configDir := filepath.Join(home, ".tokenize")

	fmt.Printf("Config directory:\n")

	info, err := os.Stat(configDir)
	if err != nil || !info.IsDir() {
		fmt.Println("  (none)")
		return 0, nil
	}

	entries, _ := os.ReadDir(configDir)
	// Only count config-related files, not model cache (handled separately)
	configFiles := []string{}
	for _, e := range entries {
		name := e.Name()
		if name == "config.yaml" || name == "config.yml" || strings.HasSuffix(name, ".bak") {
			configFiles = append(configFiles, filepath.Join(configDir, name))
		}
	}

	fmt.Printf("  %s\n", configDir)
	for _, f := range configFiles {
		fmt.Printf("    %s\n", filepath.Base(f))
	}

	if dryRun {
		if len(configFiles) > 0 {
			fmt.Printf("  → config file(s) would be removed\n")
		}
		return len(configFiles), nil
	}

	removed := 0
	for _, f := range configFiles {
		if err := os.Remove(f); err == nil {
			removed++
		}
	}

	// If the directory is now empty (or only has dirs we already cleaned), remove it
	remaining, _ := os.ReadDir(configDir)
	if len(remaining) == 0 {
		os.Remove(configDir)
	}

	fmt.Printf("  → Removed %d config file(s)\n", removed)
	return removed, nil
}

// ── Cleanup: System service ─────────────────────────────────────────────────

func cleanupService(dryRun bool) (int, error) {
	fmt.Printf("System service:\n")

	switch runtime.GOOS {
	case "darwin":
		return cleanupServiceMacOS(dryRun)
	case "linux":
		return cleanupServiceLinux(dryRun)
	default:
		fmt.Println("  (not applicable on this OS)")
		return 0, nil
	}
}

func cleanupServiceMacOS(dryRun bool) (int, error) {
	home, _ := os.UserHomeDir()
	plist := filepath.Join(home, "Library", "LaunchAgents", "com.tokenize.gpu-agent.plist")

	if _, err := os.Stat(plist); err != nil {
		fmt.Println("  (no launchd service installed)")
		return 0, nil
	}

	fmt.Printf("  LaunchAgent: %s\n", plist)

	if dryRun {
		fmt.Println("  → would unload and remove")
		return 1, nil
	}

	// Unload first (ignore errors — may already be unloaded)
	exec.Command("launchctl", "unload", plist).Run()

	if err := os.Remove(plist); err != nil {
		return 0, fmt.Errorf("could not remove %s: %w", plist, err)
	}

	fmt.Println("  → Unloaded and removed")
	return 1, nil
}

func cleanupServiceLinux(dryRun bool) (int, error) {
	unitFile := "/etc/systemd/system/tokenize-gpu-agent.service"

	if _, err := os.Stat(unitFile); err != nil {
		fmt.Println("  (no systemd service installed)")
		return 0, nil
	}

	fmt.Printf("  Systemd unit: %s\n", unitFile)

	if dryRun {
		fmt.Println("  → would stop, disable, and remove")
		return 1, nil
	}

	exec.Command("systemctl", "stop", "tokenize-gpu-agent").Run()
	exec.Command("systemctl", "disable", "tokenize-gpu-agent").Run()

	if err := os.Remove(unitFile); err != nil {
		return 0, fmt.Errorf("could not remove %s (try with sudo): %w", unitFile, err)
	}

	exec.Command("systemctl", "daemon-reload").Run()

	fmt.Println("  → Stopped, disabled, and removed")
	return 1, nil
}

// ── Cleanup: Log files ──────────────────────────────────────────────────────

func cleanupLogs(dryRun bool) (int, error) {
	fmt.Printf("Log files:\n")

	var logFiles []string

	home, _ := os.UserHomeDir()

	// macOS log location
	macLog := filepath.Join(home, "Library", "Logs", "tokenize-gpu-agent.log")
	if _, err := os.Stat(macLog); err == nil {
		logFiles = append(logFiles, macLog)
	}

	// Common log locations
	commonLogs := []string{
		"/var/log/tokenize-gpu-agent.log",
		filepath.Join(home, ".tokenize", "agent.log"),
	}
	for _, l := range commonLogs {
		if _, err := os.Stat(l); err == nil {
			logFiles = append(logFiles, l)
		}
	}

	if len(logFiles) == 0 {
		fmt.Println("  (none)")
		return 0, nil
	}

	for _, f := range logFiles {
		size := int64(0)
		if info, err := os.Stat(f); err == nil {
			size = info.Size()
		}
		fmt.Printf("  %s (%s)\n", f, humanSize(size))
	}

	if dryRun {
		fmt.Printf("  → %d log file(s) would be removed\n", len(logFiles))
		return len(logFiles), nil
	}

	removed := 0
	for _, f := range logFiles {
		if err := os.Remove(f); err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: could not remove %s: %v\n", f, err)
		} else {
			removed++
		}
	}
	fmt.Printf("  → Removed %d log file(s)\n", removed)
	return removed, nil
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// dirStats returns total size in bytes and file count for a directory.
func dirStats(path string) (int64, int) {
	var totalSize int64
	var count int
	filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			totalSize += info.Size()
			count++
		}
		return nil
	})
	return totalSize, count
}

// humanSize formats bytes into a human-readable string.
func humanSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

