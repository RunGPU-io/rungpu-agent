package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/config"
	"github.com/RunGPU-io/rungpu-agent/internal/dockermgr"
	"github.com/RunGPU-io/rungpu-agent/internal/gpu"
	"github.com/RunGPU-io/rungpu-agent/internal/pool"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

var version = "dev"

var (
	agentStopMu sync.Mutex
	agentStop   context.CancelFunc
)

func setAgentStop(fn context.CancelFunc) {
	agentStopMu.Lock()
	agentStop = fn
	agentStopMu.Unlock()
}

func requestAgentStop() bool {
	agentStopMu.Lock()
	fn := agentStop
	agentStopMu.Unlock()
	if fn == nil {
		return false
	}
	fn()
	return true
}

func agentIsRunning() bool {
	agentStopMu.Lock()
	local := agentStop != nil
	agentStopMu.Unlock()
	if local {
		return true
	}
	release, err := acquireAgentProcessLock()
	if err != nil {
		return true
	}
	release()
	return false
}

func scheduleZone(value string) string {
	if value == "" {
		return "local"
	}
	return value
}

func updateSavedConfig(path string, mutate func(*types.Config) error) error {
	cfg, err := config.Load(path)
	if err != nil {
		return err
	}
	if err := mutate(cfg); err != nil {
		return err
	}
	return config.Save(cfg, path)
}

func cmdPause(args []string) error {
	fs := flag.NewFlagSet("pause", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)
	if err := updateSavedConfig(*cfgPath, func(cfg *types.Config) error {
		cfg.Paused = true
		return nil
	}); err != nil {
		return err
	}
	fmt.Println("Paused. The start command or GUI should stay running. Use resume to earn again.")
	return nil
}

func cmdResume(args []string) error {
	fs := flag.NewFlagSet("resume", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)
	if err := updateSavedConfig(*cfgPath, func(cfg *types.Config) error {
		cfg.Paused = false
		return nil
	}); err != nil {
		return err
	}
	fmt.Println("Resume requested. If start or the GUI is running, earning continues when the schedule allows.")
	return nil
}

func cmdUnenroll(args []string) error {
	fs := flag.NewFlagSet("unenroll", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)
	requestAgentStop()
	if err := os.Remove(*cfgPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove config: %w", err)
	}
	fmt.Println("Unenrolled. This computer's config was deleted. Models and cache were left in place.")
	return nil
}

func cmdSchedule(args []string) error {
	fs := flag.NewFlagSet("schedule", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	startHour := fs.Int("start", 18, "hour earning starts (0-23)")
	endHour := fs.Int("end", 23, "hour earning pauses (0-23)")
	timezone := fs.String("timezone", "", "IANA timezone, empty means this computer's local zone")
	days := fs.String("days", "", "weekly days, comma-separated (mon,tue). Empty means every day")
	add := fs.Bool("add", false, "add this window instead of replacing the schedule")
	off := fs.Bool("off", false, "earn whenever the agent is running and not paused")
	_ = fs.Parse(args)
	weekdays, err := config.ParseDaysFlag(*days)
	if err != nil {
		return err
	}
	if err := updateSavedConfig(*cfgPath, func(cfg *types.Config) error {
		if *off {
			cfg.Schedule.Enabled = false
			return nil
		}
		if *startHour < 0 || *startHour > 23 || *endHour < 0 || *endHour > 23 {
			return fmt.Errorf("hours must be between 0 and 23")
		}
		window := types.ScheduleWindow{StartHour: *startHour, EndHour: *endHour, Days: weekdays}
		next := types.ScheduleConfig{
			Enabled:  true,
			Timezone: strings.TrimSpace(*timezone),
			Windows:  []types.ScheduleWindow{window},
		}
		if *add {
			existing := config.NormalizeSchedule(cfg.Schedule)
			if strings.TrimSpace(*timezone) == "" {
				next.Timezone = existing.Timezone
			}
			next.Windows = append(config.ScheduleWindows(existing), window)
		}
		cfg.Schedule = config.NormalizeSchedule(next)
		return nil
	}); err != nil {
		return err
	}
	if *off {
		fmt.Println("Schedule cleared. The agent earns whenever it is running and not paused.")
		return nil
	}
	fmt.Printf("Schedule set: %s %s. Keep start or the GUI running.\n", config.FormatWindows([]types.ScheduleWindow{{
		StartHour: *startHour,
		EndHour:   *endHour,
		Days:      weekdays,
	}}), scheduleZone(*timezone))
	return nil
}

func main() {
	if len(os.Args) < 2 {
		if supportsDesktopGUI() {
			if err := runGUI(); err != nil {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
				os.Exit(1)
			}
			return
		}
		usage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	var err error
	switch cmd {
	case "gui":
		err = runGUI()
	case "init":
		err = cmdInit(args)
	case "setup":
		err = cmdSetup(args)
	case "start":
		err = cmdStart(args)
	case "pause":
		err = cmdPause(args)
	case "resume":
		err = cmdResume(args)
	case "schedule":
		err = cmdSchedule(args)
	case "unenroll":
		err = cmdUnenroll(args)
	case "status":
		err = cmdStatus(args)
	case "cleanup":
		err = cmdCleanup(args)
	case "version", "--version", "-v":
		fmt.Println(version)
		return
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

Two options:
  GUI      Double-click or run "gui" — paste the token, then Enroll / Setup / Start
  Command  Typed commands: init, setup, start, status

Commands:
  gui      Open the GUI
  init     Create config (auto-detects GPUs)
	setup    Install runtime requirements with the platform package manager
  start    Keep the agent running and earn when not paused
  pause    Disconnect from the pool without stopping the process
  resume   Start earning again
  schedule Set daily or weekly earn windows, or --off to earn whenever running
  unenroll Delete this computer's config so you can enroll again
	status   Check enrollment, earning state, runtime, GPUs, and live metrics
  cleanup  Remove containers, volumes, images, caches, config, and services
  version  Print the GitHub release version of this binary

Run "tokenize-gpu-agent <command> -h" for command flags.`)
}

func cmdInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ExitOnError)
	enrollmentToken := fs.String("enrollment-token", "", "one-time fleet enrollment token")
	batchTokenFile := fs.String("batch-token-file", "", "file containing a short-lived enterprise batch token")
	poolURL := fs.String("pool-url", config.DefaultPoolURL, "pool coordinator URL")
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)

	if (*enrollmentToken == "") == (*batchTokenFile == "") {
		return fmt.Errorf("provide exactly one of --enrollment-token or --batch-token-file")
	}
	if err := initializeEnrollment(*poolURL, *enrollmentToken, *batchTokenFile, *cfgPath); err != nil {
		return err
	}

	executable, executableErr := os.Executable()
	if executableErr != nil {
		executable = "rungpu-agent"
	}
	start := startCommand(runtime.GOOS, filepath.Base(executable))
	setup := setupCommand(runtime.GOOS, filepath.Base(executable))
	printEnrollmentSuccess(*cfgPath, start)

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	return enrollmentDiagnostics(start, setup, *cfgPath, cfg)
}

func printEnrollmentSuccess(cfgPath, start string) {
	fmt.Println()
	fmt.Println("Enrollment successful. This machine is enrolled with RunGPU.")
	fmt.Printf("Credentials saved to: %s\n", cfgPath)
	fmt.Println("Start the agent to connect to the pool:")
	fmt.Printf("  %s\n", start)
}

func initializeEnrollment(poolURL, enrollmentToken, batchTokenFile, cfgPath string) error {
	unlock, err := config.LockEnrollment(cfgPath)
	if err != nil {
		return err
	}
	defer unlock()
	if err := config.PreflightSave(cfgPath); err != nil {
		return fmt.Errorf("cannot persist credentials at %s; fix the config path/permissions before retrying (no enrollment request sent): %w", cfgPath, err)
	}
	var existing *types.Config
	if batchTokenFile == "" {
		existing, err = singleEnrollmentConfig(cfgPath, poolURL)
		if err != nil {
			return fmt.Errorf("%w (no enrollment request sent)", err)
		}
	}
	installationID, err := config.LoadOrCreateInstallationID(cfgPath)
	if err != nil {
		return fmt.Errorf("prepare installation identity (no enrollment request sent): %w", err)
	}
	installationSecret, err := config.LoadOrCreateInstallationSecret(cfgPath)
	if err != nil {
		return fmt.Errorf("prepare installation secret (no enrollment request sent): %w", err)
	}
	var enrolled *enrollmentResponse
	if batchTokenFile != "" {
		tokenBytes, readErr := readBatchTokenFile(batchTokenFile)
		if readErr != nil {
			return fmt.Errorf("read batch token file: %w", readErr)
		}
		hostname, hostnameErr := os.Hostname()
		if hostnameErr != nil || hostname == "" {
			hostname = "rungpu-host"
		}
		enrolled, err = enrollMachineBatch(poolURL, strings.TrimSpace(string(tokenBytes)), installationID, installationSecret, hostname)
	} else {
		enrolled, err = enrollMachine(poolURL, enrollmentToken, installationID, installationSecret)
	}
	if err != nil {
		return fmt.Errorf("%w; retry the same enrollment command/token using config path %s and its original installation-id/installation-secret files; do not delete them. Single-token delivery recovery lasts 7 days; if expired or replaced, request Replace enrollment in the dashboard", err, cfgPath)
	}

	if batchTokenFile == "" {
		current, loadErr := singleEnrollmentConfig(cfgPath, poolURL)
		if loadErr != nil {
			return fmt.Errorf("server enrollment succeeded but local config was preserved: %w; retain the original token and installation-id/installation-secret for recovery within 7 days", loadErr)
		}
		for _, saved := range []*types.Config{existing, current} {
			if saved != nil && saved.MachineID != "" && saved.MachineID != enrolled.MachineID {
				return fmt.Errorf("server enrollment returned machine %s, but config %s belongs to machine %s; existing config was preserved. Use the token for this machine (dashboard Replace enrollment), or restore the intended installation's original config path and installation-id/installation-secret files; do not overwrite another enrollment", enrolled.MachineID, cfgPath, saved.MachineID)
			}
		}
		existing = current
	} else {
		existing, _ = config.Load(cfgPath)
	}

	if existing != nil &&
		existing.MachineID == enrolled.MachineID && existing.APIKey == enrolled.AgentKey &&
		sameEnrollmentPool(existing.PoolURL, poolURL) {
		return nil
	}
	cfg := existing
	if cfg == nil || batchTokenFile != "" {
		cfg, err = config.New(enrolled.AgentKey)
		if err != nil {
			return fmt.Errorf("server enrollment succeeded but preparing local config failed: %w; retry the SAME token with config path %s and preserve installation-id/installation-secret (single-token recovery: 7 days)", err, cfgPath)
		}
	}
	cfg.PoolURL = poolURL
	cfg.APIKey = enrolled.AgentKey
	cfg.MachineID = enrolled.MachineID
	cfg.PricePerMinute = enrolled.PricePerMinute
	cfg.ContributeFree = enrolled.PricePerMinute == 0
	config.RefreshGPUIDs(cfg)
	if err := config.Save(cfg, cfgPath); err != nil {
		return fmt.Errorf("server enrollment succeeded but saving credentials at %s failed: %w; fix local persistence and retry the SAME token with this config path; preserve installation-id and installation-secret (single-token recovery: 7 days)", cfgPath, err)
	}
	return nil
}

func sameEnrollmentPool(saved, requested string) bool {
	if saved == "" {
		saved = config.DefaultPoolURL
	}
	return strings.TrimRight(saved, "/") == strings.TrimRight(requested, "/")
}

func singleEnrollmentConfig(cfgPath, poolURL string) (*types.Config, error) {
	existing, err := config.Load(cfgPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("cannot safely read existing config %s: %w; restore or repair it before enrollment", cfgPath, err)
	}
	if !sameEnrollmentPool(existing.PoolURL, poolURL) {
		return nil, fmt.Errorf("config %s belongs to a different pool; existing config was preserved. Use its original --pool-url, or the intended installation's original --config path and installation files", cfgPath)
	}
	if existing.APIKey != "" && existing.MachineID == "" {
		return nil, fmt.Errorf("config %s has a credential but no machine ID; existing config was preserved. Restore its machine ID before using this machine's replacement enrollment token", cfgPath)
	}
	return existing, nil
}

func enrollmentDiagnostics(start, setup, cfgPath string, cfg *types.Config) error {
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║           RunGPU Agent — Setup & Diagnostics            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Step 1/4 — Checking prerequisites...")
	fmt.Println()

	issues := 0

	dockerOk := exec.Command("docker", "version").Run() == nil
	if dockerOk {
		fmt.Println("  ✅ Docker            installed")
	} else {
		fmt.Println("  ❌ Docker            not found")
		fmt.Println("     → Install: https://docs.docker.com/get-docker/")
		issues++
	}

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

	if runtime.GOOS == "linux" && dockerOk {
		nctOk := exec.Command("docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.0.0-base-ubuntu22.04", "nvidia-smi").Run() == nil
		if nctOk {
			fmt.Println("  ✅ NVIDIA Toolkit    Docker GPU access works")
		} else if nvidiaOk {
			fmt.Println("  ⚠️  NVIDIA Toolkit    Docker can't access GPU")
			fmt.Println("     → Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
		}
	}

	ollamaOk := exec.Command("ollama", "--version").Run() == nil
	if ollamaOk {
		fmt.Println("  ✅ Ollama            installed (LLM inference)")
	} else {
		fmt.Println("  ℹ️  Ollama            not installed (optional — for LLM jobs)")
		fmt.Println("     → Install: https://ollama.com")
	}

	home, _ := os.UserHomeDir()
	if stat, err := os.Stat(home); err == nil && stat.IsDir() {

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
		fmt.Println("     Enrollment is saved. Do not run init again.")
		fmt.Printf("     Install requirements: %s\n", setup)
		fmt.Printf("     Then start the agent: %s\n", start)
		fmt.Println()
	}

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

	fmt.Println("Step 3/4 — Creating configuration...")
	fmt.Println()
	fmt.Printf("  ✅ Config saved to: %s\n", cfgPath)
	fmt.Printf("  ✅ Pool URL: %s\n", cfg.PoolURL)
	fmt.Printf("  ✅ Machine ID: %s\n", cfg.MachineID)
	fmt.Println()

	fmt.Println("Step 4/4 — Ready!")
	fmt.Println()
	fmt.Println("  ┌─────────────────────────────────────────────────────┐")
	fmt.Println("  │  Your GPU agent is configured. Start it with:      │")
	fmt.Println("  │                                                     │")
	fmt.Printf("  │    %-48s │\n", start)
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

func readBatchTokenFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("batch token path must be a regular file")
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0077 != 0 {
		return nil, fmt.Errorf("batch token file permissions must be 0600 or stricter")
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(string(contents)) == "" {
		return nil, fmt.Errorf("batch token file is empty")
	}
	return contents, nil
}

func startCommand(goos, executable string) string {
	if goos == "windows" {
		return `.\` + executable + " start"
	}
	return "./" + executable + " start"
}

func setupCommand(goos, executable string) string {
	if goos == "windows" {
		return `.\` + executable + " setup"
	}
	return "./" + executable + " setup"
}

func cmdSetup(args []string) error {
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	_ = fs.Parse(args)
	fmt.Printf("RunGPU Agent %s setup\n", version)

	commands, err := setupCommands(runtime.GOOS)
	if err != nil {
		return err
	}
	for _, command := range commands {
		fmt.Printf("Running: %s\n", strings.Join(command, " "))
		cmd := exec.Command(command[0], command[1:]...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("install runtime requirement: %w", err)
		}
	}

	fmt.Println("Runtime installation finished.")
	if runtime.GOOS == "windows" {
		fmt.Println("Open Docker Desktop and finish its WSL2 setup, then restart Windows if prompted.")
	}
	fmt.Println("Run the agent again with: " + startCommand(runtime.GOOS, executableName()))
	return nil
}

func setupCommands(goos string) ([][]string, error) {
	switch goos {
	case "windows":
		if _, err := exec.LookPath("winget"); err != nil {
			return nil, fmt.Errorf("winget is required; install App Installer from the Microsoft Store")
		}
		return setupCommandPlan(goos), nil
	case "darwin":
		if _, err := exec.LookPath("brew"); err != nil {
			return nil, fmt.Errorf("Homebrew is required; install it from https://brew.sh")
		}
		return setupCommandPlan(goos), nil
	case "linux":
		return nil, fmt.Errorf("install Docker for your distribution from https://docs.docker.com/engine/install/ and Ollama from https://ollama.com/download/linux")
	default:
		return nil, fmt.Errorf("automatic setup is not supported on %s", goos)
	}
}

func setupCommandPlan(goos string) [][]string {
	switch goos {
	case "windows":
		return [][]string{
			{"winget", "install", "--id", "Docker.DockerDesktop", "--exact", "--accept-package-agreements", "--accept-source-agreements"},
			{"winget", "install", "--id", "Ollama.Ollama", "--exact", "--accept-package-agreements", "--accept-source-agreements"},
		}
	case "darwin":
		return [][]string{{"brew", "install", "ollama"}}
	default:
		return nil
	}
}

func executableName() string {
	executable, err := os.Executable()
	if err != nil {
		return "rungpu-agent"
	}
	return filepath.Base(executable)
}

type enrollmentResponse struct {
	MachineID      string  `json:"machine_id"`
	AgentKey       string  `json:"agent_key"`
	PricePerMinute float64 `json:"price_per_minute"`
}

func enrollMachine(poolURL, token, installationID, installationSecret string) (*enrollmentResponse, error) {
	base, err := enrollmentURL(poolURL, "/api/v1/fleet/enroll")
	if err != nil {
		return nil, fmt.Errorf("invalid pool URL: %w", err)
	}
	body, _ := json.Marshal(map[string]string{
		"enrollment_token": token, "installation_id": installationID, "installation_secret": installationSecret,
	})
	req, err := http.NewRequest(http.MethodPost, base.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 30 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fleet enrollment failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fleet enrollment rejected with HTTP %d", resp.StatusCode)
	}
	var result enrollmentResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("invalid enrollment response: %w", err)
	}
	if result.MachineID == "" || result.AgentKey == "" {
		return nil, fmt.Errorf("enrollment response is missing credentials")
	}
	return &result, nil
}

func enrollMachineBatch(poolURL, token, installationID, installationSecret, hostname string) (*enrollmentResponse, error) {
	base, err := enrollmentURL(poolURL, "/api/v1/fleet/batch/enroll")
	if err != nil {
		return nil, err
	}
	body, _ := json.Marshal(map[string]string{
		"batch_token":         token,
		"installation_id":     installationID,
		"installation_secret": installationSecret,
		"hostname":            hostname,
		"operating_system":    runtime.GOOS,
		"architecture":        runtime.GOARCH,
		"agent_version":       version,
	})
	req, err := http.NewRequest(http.MethodPost, base.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("batch enrollment failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("batch enrollment rejected with HTTP %d", resp.StatusCode)
	}
	var result enrollmentResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("invalid batch enrollment response: %w", err)
	}
	if result.MachineID == "" || result.AgentKey == "" {
		return nil, fmt.Errorf("batch enrollment response is missing credentials")
	}
	return &result, nil
}

func enrollmentURL(poolURL, endpoint string) (*url.URL, error) {
	base, err := url.Parse(poolURL)
	if err != nil {
		return nil, fmt.Errorf("invalid pool URL: %w", err)
	}
	if base.Scheme != "https" {
		host := base.Hostname()
		ip := net.ParseIP(host)
		if base.Scheme != "http" || (host != "localhost" && (ip == nil || !ip.IsLoopback())) {
			return nil, fmt.Errorf("fleet enrollment requires HTTPS except on localhost")
		}
	}
	base.Path = strings.TrimRight(base.Path, "/") + endpoint
	return base, nil
}

func cmdStart(args []string) error {
	fs := flag.NewFlagSet("start", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)

	releaseLock, err := acquireAgentProcessLock()
	if err != nil {
		return err
	}
	defer releaseLock()

	fmt.Printf("RunGPU Agent %s starting...\n", version)
	fmt.Printf("Loading configuration: %s\n", *cfgPath)
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	hadMachineID := cfg.MachineID != ""
	if err := config.Validate(cfg); err != nil {
		return err
	}
	if !hadMachineID {
		if err := config.Save(cfg, *cfgPath); err != nil {
			return fmt.Errorf("persist machine identity: %w", err)
		}
	}

	if err := prepareExecutionRecovery(cfg, *cfgPath); err != nil {
		return err
	}

	fmt.Println("Detecting GPU and runtime capabilities...")
	clients, err := pool.NewClients(cfg)
	if err != nil {
		return err
	}
	for _, client := range clients {
		client.SetConfigPath(*cfgPath)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	setAgentStop(stop)
	defer setAgentStop(nil)
	defer stop()

	if status := config.EvaluateEarning(cfg, time.Now()); !status.Open {
		fmt.Printf("Earning: %s\n", status.Summary)
		fmt.Println("Keep this process running. It will connect when earning is allowed.")
	} else {
		fmt.Printf("Earning: %s\n", status.Summary)
	}

	fmt.Printf("Starting %d GPU worker(s) (pool: %s)...\n", len(clients), cfg.PoolURL)
	go pool.RunCustomAssetCleanup(ctx, cfg, clients)
	var workers sync.WaitGroup
	errCh := make(chan error, len(clients))
	for _, client := range clients {
		workers.Add(1)
		go func(client *pool.Client) {
			defer workers.Done()
			if runErr := client.Run(ctx); runErr != nil && runErr != context.Canceled {
				errCh <- runErr
				stop()
			}
		}(client)
	}
	workers.Wait()
	select {
	case runErr := <-errCh:
		return runErr
	default:
	}
	fmt.Println("Agent stopped.")
	return nil
}

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	cfgPath := fs.String("config", config.DefaultConfigPath(), "config file path")
	_ = fs.Parse(args)

	fmt.Println("\n=== RunGPU Agent Status ===")
	fmt.Printf("Version: %s\n", version)
	fmt.Println("Checking GPU and runtime capabilities...")
	monitor := gpu.NewMonitor()
	gpus := monitor.GPUs()
	backend := monitor.Backend()
	capabilities := gpu.RuntimeCapabilities()
	requiredRuntime, runtimeReady := runtimeReadiness(backend, capabilities)

	if cfg, err := config.Load(*cfgPath); err == nil && cfg.APIKey != "" && cfg.MachineID != "" {
		fmt.Printf("Enrollment: Configured (%s)\n", cfg.MachineID)
		earning := config.EvaluateEarning(cfg, time.Now())
		fmt.Printf("Earning: %s\n", earning.Summary)
		if cfg.Schedule.Enabled {
			fmt.Printf("Schedule: %s %s\n", config.FormatWindows(config.ScheduleWindows(cfg.Schedule)), scheduleZone(cfg.Schedule.Timezone))
		}
		if earning.NextLabel != "" {
			fmt.Printf("Next: %s\n", earning.NextLabel)
		}
		if agentIsRunning() {
			fmt.Println("Process: Running (start or GUI is up)")
		} else {
			fmt.Println("Process: Not running — start the GUI or run start to earn")
		}
	} else {
		fmt.Println("Enrollment: Not configured")
		fmt.Println("  Run the enrollment command from the RunGPU host dashboard.")
	}
	fmt.Printf("Backend: %s\n", backend)
	if backend != "cuda" {
		fmt.Println("  (non-NVIDIA host: jobs run natively via Ollama; install from https://ollama.com)")
	}
	if runtimeReady {
		fmt.Printf("Runtime: Ready (%s)\n", requiredRuntime)
	} else {
		fmt.Printf("Runtime: Setup required (%s is unavailable)\n", requiredRuntime)
		fmt.Printf("  Run: %s\n", setupCommand(runtime.GOOS, executableName()))
	}
	if len(capabilities) > 0 {
		fmt.Printf("Capabilities: %s\n", strings.Join(capabilities, ", "))
	} else {
		fmt.Println("Capabilities: none")
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

func runtimeReadiness(backend string, capabilities []string) (string, bool) {
	required := "ollama"
	if backend == "cuda" {
		required = "docker"
	}
	for _, capability := range capabilities {
		if capability == required {
			return required, true
		}
	}
	return required, false
}

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

	nothingSelected := !*all && !*containers && !*volumes && !*images && !*cache && !*ollama && !*configFile && !*service && !*logs
	if nothingSelected {
		*dryRun = true

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

	if *containers && hasDocker {
		cleaned, err := cleanupContainers(ctx, docker, *dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: container cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *volumes && hasDocker {
		cleaned, err := cleanupVolumes(ctx, docker, *dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: volume cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *images && hasDocker {
		cleaned, err := cleanupImages(ctx, docker, *dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: image cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *cache {
		cleaned, err := cleanupCache(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: cache cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *ollama {
		cleaned, err := cleanupOllama(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: ollama cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *configFile {
		cleaned, err := cleanupConfig(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: config cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *service {
		cleaned, err := cleanupService(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: service cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

	if *logs {
		cleaned, err := cleanupLogs(*dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Warning: log cleanup: %v\n", err)
		}
		totalCleaned += cleaned
	}

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

func cleanupCache(dryRun bool) (int, error) {
	home, _ := os.UserHomeDir()
	cacheDir := filepath.Join(home, ".tokenize", "models")
	stagingDir := filepath.Join(home, ".tokenize", "staging")
	outputDir := filepath.Join(home, ".tokenize", "output")
	workspacesDir := filepath.Join(home, ".tokenize", "workspaces")

	dirs := []struct {
		path string
		desc string
	}{
		{cacheDir, "model cache"},
		{stagingDir, "job staging files"},
		{outputDir, "job output files"},
		{workspacesDir, "workspace data"},
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

	models := []string{}
	for i, line := range lines {
		if i == 0 {
			continue
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

	configFiles := []string{}
	for _, e := range entries {
		name := e.Name()
		if name == "config.yaml" || name == "config.yml" || name == "installation-id" || name == "installation-secret" || strings.HasSuffix(name, ".bak") {
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

	remaining, _ := os.ReadDir(configDir)
	if len(remaining) == 0 {
		os.Remove(configDir)
	}

	fmt.Printf("  → Removed %d config file(s)\n", removed)
	return removed, nil
}

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

func cleanupLogs(dryRun bool) (int, error) {
	fmt.Printf("Log files:\n")

	var logFiles []string

	home, _ := os.UserHomeDir()

	macLog := filepath.Join(home, "Library", "Logs", "tokenize-gpu-agent.log")
	if _, err := os.Stat(macLog); err == nil {
		logFiles = append(logFiles, macLog)
	}

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
