// Package gpu provides cross-platform, cgo-free GPU detection and monitoring.
// NVIDIA GPUs are read by parsing `nvidia-smi` (Linux/Windows); macOS falls
// back to `sysctl`. If nothing is found, a CPU-only placeholder is returned so
// the agent still runs everywhere.
package gpu

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

// Detect returns the available accelerators. It never fails and always returns
// at least one entry.
func Detect() []types.GPUInfo {
	if gpus := detectNvidia(); len(gpus) > 0 {
		return gpus
	}
	if runtime.GOOS == "darwin" {
		if gpus := detectMacOS(); len(gpus) > 0 {
			return gpus
		}
	}
	// CPU-only fallback (e.g. macOS, or a host without NVIDIA GPUs).
	return []types.GPUInfo{{
		Index:             0,
		Name:              "cpu-only",
		ComputeCapability: "n/a",
		MemoryMB:          0,
		DriverVersion:     "n/a",
	}}
}

// detectNvidia parses `nvidia-smi`. Returns nil if nvidia-smi is unavailable.
func detectNvidia() []types.GPUInfo {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,driver_version",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}

	var gpus []types.GPUInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		fields := splitCSV(line)
		if len(fields) < 4 {
			continue
		}
		idx, _ := strconv.Atoi(fields[0])
		memMB, _ := strconv.ParseUint(fields[2], 10, 64)
		gpus = append(gpus, types.GPUInfo{
			Index:             idx,
			Name:              fields[1],
			ComputeCapability: "cuda",
			MemoryMB:          memMB,
			DriverVersion:     fields[3],
		})
	}
	return gpus
}

// detectMacOS reports the Apple Silicon SoC via sysctl and system_profiler;
// unified memory is used as the GPU memory figure. On Intel Macs without a
// discrete GPU, the integrated GPU name is read from system_profiler.
func detectMacOS() []types.GPUInfo {
	name := sysctl("machdep.cpu.brand_string")
	if name == "" {
		name = "Apple GPU"
	}

	// Try to get the actual GPU chip name from system_profiler (e.g. "Apple M2 Pro").
	if gpuName := detectMacGPUName(); gpuName != "" {
		name = gpuName
	}

	var memMB uint64
	if v := sysctl("hw.memsize"); v != "" {
		if bytes, err := strconv.ParseUint(v, 10, 64); err == nil {
			memMB = bytes / (1024 * 1024)
		}
	}

	// Determine compute capability: Apple Silicon supports Metal, Intel Macs
	// may only have an integrated GPU without Metal compute support.
	capability := "metal"
	driver := macOSVersion()
	if driver == "" {
		driver = "macos"
	}

	return []types.GPUInfo{{
		Index:             0,
		Name:              name,
		ComputeCapability: capability,
		MemoryMB:          memMB,
		DriverVersion:     driver,
	}}
}

// detectMacGPUName uses system_profiler to find the GPU chip name. Returns ""
// if unavailable. Works on both Apple Silicon and Intel Macs.
func detectMacGPUName() string {
	out, err := exec.Command("system_profiler", "SPDisplaysDataType", "-detailLevel", "mini").Output()
	if err != nil {
		return ""
	}
	// Look for "Chipset Model:" or "Chip:" lines.
	for _, line := range strings.Split(string(out), "\n") {
		trimmed := strings.TrimSpace(line)
		for _, prefix := range []string{"Chipset Model:", "Chip:"} {
			if strings.HasPrefix(trimmed, prefix) {
				val := strings.TrimSpace(strings.TrimPrefix(trimmed, prefix))
				if val != "" {
					return val
				}
			}
		}
	}
	return ""
}

// macOSVersion returns the macOS version string (e.g. "15.5") or "".
func macOSVersion() string {
	out, err := exec.Command("sw_vers", "-productVersion").Output()
	if err != nil {
		return ""
	}
	return "macOS " + strings.TrimSpace(string(out))
}

func sysctl(key string) string {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// Monitor exposes live GPU metrics.
type Monitor struct {
	gpus     []types.GPUInfo
	hasNvidia bool
}

func NewMonitor() *Monitor {
	gpus := Detect()
	hasNvidia := len(gpus) > 0 && gpus[0].ComputeCapability == "cuda"
	return &Monitor{gpus: gpus, hasNvidia: hasNvidia}
}

func (m *Monitor) GPUs() []types.GPUInfo { return m.gpus }

// Backend reports the execution backend this host supports, derived from the
// primary accelerator: "cuda" (NVIDIA), "metal" (Apple Silicon), or "cpu".
func (m *Monitor) Backend() string {
	if len(m.gpus) == 0 {
		return "cpu"
	}
	switch m.gpus[0].ComputeCapability {
	case "cuda":
		return "cuda"
	case "metal":
		return "metal"
	default:
		return "cpu"
	}
}

// CollectMetrics returns live telemetry via nvidia-smi, or static zeros when
// NVIDIA isn't present.
func (m *Monitor) CollectMetrics() []types.GPUMetrics {
	if m.hasNvidia {
		if metrics := collectNvidiaMetrics(); len(metrics) > 0 {
			return metrics
		}
	}
	// Fallback: static info, no live telemetry.
	metrics := make([]types.GPUMetrics, 0, len(m.gpus))
	for _, g := range m.gpus {
		metrics = append(metrics, types.GPUMetrics{
			GPUIndex:      g.Index,
			MemoryTotalMB: g.MemoryMB,
		})
	}
	return metrics
}

func collectNvidiaMetrics() []types.GPUMetrics {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}

	var metrics []types.GPUMetrics
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := splitCSV(line)
		if len(f) < 6 {
			continue
		}
		idx, _ := strconv.Atoi(f[0])
		util, _ := strconv.ParseFloat(f[1], 64)
		used, _ := strconv.ParseUint(f[2], 10, 64)
		total, _ := strconv.ParseUint(f[3], 10, 64)

		m := types.GPUMetrics{
			GPUIndex:           idx,
			UtilizationPercent: util,
			MemoryUsedMB:       used,
			MemoryTotalMB:      total,
		}
		if temp, err := strconv.ParseFloat(f[4], 64); err == nil {
			m.TemperatureC = &temp
		}
		if power, err := strconv.ParseFloat(f[5], 64); err == nil {
			m.PowerDrawW = &power
		}
		metrics = append(metrics, m)
	}
	return metrics
}

// IsHealthy returns false if any GPU is too hot or near memory exhaustion.
func (m *Monitor) IsHealthy() bool {
	for _, metric := range m.CollectMetrics() {
		if metric.TemperatureC != nil && *metric.TemperatureC > 85.0 {
			return false
		}
		if metric.MemoryTotalMB > 0 {
			pct := float64(metric.MemoryUsedMB) / float64(metric.MemoryTotalMB) * 100.0
			if pct > 95.0 {
				return false
			}
		}
	}
	return true
}

// ── System info (CPU, RAM, OS) ──────────────────────────────────────────────

// SystemInfo holds host-level info beyond the GPU.
type SystemInfo struct {
	CPUModel   string
	CPUCores   int
	RAMTotalGB float64
	OSInfo     string // "darwin/arm64", "linux/amd64"
}

// DetectSystem returns CPU, RAM, and OS info for the host.
func DetectSystem() SystemInfo {
	info := SystemInfo{
		OSInfo: runtime.GOOS + "/" + runtime.GOARCH,
		CPUCores: runtime.NumCPU(),
	}

	// CPU model
	switch runtime.GOOS {
	case "darwin":
		info.CPUModel = sysctl("machdep.cpu.brand_string")
	case "linux":
		if out, err := exec.Command("lscpu").Output(); err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				if strings.HasPrefix(line, "Model name:") {
					info.CPUModel = strings.TrimSpace(strings.TrimPrefix(line, "Model name:"))
					break
				}
			}
		}
	}
	if info.CPUModel == "" {
		info.CPUModel = runtime.GOARCH
	}

	// Total RAM
	switch runtime.GOOS {
	case "darwin":
		if v := sysctl("hw.memsize"); v != "" {
			if bytes, err := strconv.ParseUint(v, 10, 64); err == nil {
				info.RAMTotalGB = float64(bytes) / (1024 * 1024 * 1024)
			}
		}
	case "linux":
		if out, err := exec.Command("grep", "MemTotal", "/proc/meminfo").Output(); err == nil {
			fields := strings.Fields(string(out))
			if len(fields) >= 2 {
				if kb, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
					info.RAMTotalGB = float64(kb) / (1024 * 1024)
				}
			}
		}
	}

	return info
}

// RAMUsage returns current RAM used and total in GB.
func RAMUsage() (usedGB, totalGB float64) {
	switch runtime.GOOS {
	case "darwin":
		if v := sysctl("hw.memsize"); v != "" {
			if bytes, err := strconv.ParseUint(v, 10, 64); err == nil {
				totalGB = float64(bytes) / (1024 * 1024 * 1024)
			}
		}
		// vm_stat gives page-level memory info on macOS
		if out, err := exec.Command("vm_stat").Output(); err == nil {
			var active, wired, compressed uint64
			pageSize := uint64(16384) // Apple Silicon default
			if ps := sysctl("hw.pagesize"); ps != "" {
				if v, err := strconv.ParseUint(ps, 10, 64); err == nil {
					pageSize = v
				}
			}
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "Pages active:") {
					active = parseVMStatValue(line)
				} else if strings.HasPrefix(line, "Pages wired down:") {
					wired = parseVMStatValue(line)
				} else if strings.HasPrefix(line, "Pages occupied by compressor:") {
					compressed = parseVMStatValue(line)
				}
			}
			usedGB = float64((active+wired+compressed)*pageSize) / (1024 * 1024 * 1024)
		}
	case "linux":
		if out, err := exec.Command("cat", "/proc/meminfo").Output(); err == nil {
			var total, available uint64
			for _, line := range strings.Split(string(out), "\n") {
				fields := strings.Fields(line)
				if len(fields) < 2 {
					continue
				}
				val, _ := strconv.ParseUint(fields[1], 10, 64)
				switch {
				case strings.HasPrefix(line, "MemTotal:"):
					total = val
				case strings.HasPrefix(line, "MemAvailable:"):
					available = val
				}
			}
			totalGB = float64(total) / (1024 * 1024)
			usedGB = float64(total-available) / (1024 * 1024)
		}
	}
	return
}

func parseVMStatValue(line string) uint64 {
	// "Pages active:    123456." → 123456
	parts := strings.SplitN(line, ":", 2)
	if len(parts) < 2 {
		return 0
	}
	s := strings.TrimSpace(parts[1])
	s = strings.TrimSuffix(s, ".")
	v, _ := strconv.ParseUint(s, 10, 64)
	return v
}

// OllamaModels returns the list of models currently pulled in Ollama.
// Returns nil if Ollama is not installed or not running.
func OllamaModels() []string {
	out, err := exec.Command("ollama", "list").Output()
	if err != nil {
		return nil
	}
	var models []string
	for i, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if i == 0 {
			continue // skip header
		}
		fields := strings.Fields(line)
		if len(fields) > 0 {
			models = append(models, fields[0])
		}
	}
	return models
}

// RuntimeCapabilities returns what job types this host can serve.
func RuntimeCapabilities() []string {
	var caps []string
	if exec.Command("ollama", "--version").Run() == nil {
		caps = append(caps, "ollama")
	}
	if exec.Command("docker", "version").Run() == nil {
		caps = append(caps, "docker", "workspace")
	}
	return caps
}

// splitCSV splits a comma-separated nvidia-smi line and trims each field.
func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
