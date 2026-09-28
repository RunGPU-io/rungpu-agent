package gpu

import (
	"context"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func Detect() []types.GPUInfo {
	if gpus := detectNvidia(); len(gpus) > 0 {
		return gpus
	}
	if runtime.GOOS == "darwin" {
		if gpus := detectMacOS(); len(gpus) > 0 {
			return gpus
		}
	}

	return []types.GPUInfo{{
		Index:             0,
		Name:              "cpu-only",
		ComputeCapability: "n/a",
		MemoryMB:          0,
		DriverVersion:     "n/a",
	}}
}

func detectNvidia() []types.GPUInfo {
	out, err := commandOutput("nvidia-smi",
		"--query-gpu=index,name,memory.total,driver_version",
		"--format=csv,noheader,nounits")
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

func detectMacOS() []types.GPUInfo {
	name := sysctl("machdep.cpu.brand_string")
	if name == "" {
		name = "Apple GPU"
	}

	if gpuName := detectMacGPUName(); gpuName != "" {
		name = gpuName
	}

	var memMB uint64
	if v := sysctl("hw.memsize"); v != "" {
		if bytes, err := strconv.ParseUint(v, 10, 64); err == nil {
			memMB = bytes / (1024 * 1024)
		}
	}

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

func detectMacGPUName() string {
	out, err := exec.Command("system_profiler", "SPDisplaysDataType", "-detailLevel", "mini").Output()
	if err != nil {
		return ""
	}

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

type Monitor struct {
	gpus      []types.GPUInfo
	hasNvidia bool
}

func NewMonitor() *Monitor {
	gpus := Detect()
	hasNvidia := len(gpus) > 0 && gpus[0].ComputeCapability == "cuda"
	return &Monitor{gpus: gpus, hasNvidia: hasNvidia}
}

func (m *Monitor) GPUs() []types.GPUInfo { return m.gpus }

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

func (m *Monitor) CollectMetrics() []types.GPUMetrics {
	if m.hasNvidia {
		if metrics := collectNvidiaMetrics(); len(metrics) > 0 {
			return metrics
		}
	}

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
	out, err := commandOutput("nvidia-smi",
		"--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits")
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

type SystemInfo struct {
	CPUModel   string
	CPUCores   int
	RAMTotalGB float64
	OSInfo     string
}

func DetectSystem() SystemInfo {
	info := SystemInfo{
		OSInfo:   runtime.GOOS + "/" + runtime.GOARCH,
		CPUCores: runtime.NumCPU(),
	}

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

func RAMUsage() (usedGB, totalGB float64) {
	switch runtime.GOOS {
	case "darwin":
		if v := sysctl("hw.memsize"); v != "" {
			if bytes, err := strconv.ParseUint(v, 10, 64); err == nil {
				totalGB = float64(bytes) / (1024 * 1024 * 1024)
			}
		}

		if out, err := exec.Command("vm_stat").Output(); err == nil {
			var active, wired, compressed uint64
			pageSize := uint64(16384)
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

	parts := strings.SplitN(line, ":", 2)
	if len(parts) < 2 {
		return 0
	}
	s := strings.TrimSpace(parts[1])
	s = strings.TrimSuffix(s, ".")
	v, _ := strconv.ParseUint(s, 10, 64)
	return v
}

func OllamaModels() []string {
	out, err := commandOutput("ollama", "list")
	if err != nil {
		return nil
	}
	var models []string
	for i, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if i == 0 {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) > 0 {
			models = append(models, fields[0])
		}
	}
	return models
}

func RuntimeCapabilities() []string {
	var caps []string
	if commandRuns("ollama", "--version") {
		caps = append(caps, "ollama")
	}
	if commandRuns("docker", "version") {
		caps = append(caps, "docker", "workspace")
	}
	return caps
}

func commandOutput(name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).Output()
}

func commandRuns(name string, args ...string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).Run() == nil
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
