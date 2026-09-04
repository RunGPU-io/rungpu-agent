package types

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
)

const JobProtocolVersion = 3

var safeJobID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

type MetricsConfig struct {
	EnableGPUMonitoring    bool `yaml:"enable_gpu_monitoring"`
	MonitoringIntervalSecs int  `yaml:"monitoring_interval_secs"`
}

type SecurityConfig struct {
	AllowAnyImage     bool     `yaml:"allow_any_image"`
	TrustedRegistries []string `yaml:"trusted_registries"`
	MaxMemoryGB       int      `yaml:"max_memory_gb"`
	MaxCPUs           float64  `yaml:"max_cpus"`
	AllowHostNetwork  bool     `yaml:"allow_host_network"`
	HFToken           string   `yaml:"-"`
}

type ScheduleWindow struct {
	StartHour int      `yaml:"start_hour" json:"start_hour"`
	EndHour   int      `yaml:"end_hour" json:"end_hour"`
	Days      []string `yaml:"days,omitempty" json:"days,omitempty"`
}

type ScheduleConfig struct {
	Enabled   bool             `yaml:"enabled"`
	StartHour int              `yaml:"start_hour"`
	EndHour   int              `yaml:"end_hour"`
	Timezone  string           `yaml:"timezone"`
	Windows   []ScheduleWindow `yaml:"windows,omitempty"`
}

type Config struct {
	APIKey                string         `yaml:"api_key"`
	MachineID             string         `yaml:"machine_id"`
	PoolURL               string         `yaml:"pool_url"`
	GPUIDs                []string       `yaml:"gpu_ids"`
	PricePerMinute        float64        `yaml:"price_per_minute"`
	ContributeFree        bool           `yaml:"contribute_free,omitempty"`
	ModelCacheDir         string         `yaml:"model_cache_dir"`
	MaxModelCacheGB       int            `yaml:"max_model_cache_gb"`
	CleanupIntervalHours  int            `yaml:"cleanup_interval_hours"`
	CustomAssetTTLDays    int            `yaml:"custom_asset_ttl_days"`
	MaxCustomAssetCacheGB int            `yaml:"max_custom_asset_cache_gb"`
	HeartbeatIntervalSecs int            `yaml:"heartbeat_interval_secs"`
	AllowCPUServing       bool           `yaml:"allow_cpu_serving"`
	Paused                bool           `yaml:"paused,omitempty"`
	Schedule              ScheduleConfig `yaml:"schedule"`

	GPUDevice string `yaml:"gpu_device"`

	JobTimeoutMinutes int `yaml:"job_timeout_minutes"`

	MaxCustomFileGB int `yaml:"max_custom_file_gb"`

	Metrics  MetricsConfig  `yaml:"metrics"`
	Security SecurityConfig `yaml:"security"`
}

type GPUInfo struct {
	Index             int
	Name              string
	ComputeCapability string
	MemoryMB          uint64
	DriverVersion     string
}

type GPUMetrics struct {
	GPUIndex           int
	UtilizationPercent float64
	MemoryUsedMB       uint64
	MemoryTotalMB      uint64
	TemperatureC       *float64
	PowerDrawW         *float64
}

type Envelope struct {
	Type string `json:"type"`
}

type JobAssignment struct {
	ProtocolVersion int                    `json:"protocol_version,omitempty"`
	Type            string                 `json:"type"`
	JobID           string                 `json:"job_id"`
	ModelName       string                 `json:"model_name"`
	Runtime         string                 `json:"runtime,omitempty"`
	Source          string                 `json:"source,omitempty"`
	ModelURL        string                 `json:"model_url,omitempty"`
	Input           map[string]interface{} `json:"input"`
	Parameters      map[string]interface{} `json:"parameters"`
	VRAMRequiredGB  float64                `json:"vram_required_gb"`

	DockerImage string `json:"docker_image,omitempty"`

	CustomFiles    []CustomFile `json:"custom_files,omitempty"`
	WorkflowJSON   string       `json:"workflow_json,omitempty"`
	WorkflowSHA256 string       `json:"workflow_sha256,omitempty"`

	UploadURL string `json:"upload_url,omitempty"`

	Workspace bool     `json:"workspace,omitempty"`
	Ports     []string `json:"ports,omitempty"`
}

func (a JobAssignment) Validate() error {
	if a.ProtocolVersion < 0 || a.ProtocolVersion > JobProtocolVersion {
		return fmt.Errorf("unsupported job protocol version %d", a.ProtocolVersion)
	}
	if !safeJobID.MatchString(a.JobID) {
		return fmt.Errorf("invalid job_id")
	}
	if a.ModelName == "" {
		return fmt.Errorf("model_name is required")
	}
	if a.ProtocolVersion >= 1 && a.Runtime == "" {
		return fmt.Errorf("runtime is required by job protocol version %d", a.ProtocolVersion)
	}
	if a.ProtocolVersion >= 2 {
		for _, file := range a.CustomFiles {
			if !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(file.SHA256) {
				return fmt.Errorf("custom file sha256 is required by job protocol version %d", a.ProtocolVersion)
			}
		}
		if a.WorkflowJSON != "" {
			if len(a.WorkflowJSON) > 256*1024 || !json.Valid([]byte(a.WorkflowJSON)) {
				return fmt.Errorf("workflow_json must be valid JSON no larger than 256 KiB")
			}
			if !regexp.MustCompile(`^[a-f0-9]{64}$`).MatchString(a.WorkflowSHA256) {
				return fmt.Errorf("workflow_sha256 is required for inline workflows")
			}
			for _, file := range a.CustomFiles {
				if filepath.Clean(file.Path) == "workflows/workflow.json" {
					return fmt.Errorf("custom files cannot override the inline workflow")
				}
			}
		}
	}
	return nil
}

type JobControl struct {
	Type  string `json:"type"`
	JobID string `json:"job_id"`
}

type AssetCleanupRequest struct {
	Type       string   `json:"type"`
	RequestID  string   `json:"request_id"`
	Phase      string   `json:"phase"`
	Categories []string `json:"categories"`
}

type AssetCleanupResult struct {
	Type       string                 `json:"type"`
	RequestID  string                 `json:"request_id"`
	Phase      string                 `json:"phase"`
	Success    bool                   `json:"success"`
	Error      string                 `json:"error,omitempty"`
	Categories map[string]interface{} `json:"categories,omitempty"`
	TotalBytes int64                  `json:"total_bytes,omitempty"`
	ActiveJobs int                    `json:"active_jobs,omitempty"`
}

type CustomFile struct {
	URL    string `json:"url"`
	Path   string `json:"path"`
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

type JobResult struct {
	Type       string                 `json:"type"`
	JobID      string                 `json:"job_id"`
	GPUID      string                 `json:"gpu_id"`
	Success    bool                   `json:"success"`
	Result     map[string]interface{} `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
	DurationMS int64                  `json:"duration_ms"`
}

type JobProgress struct {
	Type     string  `json:"type"`
	JobID    string  `json:"job_id"`
	GPUID    string  `json:"gpu_id"`
	Stage    string  `json:"stage"`
	Progress float64 `json:"progress"`
	Message  string  `json:"message"`
}

type RegisterMessage struct {
	Type                string   `json:"type"`
	GPUID               string   `json:"gpu_id"`
	MachineID           string   `json:"machine_id,omitempty"`
	DeviceIndex         int      `json:"device_index"`
	DetectedDeviceCount int      `json:"detected_device_count"`
	GPUType             string   `json:"gpu_type"`
	Backend             string   `json:"backend"`
	VRAMGB              float64  `json:"vram_gb"`
	PricePerMinute      float64  `json:"price_per_minute"`
	ModelsCached        []string `json:"models_cached"`
	DriverVersion       string   `json:"driver_version"`

	Capabilities []string `json:"capabilities"`
	OllamaModels []string `json:"ollama_models,omitempty"`
}

type HeartbeatMessage struct {
	Type            string   `json:"type"`
	GPUID           string   `json:"gpu_id"`
	AvailableVRAMGB float64  `json:"available_vram_gb"`
	CurrentJobs     int      `json:"current_jobs"`
	ModelsCached    []string `json:"models_cached"`

	OllamaModels []string `json:"ollama_models,omitempty"`
}
