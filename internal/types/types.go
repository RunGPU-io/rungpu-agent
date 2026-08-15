// Package types defines the agent's configuration and the wire protocol shared
// with the pool coordinator (raw WebSocket, JSON frames of {type, ...payload}).
package types

// ── Configuration (persisted as YAML) ────────────────────────────────────────

type MetricsConfig struct {
	EnableGPUMonitoring    bool `yaml:"enable_gpu_monitoring"`
	MonitoringIntervalSecs int  `yaml:"monitoring_interval_secs"`
}

type SecurityConfig struct {
	AllowAnyImage     bool     `yaml:"allow_any_image"`    // if true, skip image allowlist (DANGEROUS)
	TrustedRegistries []string `yaml:"trusted_registries"` // additional trusted registries
	MaxMemoryGB       int      `yaml:"max_memory_gb"`      // container memory limit (0 = no limit)
	MaxCPUs           float64  `yaml:"max_cpus"`           // container CPU limit (0 = no limit)
	AllowHostNetwork  bool     `yaml:"allow_host_network"` // if true, run containers with --network host (DANGEROUS)
	HFToken           string   `yaml:"-"`                  // runtime-only HuggingFace token from HF_TOKEN
}

type Config struct {
	APIKey                string   `yaml:"api_key"`
	MachineID             string   `yaml:"machine_id"`
	PoolURL               string   `yaml:"pool_url"`
	GPUIDs                []string `yaml:"gpu_ids"`
	PricePerMinute        float64  `yaml:"price_per_minute"`
	ContributeFree        bool     `yaml:"contribute_free,omitempty"`
	ModelCacheDir         string   `yaml:"model_cache_dir"`
	MaxModelCacheGB       int      `yaml:"max_model_cache_gb"`
	CleanupIntervalHours  int      `yaml:"cleanup_interval_hours"`
	CustomAssetTTLDays    int      `yaml:"custom_asset_ttl_days"`
	MaxCustomAssetCacheGB int      `yaml:"max_custom_asset_cache_gb"`
	HeartbeatIntervalSecs int      `yaml:"heartbeat_interval_secs"`
	AllowCPUServing       bool     `yaml:"allow_cpu_serving"`

	// GPUDevice scopes containers to a specific GPU passed to `docker --gpus`.
	// Empty or "all" exposes every GPU; otherwise it is passed as
	// `--gpus device=<GPUDevice>` (e.g. "0" or a GPU UUID) so a job only sees
	// the reserved device. Recommended on multi-GPU hosts.
	GPUDevice string `yaml:"gpu_device"`

	// JobTimeoutMinutes caps how long a batch (non-workspace) job may run before
	// the container is killed. 0 → default (60 minutes). A job may request a
	// shorter value via Parameters["timeout_minutes"].
	JobTimeoutMinutes int `yaml:"job_timeout_minutes"`

	// MaxCustomFileGB caps the size of any single downloaded custom file
	// (LoRA, checkpoint, workflow). 0 → unlimited.
	MaxCustomFileGB int `yaml:"max_custom_file_gb"`

	Metrics  MetricsConfig  `yaml:"metrics"`
	Security SecurityConfig `yaml:"security"`
}

// ── GPU detection / monitoring ───────────────────────────────────────────────

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

// ── Wire protocol ─────────────────────────────────────────────────────────────

// Envelope is used to peek at the "type" field of any inbound message.
type Envelope struct {
	Type string `json:"type"`
}

// JobAssignment is pushed by the server down the open socket.
type JobAssignment struct {
	Type           string                 `json:"type"`
	JobID          string                 `json:"job_id"`
	ModelName      string                 `json:"model_name"`
	ModelURL       string                 `json:"model_url,omitempty"`
	Input          map[string]interface{} `json:"input"`
	Parameters     map[string]interface{} `json:"parameters"`
	VRAMRequiredGB float64                `json:"vram_required_gb"`

	// ── Docker image source ─────────────────────────────────────────────
	// The image to run. Can be:
	//   - A HuggingFace Docker image URL (e.g. "registry.hf.space/org/model")
	//   - A Docker Hub / GHCR image (e.g. "ghcr.io/tokenize/ltx-video:latest")
	//   - Empty → derived from ModelName for well-known models
	DockerImage string `json:"docker_image,omitempty"`

	// ── Custom files to inject into the container ───────────────────────
	// Each entry is a URL to download → mount path inside the container.
	// Used for LoRAs, ComfyUI workflows, custom checkpoints, etc.
	//   e.g. [{"url": "https://civitai.com/api/download/models/123", "path": "/models/loras/my.safetensors"}]
	CustomFiles []CustomFile `json:"custom_files,omitempty"`

	// ── Output upload ───────────────────────────────────────────────────
	// Where the agent should upload generated files (images, videos).
	// The agent uploads to this pre-signed URL and reports the public URL in the result.
	UploadURL string `json:"upload_url,omitempty"`

	// ── Workspace mode ──────────────────────────────────────────────────
	// If true, the container stays running with exposed ports (for ComfyUI, Jupyter, etc.)
	// instead of running a batch job and exiting.
	Workspace bool     `json:"workspace,omitempty"`
	Ports     []string `json:"ports,omitempty"` // e.g. ["8188:8188"]
}

// JobControl is a server→agent control message to stop/cancel a running job or
// tear down a workspace container. Matched on Type ∈ {job_cancel, job_stop, stop_job}.
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

// CustomFile is a file to download and inject into the container.
type CustomFile struct {
	URL  string `json:"url"`  // download URL (HuggingFace, Civitai, direct link)
	Path string `json:"path"` // mount path inside container (e.g. "/models/loras/my.safetensors")
	Name string `json:"name"` // human-readable name (optional)
}

// JobResult is sent back by the agent when a job finishes.
type JobResult struct {
	Type       string                 `json:"type"`
	JobID      string                 `json:"job_id"`
	GPUID      string                 `json:"gpu_id"`
	Success    bool                   `json:"success"`
	Result     map[string]interface{} `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
	DurationMS int64                  `json:"duration_ms"`
}

// JobProgress is sent periodically while a job is running.
type JobProgress struct {
	Type     string  `json:"type"` // "job_progress"
	JobID    string  `json:"job_id"`
	GPUID    string  `json:"gpu_id"`
	Stage    string  `json:"stage"`    // "pulling_image" | "downloading_files" | "running" | "uploading"
	Progress float64 `json:"progress"` // 0.0 - 1.0
	Message  string  `json:"message"`
}

// RegisterMessage announces a GPU to the pool. api_key is injected server-side
// from the authenticated connection, so it is not sent here.
type RegisterMessage struct {
	Type                string   `json:"type"` // "gpu_register"
	GPUID               string   `json:"gpu_id"`
	MachineID           string   `json:"machine_id,omitempty"`
	DeviceIndex         int      `json:"device_index"`
	DetectedDeviceCount int      `json:"detected_device_count"`
	GPUType             string   `json:"gpu_type"`
	Backend             string   `json:"backend"` // "cuda" | "metal" | "cpu"
	VRAMGB              float64  `json:"vram_gb"`
	PricePerMinute      float64  `json:"price_per_minute"`
	ModelsCached        []string `json:"models_cached"`
	DriverVersion       string   `json:"driver_version"`

	// ── Runtime capabilities (what job types this host can serve) ────────
	Capabilities []string `json:"capabilities,omitempty"`  // ["ollama","docker","workspace"]
	OllamaModels []string `json:"ollama_models,omitempty"` // models pulled in Ollama (not just cache dir)
}

// HeartbeatMessage reports liveness and current availability.
type HeartbeatMessage struct {
	Type            string   `json:"type"` // "gpu_heartbeat"
	GPUID           string   `json:"gpu_id"`
	AvailableVRAMGB float64  `json:"available_vram_gb"`
	CurrentJobs     int      `json:"current_jobs"`
	ModelsCached    []string `json:"models_cached"`

	OllamaModels []string `json:"ollama_models,omitempty"` // current Ollama model list
}
