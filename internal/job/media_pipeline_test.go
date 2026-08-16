package job

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tokenize/gpu-agent/internal/types"
)

// mockDockerRuntime simulates the Docker Custom runtime without needing real Docker.
// It "runs" a container by writing a fake output file to the output directory,
// just like a real video/image generation container would.
type mockDockerRuntime struct {
	outputFileName string // e.g. "output.mp4" or "result.png"
	outputContent  []byte // fake file content
	modelName      string // what model was requested
	prepareCount   int
	runCount       int
}

func (m *mockDockerRuntime) Name() string { return "docker-custom-mock" }

func (m *mockDockerRuntime) Prepare(_ context.Context, a types.JobAssignment) error {
	m.prepareCount++
	m.modelName = a.ModelName
	return nil
}

func (m *mockDockerRuntime) Run(_ context.Context, a types.JobAssignment) (map[string]interface{}, error) {
	m.runCount++

	// Simulate: container writes output file to /output (which is cacheDir/output/jobID)
	// In real life, the Docker container writes to the mounted /output volume.
	// Here we write directly since we're mocking.
	result := map[string]interface{}{
		"status":  "completed",
		"backend": "docker-custom",
		"model":   a.ModelName,
	}

	if m.outputFileName != "" && m.outputContent != nil {
		result["output_file"] = m.outputFileName // the executor reads this to upload
	}

	return result, nil
}

func (m *mockDockerRuntime) Cleanup(force bool) error { return nil }

// ── Video Generation Pipeline ───────────────────────────────────────────────

func TestVideoGenerationPipeline(t *testing.T) {
	// Mock storage server (Google Cloud Storage / S3 / Supabase Storage)
	var uploadedBytes []byte
	var uploadedType string

	storageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			uploadedType = r.Header.Get("Content-Type")
			uploadedBytes, _ = io.ReadAll(r.Body)
			w.WriteHeader(200)
			return
		}
		http.Error(w, "bad method", 405)
	}))
	defer storageSrv.Close()

	cacheDir := t.TempDir()

	// Create the fake video file that the "container" would produce
	outputDir := filepath.Join(cacheDir, "output", "video-job-1")
	os.MkdirAll(outputDir, 0755)
	videoPath := filepath.Join(outputDir, "generated_video.mp4")
	videoContent := []byte("FAKE_MP4_HEADER_" + strings.Repeat("video_frame_data_", 100))
	os.WriteFile(videoPath, videoContent, 0644)

	// Mock Docker runtime that returns the video file path
	mockDocker := &mockDockerRuntime{
		outputFileName: videoPath,
		outputContent:  videoContent,
	}

	executor := &Executor{
		runtime:  mockDocker,
		gpuID:    "gpu-video-0",
		backend:  "cuda",
		cacheDir: cacheDir,
	}

	// Track progress
	var stages []string
	executor.OnProgress = func(p types.JobProgress) {
		stages = append(stages, p.Stage)
		t.Logf("[%s] %.0f%% — %s", p.Stage, p.Progress*100, p.Message)
	}

	// Execute: user wants to generate a video with LTX-Video
	result := executor.Execute(context.Background(), types.JobAssignment{
		JobID:       "video-job-1",
		ModelName:   "ltx-video",
		DockerImage: "ghcr.io/tokenize/ltx-video:latest",
		Input:       map[string]interface{}{"prompt": "A cat playing piano in a jazz club, cinematic lighting"},
		UploadURL:   storageSrv.URL + "/bucket/videos/video-job-1.mp4",
	})

	// ── Verify result ───────────────────────────────────────────────────
	if !result.Success {
		t.Fatalf("video job failed: %s", result.Error)
	}
	if result.JobID != "video-job-1" {
		t.Errorf("JobID = %q", result.JobID)
	}
	if result.GPUID != "gpu-video-0" {
		t.Errorf("GPUID = %q", result.GPUID)
	}
	if result.DurationMS < 0 {
		t.Errorf("DurationMS = %d", result.DurationMS)
	}

	// Verify the runtime was called
	if mockDocker.prepareCount != 1 {
		t.Errorf("prepare called %d times", mockDocker.prepareCount)
	}
	if mockDocker.runCount != 1 {
		t.Errorf("run called %d times", mockDocker.runCount)
	}

	// Verify upload happened
	if result.Result["uploaded"] != true {
		t.Error("video should have been uploaded")
	}
	if result.Result["upload_url"] != nil {
		t.Errorf("upload_url must not be returned, got %v", result.Result["upload_url"])
	}
	if result.Result["output_content_type"] != "video/mp4" {
		t.Errorf("output_content_type = %v", result.Result["output_content_type"])
	}
	if result.Result["output_size_bytes"] != int64(len(videoContent)) {
		t.Errorf("output_size_bytes = %v", result.Result["output_size_bytes"])
	}

	// Verify storage received the correct data
	if len(uploadedBytes) != len(videoContent) {
		t.Errorf("uploaded %d bytes, expected %d", len(uploadedBytes), len(videoContent))
	}
	if uploadedType != "video/mp4" {
		t.Errorf("Content-Type = %q, want video/mp4", uploadedType)
	}

	// Verify progress stages
	hasRunning := false
	hasUploading := false
	for _, s := range stages {
		if s == "running" {
			hasRunning = true
		}
		if s == "uploading" {
			hasUploading = true
		}
	}
	if !hasRunning {
		t.Error("missing 'running' progress stage")
	}
	if !hasUploading {
		t.Error("missing 'uploading' progress stage")
	}

	t.Logf("✅ Video pipeline: %d bytes uploaded as %s in %dms", len(uploadedBytes), uploadedType, result.DurationMS)
}

// ── Image Generation Pipeline ───────────────────────────────────────────────

func TestImageGenerationPipeline(t *testing.T) {
	var uploadedBytes []byte
	var uploadedType string

	storageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			uploadedType = r.Header.Get("Content-Type")
			uploadedBytes, _ = io.ReadAll(r.Body)
			w.WriteHeader(200)
			return
		}
	}))
	defer storageSrv.Close()

	cacheDir := t.TempDir()

	// Create fake PNG output
	outputDir := filepath.Join(cacheDir, "output", "img-job-1")
	os.MkdirAll(outputDir, 0755)
	imgPath := filepath.Join(outputDir, "generated_image.png")
	imgContent := []byte("FAKE_PNG_HEADER_" + strings.Repeat("pixel_data_", 200))
	os.WriteFile(imgPath, imgContent, 0644)

	mockDocker := &mockDockerRuntime{outputFileName: imgPath, outputContent: imgContent}
	executor := &Executor{runtime: mockDocker, gpuID: "gpu-img-0", backend: "cuda", cacheDir: cacheDir}

	result := executor.Execute(context.Background(), types.JobAssignment{
		JobID:       "img-job-1",
		ModelName:   "stable-diffusion",
		DockerImage: "ghcr.io/tokenize/stable-diffusion:latest",
		Input:       map[string]interface{}{"prompt": "A beautiful sunset over mountains, oil painting style"},
		UploadURL:   storageSrv.URL + "/bucket/images/img-job-1.png",
	})

	if !result.Success {
		t.Fatalf("image job failed: %s", result.Error)
	}
	if result.Result["uploaded"] != true {
		t.Error("image should have been uploaded")
	}
	if uploadedType != "image/png" {
		t.Errorf("Content-Type = %q, want image/png", uploadedType)
	}
	if len(uploadedBytes) != len(imgContent) {
		t.Errorf("uploaded %d bytes, expected %d", len(uploadedBytes), len(imgContent))
	}

	t.Logf("✅ Image pipeline: %d bytes uploaded as %s", len(uploadedBytes), uploadedType)
}

// ── Video with Custom LoRA Pipeline ─────────────────────────────────────────

func TestVideoWithLoRAPipeline(t *testing.T) {
	// Mock storage
	var uploadedBytes []byte
	storageSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			uploadedBytes, _ = io.ReadAll(r.Body)
			w.WriteHeader(200)
		}
	}))
	defer storageSrv.Close()

	cacheDir := t.TempDir()

	// Pre-stage the LoRA file directly (bypasses URL validation which requires
	// HTTPS, but httptest.NewServer only provides HTTP). URL validation is
	// tested separately in TestValidateCustomFileURL.
	loraContent := "FAKE_SAFETENSORS_LORA_WEIGHTS_" + strings.Repeat("x", 1000)
	loraPath := filepath.Join(cacheDir, "staging", "lora-job-1", "models/loras/anime.safetensors")
	os.MkdirAll(filepath.Dir(loraPath), 0755)
	os.WriteFile(loraPath, []byte(loraContent), 0644)

	// Create fake video output
	outputDir := filepath.Join(cacheDir, "output", "lora-job-1")
	os.MkdirAll(outputDir, 0755)
	videoPath := filepath.Join(outputDir, "lora_video.mp4")
	os.WriteFile(videoPath, []byte("LORA_ENHANCED_VIDEO_CONTENT"), 0644)

	loraVideoContent := []byte("LORA_ENHANCED_VIDEO_CONTENT")
	mockDocker := &mockDockerRuntime{outputFileName: videoPath, outputContent: loraVideoContent}
	executor := &Executor{runtime: mockDocker, gpuID: "gpu-lora-0", backend: "cuda", cacheDir: cacheDir}

	var stages []string
	executor.OnProgress = func(p types.JobProgress) {
		stages = append(stages, p.Stage)
		t.Logf("[%s] %.0f%% — %s", p.Stage, p.Progress*100, p.Message)
	}

	result := executor.Execute(context.Background(), types.JobAssignment{
		JobID:       "lora-job-1",
		ModelName:   "wan2",
		DockerImage: "ghcr.io/tokenize/wan2-video:latest",
		Input:       map[string]interface{}{"prompt": "Anime girl walking in rain, studio ghibli style"},
		// No CustomFiles — LoRA is pre-staged above to avoid HTTP vs HTTPS
		// issues with httptest.NewServer. URL validation is tested separately.
		UploadURL: storageSrv.URL + "/bucket/videos/lora-job-1.mp4",
	})

	if !result.Success {
		t.Fatalf("LoRA job failed: %s", result.Error)
	}

	// Verify pre-staged LoRA file exists
	loraData, err := os.ReadFile(loraPath)
	if err != nil {
		t.Fatalf("LoRA file not found: %v", err)
	}
	if !strings.HasPrefix(string(loraData), "FAKE_SAFETENSORS_LORA_WEIGHTS_") {
		t.Errorf("LoRA content wrong: %q", string(loraData)[:50])
	}

	// Verify video was uploaded
	if result.Result["uploaded"] != true {
		t.Error("video should have been uploaded")
	}
	if string(uploadedBytes) != "LORA_ENHANCED_VIDEO_CONTENT" {
		t.Errorf("uploaded content = %q", string(uploadedBytes))
	}

	t.Logf("✅ LoRA video pipeline: LoRA=%d bytes, video=%d bytes uploaded", len(loraData), len(uploadedBytes))
}

// ── Multiple Output Formats ─────────────────────────────────────────────────

func TestOutputContentTypeDetection(t *testing.T) {
	cases := []struct {
		filename string
		wantType string
	}{
		{"output.mp4", "video/mp4"},
		{"output.webm", "video/webm"},
		{"output.png", "image/png"},
		{"output.jpg", "image/jpeg"},
		{"output.jpeg", "image/jpeg"},
		{"output.webp", "image/webp"},
	}

	for _, tc := range cases {
		t.Run(tc.filename, func(t *testing.T) {
			var gotType string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotType = r.Header.Get("Content-Type")
				io.ReadAll(r.Body)
				w.WriteHeader(200)
			}))
			defer srv.Close()

			tmpFile := filepath.Join(t.TempDir(), tc.filename)
			os.WriteFile(tmpFile, []byte("test"), 0644)

			err := UploadOutput(context.Background(), tmpFile, srv.URL+"/upload")
			if err != nil {
				t.Fatalf("upload error: %v", err)
			}
			if gotType != tc.wantType {
				t.Errorf("Content-Type = %q, want %q", gotType, tc.wantType)
			}
		})
	}
}

// ── Timing Accuracy ─────────────────────────────────────────────────────────

func TestTimingTracking(t *testing.T) {
	// Runtime that takes a known amount of time
	slowRuntime := &timedRuntime{delay: 100 * time.Millisecond}
	executor := &Executor{runtime: slowRuntime, gpuID: "gpu-time", backend: "cpu", cacheDir: t.TempDir()}

	result := executor.Execute(context.Background(), types.JobAssignment{
		JobID: "time-job", ModelName: "test", Input: map[string]interface{}{},
	})

	if !result.Success {
		t.Fatalf("failed: %s", result.Error)
	}
	if result.DurationMS < 100 {
		t.Errorf("DurationMS = %d, expected >= 100", result.DurationMS)
	}
	if result.DurationMS > 500 {
		t.Errorf("DurationMS = %d, expected < 500 (too slow)", result.DurationMS)
	}
	t.Logf("✅ Timing: %dms (expected ~100ms)", result.DurationMS)
}

type timedRuntime struct{ delay time.Duration }

func (r *timedRuntime) Name() string                                           { return "timed" }
func (r *timedRuntime) Prepare(_ context.Context, _ types.JobAssignment) error { return nil }
func (r *timedRuntime) Run(_ context.Context, _ types.JobAssignment) (map[string]interface{}, error) {
	time.Sleep(r.delay)
	return map[string]interface{}{"status": "completed"}, nil
}
func (r *timedRuntime) Cleanup(force bool) error { return nil }
