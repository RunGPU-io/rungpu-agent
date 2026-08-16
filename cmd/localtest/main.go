// Command localtest proves the GPU agent works end-to-end on your local machine
// without connecting to the pool coordinator.
//
// There are two personas in the Tokenize marketplace:
//
//   GPU HOST (you, running this agent):
//     Your machine runs the agent. The agent detects your GPU, pulls models/images,
//     manages Docker containers, and serves inference. You earn money.
//
//   CUSTOMER (someone on the internet):
//     They submit jobs via the Tokenize API or dashboard. The pool coordinator
//     routes their request to your agent. They get back results.
//
// This local test simulates both sides:
//   1. It starts the agent (host side)
//   2. It exposes an HTTP inference URL (what the customer hits)
//
// Usage:
//
//	# One-shot: pull model, run inference, print result
//	go run ./cmd/localtest --model llama3.2 --prompt "Hello world"
//
//	# Serve mode: start agent + expose inference URL for customers
//	go run ./cmd/localtest --model llama3.2 --serve
//	# -> http://localhost:8090/v1/chat/completions  (customer hits this)
//
//	# Workspace: start ComfyUI with persistent model storage
//	go run ./cmd/localtest --model comfyui --workspace
//	# -> http://localhost:8188  (customer opens this in browser)
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/tokenize/gpu-agent/internal/gpu"
	"github.com/tokenize/gpu-agent/internal/job"
	"github.com/tokenize/gpu-agent/internal/types"
)

func main() {
	model := flag.String("model", "llama3.2", "Model to run (llama3.2, comfyui, stable-diffusion, etc.)")
	prompt := flag.String("prompt", "Explain what a GPU agent is in 2 sentences.", "Prompt for one-shot inference")
	lora := flag.String("lora", "", "LoRA URL to download (from Civitai or HuggingFace)")
	loraName := flag.String("lora-name", "custom_lora", "Name for the LoRA file")
	serve := flag.Bool("serve", false, "Start agent + expose HTTP inference URL (simulates full flow)")
	port := flag.Int("port", 8090, "Port for the inference server (--serve mode)")
	workspace := flag.Bool("workspace", false, "Start a Docker workspace (ComfyUI, Jupyter) with persistent storage")
	dockerImage := flag.String("docker-image", "", "Explicit Docker image to use")
	timeout := flag.Duration("timeout", 10*time.Minute, "Max time to wait for one-shot jobs")
	flag.Parse()

	fmt.Println("==========================================================")
	fmt.Println("  Tokenize GPU Agent - Local End-to-End Test")
	fmt.Println("==========================================================")

	// Detect GPU (same as production agent)
	monitor := gpu.NewMonitor()
	gpus := monitor.GPUs()
	backend := monitor.Backend()
	fmt.Printf("\nGPU: %s (%d MB)\n", gpus[0].Name, gpus[0].MemoryMB)
	fmt.Printf("Backend: %s\n", backend)

	// Create executor (same as production agent)
	cacheDir := os.TempDir() + "/tokenize-localtest"
	os.MkdirAll(cacheDir, 0755)
	fmt.Printf("Cache: %s\n", cacheDir)

	executor, err := job.NewExecutor(cacheDir, 50, "gpu-local-0", backend)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create executor: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Runtime: %s\n\n", executor.Runtime())

	// Wire up progress reporting
	executor.OnProgress = func(p types.JobProgress) {
		fmt.Printf("  [%s] %3.0f%% - %s\n", p.Stage, p.Progress*100, p.Message)
	}

	// Signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nInterrupted - cleaning up...")
		cancel()
	}()

	// Route to the right mode
	switch {
	case *serve:
		runServeMode(ctx, executor, *model, *port, *dockerImage)
	case *workspace:
		runWorkspaceMode(ctx, executor, *model, *dockerImage)
	default:
		runOneShotMode(ctx, executor, *model, *prompt, *lora, *loraName, *dockerImage, *timeout)
	}
}

// =============================================================================
// Mode 1: One-shot inference
// Proves: model pull -> inference -> result. Exits when done.
// =============================================================================

func runOneShotMode(ctx context.Context, executor *job.Executor, model, prompt, lora, loraName, dockerImage string, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	assignment := types.JobAssignment{
		Type:        "job_assignment",
		JobID:       fmt.Sprintf("local-test-%d", time.Now().Unix()),
		ModelName:   model,
		DockerImage: dockerImage,
		Input:       map[string]interface{}{"prompt": prompt},
	}

	if lora != "" {
		assignment.CustomFiles = []types.CustomFile{
			{URL: lora, Path: fmt.Sprintf("models/loras/%s.safetensors", loraName), Name: loraName},
		}
		fmt.Printf("LoRA: %s -> %s\n", lora, loraName)
	}

	fmt.Printf("Running job: %s (model=%s)\n\n", assignment.JobID, model)
	start := time.Now()
	result := executor.Execute(ctx, assignment)
	duration := time.Since(start)

	printResult(result, duration)

	if !result.Success {
		os.Exit(1)
	}
}

// =============================================================================
// Mode 2: Serve mode (--serve)
//
// This is the full end-to-end proof:
//   HOST SIDE:  Agent pulls model, starts Ollama, keeps it warm
//   CUSTOMER SIDE: HTTP API at localhost:8090 - OpenAI-compatible
//
// In production, the customer's request arrives via WebSocket from the pool
// coordinator. Here we simulate that with a local HTTP server.
// =============================================================================

func runServeMode(ctx context.Context, executor *job.Executor, model string, port int, dockerImage string) {
	fmt.Println("Starting inference server...")
	fmt.Printf("   Model: %s\n", model)

	// Step 1: Pull model + verify it works
	fmt.Println("\nPreparing model (pulling if not cached)...")
	prepStart := time.Now()

	prepResult := executor.Execute(ctx, types.JobAssignment{
		Type:        "job_assignment",
		JobID:       fmt.Sprintf("warmup-%d", time.Now().Unix()),
		ModelName:   model,
		DockerImage: dockerImage,
		Input:       map[string]interface{}{"prompt": "Say OK."},
	})
	if !prepResult.Success {
		fmt.Fprintf(os.Stderr, "Model preparation failed: %s\n", prepResult.Error)
		fmt.Fprintf(os.Stderr, "Make sure the model is compatible with your backend (%s)\n", executor.Backend())
		os.Exit(1)
	}
	fmt.Printf("Model ready (took %s, %dms inference)\n", time.Since(prepStart).Round(time.Millisecond), prepResult.DurationMS)

	if resp, ok := prepResult.Result["response"].(string); ok {
		preview := resp
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		fmt.Printf("   Warmup response: %s\n", strings.TrimSpace(preview))
	}

	// Step 2: Start HTTP server (the "inference URL" customers use)
	addr := fmt.Sprintf(":%d", port)
	baseURL := fmt.Sprintf("http://localhost:%d", port)

	fmt.Println()
	fmt.Println("==========================================================")
	fmt.Println("INFERENCE URL IS LIVE - customers can send requests here")
	fmt.Println()
	fmt.Printf("   Base URL:          %s\n", baseURL)
	fmt.Printf("   Chat completions:  %s/v1/chat/completions\n", baseURL)
	fmt.Printf("   Generate:          %s/v1/generate\n", baseURL)
	fmt.Printf("   Models:            %s/v1/models\n", baseURL)
	fmt.Printf("   Health:            %s/health\n", baseURL)
	fmt.Println()
	fmt.Printf("   Model:   %s\n", model)
	fmt.Printf("   Backend: %s\n", executor.Backend())
	fmt.Printf("   Runtime: %s\n", executor.Runtime())
	fmt.Println()
	fmt.Println("   -- Customer sends a request like this --")
	fmt.Println()
	fmt.Printf("   curl -s %s/v1/chat/completions \\\n", baseURL)
	fmt.Println("     -H 'Content-Type: application/json' \\")
	fmt.Printf("     -d '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"What is Tokenize?\"}]}' | jq .\n", model)
	fmt.Println()
	fmt.Println("   Or with Python (OpenAI client):")
	fmt.Println()
	fmt.Printf("   from openai import OpenAI\n")
	fmt.Printf("   client = OpenAI(base_url=\"%s/v1\", api_key=\"local\")\n", baseURL)
	fmt.Printf("   r = client.chat.completions.create(model=\"%s\", messages=[{\"role\":\"user\",\"content\":\"Hi\"}])\n", model)
	fmt.Println("   print(r.choices[0].message.content)")
	fmt.Println()
	fmt.Println("   Press Ctrl+C to stop.")
	fmt.Println("==========================================================")

	srv := &inferenceServer{
		executor: executor,
		model:    model,
		docker:   dockerImage,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("/v1/generate", srv.handleGenerate)
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/", srv.handleRoot)

	httpServer := &http.Server{
		Addr:         addr,
		Handler:      corsMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 5 * time.Minute,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		<-ctx.Done()
		fmt.Println("\nShutting down inference server...")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		httpServer.Shutdown(shutdownCtx)
	}()

	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Server stopped.")
}

// =============================================================================
// Mode 3: Workspace (--workspace)
//
// Starts a long-running Docker container (ComfyUI, Jupyter, etc.) with
// persistent Docker named volumes. Models, custom nodes, and extensions
// are stored in these volumes and survive container restarts - no
// re-downloading.
//
// This is what happens in production when a customer rents a workspace.
// The agent starts the container, the customer gets the access URL.
// =============================================================================

func runWorkspaceMode(ctx context.Context, executor *job.Executor, model, dockerImage string) {
	fmt.Println("Starting workspace...")
	fmt.Printf("   Model: %s\n", model)
	if dockerImage != "" {
		fmt.Printf("   Image: %s\n", dockerImage)
	}
	fmt.Println()
	fmt.Println("   Docker named volumes are used for persistent storage.")
	fmt.Println("   Models, custom nodes, and outputs survive container restarts.")
	fmt.Println("   No re-downloading needed between sessions.")

	assignment := types.JobAssignment{
		Type:        "job_assignment",
		JobID:       fmt.Sprintf("ws-%d", time.Now().Unix()),
		ModelName:   model,
		DockerImage: dockerImage,
		Workspace:   true,
		Input:       map[string]interface{}{},
	}

	fmt.Printf("\nStarting workspace: %s\n\n", assignment.JobID)
	start := time.Now()
	result := executor.Execute(ctx, assignment)
	duration := time.Since(start)

	if !result.Success {
		fmt.Println("\n==========================================================")
		fmt.Printf("WORKSPACE FAILED in %s\n", duration.Round(time.Millisecond))
		fmt.Printf("   Error: %s\n", result.Error)
		fmt.Println("==========================================================")
		os.Exit(1)
	}

	fmt.Println("\n==========================================================")
	fmt.Println("WORKSPACE RUNNING - customer accesses it at these URLs:")
	fmt.Println()

	if urls, ok := result.Result["access_urls"].([]interface{}); ok {
		for _, u := range urls {
			fmt.Printf("   %s\n", u)
		}
	} else if urls, ok := result.Result["access_urls"].([]string); ok {
		for _, u := range urls {
			fmt.Printf("   %s\n", u)
		}
	}

	if msg, ok := result.Result["message"].(string); ok {
		fmt.Printf("\n   %s\n", msg)
	}
	if instr, ok := result.Result["instructions"].(string); ok {
		fmt.Printf("   %s\n", instr)
	}

	// Show persistent volume info
	if vols, ok := result.Result["persistent_volumes"].([]string); ok {
		fmt.Println("\n   Persistent Docker volumes (survive container restarts):")
		for _, v := range vols {
			fmt.Printf("      %s\n", v)
		}
	}
	if note, ok := result.Result["storage_note"].(string); ok {
		fmt.Printf("\n   %s\n", note)
	}

	containerName := ""
	if c, ok := result.Result["container"].(string); ok {
		containerName = c
		fmt.Printf("\n   Container: %s\n", c)
		fmt.Printf("   View logs: docker logs -f %s\n", c)
		fmt.Printf("   Stop:      docker stop %s\n", c)
	}

	fmt.Println("\n   Press Ctrl+C to stop the workspace.")
	fmt.Println("==========================================================")

	// Block until interrupted
	<-ctx.Done()

	if containerName != "" {
		fmt.Printf("\nStopping workspace container %s...\n", containerName)
		stopCtx, stopCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer stopCancel()
		exec.CommandContext(stopCtx, "docker", "stop", containerName).Run()
		exec.CommandContext(stopCtx, "docker", "rm", "-f", containerName).Run()
		fmt.Println("Container stopped. Docker volumes are preserved - models are still cached.")
	}
}

// =============================================================================
// Inference server - OpenAI-compatible HTTP API
// This is what the customer's requests hit.
// =============================================================================

type inferenceServer struct {
	executor *job.Executor
	model    string
	docker   string
	jobSeq   int64 // atomic counter
}

// OpenAI-compatible request/response types

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type generateRequest struct {
	Model      string                 `json:"model"`
	Prompt     string                 `json:"prompt"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Stream     bool                   `json:"stream"`
}

type generateResponse struct {
	ID         string                 `json:"id"`
	Model      string                 `json:"model"`
	Response   string                 `json:"response"`
	DurationMS int64                  `json:"duration_ms"`
	Backend    string                 `json:"backend"`
	Details    map[string]interface{} `json:"details,omitempty"`
}

// Handlers

func (s *inferenceServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"service": "tokenize-gpu-agent",
		"version": "1.0.0",
		"model":   s.model,
		"backend": s.executor.Backend(),
		"endpoints": map[string]string{
			"chat_completions": "/v1/chat/completions",
			"generate":         "/v1/generate",
			"models":           "/v1/models",
			"health":           "/health",
		},
	})
}

func (s *inferenceServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "healthy",
		"model":   s.model,
		"backend": s.executor.Backend(),
		"runtime": s.executor.Runtime(),
	})
}

func (s *inferenceServer) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       s.model,
				"object":   "model",
				"owned_by": "tokenize-agent",
				"backend":  s.executor.Backend(),
			},
		},
	})
}

func (s *inferenceServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error": map[string]string{"message": "method not allowed", "type": "invalid_request"},
		})
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]string{"message": fmt.Sprintf("invalid JSON: %s", err), "type": "invalid_request"},
		})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = s.model
	}

	prompt := chatMessagesToPrompt(req.Messages)

	params := map[string]interface{}{}
	if req.Temperature != nil {
		params["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		params["num_predict"] = *req.MaxTokens
	}

	seq := atomic.AddInt64(&s.jobSeq, 1)
	jobID := fmt.Sprintf("chat-%d-%d", time.Now().Unix(), seq)

	assignment := types.JobAssignment{
		Type:        "job_assignment",
		JobID:       jobID,
		ModelName:   modelName,
		DockerImage: s.docker,
		Input:       map[string]interface{}{"prompt": prompt},
		Parameters:  params,
	}

	start := time.Now()
	result := s.executor.Execute(r.Context(), assignment)
	duration := time.Since(start)

	if !result.Success {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": map[string]string{"message": result.Error, "type": "inference_error"},
		})
		fmt.Printf("  [ERR] %s failed (%dms): %s\n", jobID, duration.Milliseconds(), result.Error)
		return
	}

	responseText := extractResponseText(result.Result)
	evalCount := extractEvalCount(result.Result)

	writeJSON(w, http.StatusOK, chatResponse{
		ID:      "chatcmpl-" + jobID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []chatChoice{
			{
				Index:        0,
				Message:      chatMessage{Role: "assistant", Content: responseText},
				FinishReason: "stop",
			},
		},
		Usage: chatUsage{
			PromptTokens:     len(prompt) / 4,
			CompletionTokens: evalCount,
			TotalTokens:      len(prompt)/4 + evalCount,
		},
	})

	fmt.Printf("  [OK] %s -> %dms, %d tokens\n", jobID, duration.Milliseconds(), evalCount)
}

func (s *inferenceServer) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]interface{}{
			"error": map[string]string{"message": "method not allowed", "type": "invalid_request"},
		})
		return
	}

	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]string{"message": fmt.Sprintf("invalid JSON: %s", err), "type": "invalid_request"},
		})
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = s.model
	}

	seq := atomic.AddInt64(&s.jobSeq, 1)
	jobID := fmt.Sprintf("gen-%d-%d", time.Now().Unix(), seq)

	assignment := types.JobAssignment{
		Type:        "job_assignment",
		JobID:       jobID,
		ModelName:   modelName,
		DockerImage: s.docker,
		Input:       map[string]interface{}{"prompt": req.Prompt},
		Parameters:  req.Parameters,
	}

	start := time.Now()
	result := s.executor.Execute(r.Context(), assignment)
	duration := time.Since(start)

	if !result.Success {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": map[string]string{"message": result.Error, "type": "inference_error"},
		})
		fmt.Printf("  [ERR] %s failed (%dms): %s\n", jobID, duration.Milliseconds(), result.Error)
		return
	}

	responseText := extractResponseText(result.Result)

	writeJSON(w, http.StatusOK, generateResponse{
		ID:         jobID,
		Model:      modelName,
		Response:   responseText,
		DurationMS: result.DurationMS,
		Backend:    s.executor.Backend(),
		Details:    result.Result,
	})

	fmt.Printf("  [OK] %s -> %dms\n", jobID, duration.Milliseconds())
}

// =============================================================================
// Helpers
// =============================================================================

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// chatMessagesToPrompt converts OpenAI-style chat messages into a single prompt.
func chatMessagesToPrompt(messages []chatMessage) string {
	if len(messages) == 0 {
		return ""
	}
	if len(messages) == 1 && messages[0].Role == "user" {
		return messages[0].Content
	}

	var b strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			b.WriteString(fmt.Sprintf("System: %s\n\n", msg.Content))
		case "user":
			b.WriteString(fmt.Sprintf("User: %s\n\n", msg.Content))
		case "assistant":
			b.WriteString(fmt.Sprintf("Assistant: %s\n\n", msg.Content))
		}
	}
	b.WriteString("Assistant: ")
	return b.String()
}

func extractResponseText(result map[string]interface{}) string {
	if result == nil {
		return ""
	}
	if resp, ok := result["response"].(string); ok {
		return resp
	}
	if output, ok := result["output"].(string); ok {
		return output
	}
	return ""
}

func extractEvalCount(result map[string]interface{}) int {
	if result == nil {
		return 0
	}
	if ec, ok := result["eval_count"].(int); ok {
		return ec
	}
	if ec, ok := result["eval_count"].(float64); ok {
		return int(ec)
	}
	return 0
}

func printResult(result types.JobResult, duration time.Duration) {
	fmt.Println("\n==========================================================")
	if result.Success {
		fmt.Printf("SUCCESS in %s\n", duration.Round(time.Millisecond))
	} else {
		fmt.Printf("FAILED in %s\n", duration.Round(time.Millisecond))
		fmt.Printf("   Error: %s\n", result.Error)
	}

	if result.Result != nil {
		if resp, ok := result.Result["response"].(string); ok {
			fmt.Println("\n   Response:")
			for _, line := range wordWrap(strings.TrimSpace(resp), 70) {
				fmt.Printf("   %s\n", line)
			}
		}

		out, _ := json.MarshalIndent(result.Result, "   ", "  ")
		fmt.Printf("\n   Full result: %s\n", string(out))
	}

	fmt.Printf("\n   Duration: %dms\n", result.DurationMS)
	fmt.Printf("   Job ID:   %s\n", result.JobID)
	fmt.Printf("   GPU ID:   %s\n", result.GPUID)
	fmt.Println("==========================================================")
}

func wordWrap(text string, width int) []string {
	if len(text) == 0 {
		return nil
	}
	words := strings.Fields(text)
	var lines []string
	current := ""
	for _, word := range words {
		if current == "" {
			current = word
		} else if len(current)+1+len(word) <= width {
			current += " " + word
		} else {
			lines = append(lines, current)
			current = word
		}
	}
	if current != "" {
		lines = append(lines, current)
	}
	return lines
}
