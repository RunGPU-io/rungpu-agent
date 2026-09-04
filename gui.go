package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/config"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

//go:embed gui.html
var guiHTML []byte

type guiController struct {
	mu       sync.Mutex
	logs     strings.Builder
	busy     string
	running  bool
	lastPing time.Time
}

func parseEnrollmentInput(raw string) string {
	value := strings.TrimSpace(raw)
	if value == "" {
		return ""
	}
	const flag = "--enrollment-token"
	idx := strings.Index(value, flag)
	if idx < 0 {
		return strings.Trim(value, `"'`)
	}
	rest := strings.TrimSpace(value[idx+len(flag):])
	rest = strings.TrimPrefix(rest, "=")
	rest = strings.TrimSpace(rest)
	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return ""
	}
	return strings.Trim(fields[0], `"'`)
}

func supportsDesktopGUI() bool {
	return runtime.GOOS == "windows" || runtime.GOOS == "darwin" || runtime.GOOS == "linux"
}

func runGUI() error {
	ctrl := &guiController{lastPing: time.Now()}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("open agent window: %w", err)
	}
	url := "http://" + listener.Addr().String()
	if launchedWithoutSharedConsole() {
		hideOwnConsole()
	} else {
		fmt.Printf("Opening the RunGPU Agent app at %s\n", url)
		fmt.Println("Command-line commands still work: init, setup, start, status, help.")
	}

	server := &http.Server{Handler: onlyLoopback(ctrl.handler())}
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if ctrl.shouldExit() {
				_ = server.Close()
				return
			}
		}
	}()
	if err := openGUIWindow(url); err != nil {
		fmt.Fprintf(os.Stderr, "open window: %v\nOpen this address in a browser: %s\n", err, url)
	}
	err = server.Serve(listener)
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

func onlyLoopback(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host, _, err := net.SplitHostPort(r.RemoteAddr)
		if err != nil {
			host = r.RemoteAddr
		}
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (c *guiController) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(guiHTML)
	})
	mux.HandleFunc("/api/state", func(w http.ResponseWriter, r *http.Request) {
		c.writeState(w)
	})
	mux.HandleFunc("/api/ping", func(w http.ResponseWriter, r *http.Request) {
		c.mu.Lock()
		c.lastPing = time.Now()
		c.mu.Unlock()
		c.writeState(w)
	})
	mux.HandleFunc("/api/enroll", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Token string `json:"token"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		token := parseEnrollmentInput(body.Token)
		if token == "" {
			c.writeError(w, http.StatusBadRequest, "paste the enrollment token from the host dashboard")
			return
		}
		c.runAction(w, "enroll", func() error {
			return cmdInit([]string{"--enrollment-token", token})
		})
	})
	mux.HandleFunc("/api/setup", func(w http.ResponseWriter, r *http.Request) {
		c.runAction(w, "setup", func() error { return cmdSetup(nil) })
	})
	mux.HandleFunc("/api/start", func(w http.ResponseWriter, r *http.Request) {
		c.mu.Lock()
		if c.running || c.busy != "" {
			c.mu.Unlock()
			c.writeState(w)
			return
		}
		c.busy = "start"
		c.mu.Unlock()
		c.appendLog("Starting the agent…\n")
		go c.startAgent()
		c.writeState(w)
	})
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		c.mu.Lock()
		running := c.running
		c.mu.Unlock()
		if running {
			c.writeState(w)
			return
		}
		c.runAction(w, "status", func() error { return cmdStatus(nil) })
	})
	mux.HandleFunc("/api/pause", func(w http.ResponseWriter, r *http.Request) {
		if err := cmdPause(nil); err != nil {
			c.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		c.appendLog("Paused — process stays running.\n")
		c.writeState(w)
	})
	mux.HandleFunc("/api/resume", func(w http.ResponseWriter, r *http.Request) {
		if err := cmdResume(nil); err != nil {
			c.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		c.appendLog("Resume requested.\n")
		c.writeState(w)
	})
	mux.HandleFunc("/api/unenroll", func(w http.ResponseWriter, r *http.Request) {
		if err := cmdUnenroll(nil); err != nil {
			c.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		c.appendLog("Unenrolled — config deleted. Paste a new token to enroll again.\n")
		c.writeState(w)
	})
	mux.HandleFunc("/api/stop", func(w http.ResponseWriter, r *http.Request) {
		if !requestAgentStop() {
			c.writeError(w, http.StatusBadRequest, "the agent is not running")
			return
		}
		c.appendLog("Stopping the agent process.\n")
		c.writeState(w)
	})
	mux.HandleFunc("/api/schedule", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Enabled   bool                   `json:"enabled"`
			Timezone  string                 `json:"timezone"`
			Windows   []types.ScheduleWindow `json:"windows"`
			StartHour int                    `json:"start_hour"`
			EndHour   int                    `json:"end_hour"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if err := updateSavedConfig(config.DefaultConfigPath(), func(cfg *types.Config) error {
			if !body.Enabled {
				cfg.Schedule.Enabled = false
				if strings.TrimSpace(body.Timezone) != "" {
					cfg.Schedule.Timezone = strings.TrimSpace(body.Timezone)
				}
				return nil
			}
			windows := body.Windows
			if len(windows) == 0 {
				windows = []types.ScheduleWindow{{StartHour: body.StartHour, EndHour: body.EndHour}}
			}
			cfg.Schedule = config.NormalizeSchedule(types.ScheduleConfig{
				Enabled:  true,
				Timezone: strings.TrimSpace(body.Timezone),
				Windows:  windows,
			})
			return nil
		}); err != nil {
			c.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		c.appendLog("Schedule updated.\n")
		c.writeState(w)
	})
	return mux
}

func (c *guiController) startAgent() {
	c.mu.Lock()
	c.running = true
	c.busy = ""
	c.mu.Unlock()
	err := c.withOutput(func() error { return cmdStart(nil) })
	c.mu.Lock()
	c.running = false
	c.busy = ""
	c.mu.Unlock()
	if err != nil {
		c.appendLog("error: " + err.Error() + "\n")
		return
	}
	c.appendLog("Agent stopped.\n")
}

func (c *guiController) runAction(w http.ResponseWriter, name string, fn func() error) {
	c.mu.Lock()
	if c.busy != "" || c.running {
		c.mu.Unlock()
		c.writeState(w)
		return
	}
	c.busy = name
	c.mu.Unlock()

	err := c.withOutput(fn)

	c.mu.Lock()
	c.busy = ""
	c.mu.Unlock()
	if err != nil {
		c.appendLog("error: " + err.Error() + "\n")
		c.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	c.writeState(w)
}

func (c *guiController) withOutput(fn func() error) error {
	reader, writer, err := os.Pipe()
	if err != nil {
		return fn()
	}
	oldOut, oldErr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = writer, writer
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(guiLogWriter{ctrl: c}, reader)
		close(done)
	}()
	runErr := fn()
	_ = writer.Close()
	<-done
	os.Stdout, os.Stderr = oldOut, oldErr
	return runErr
}

type guiLogWriter struct {
	ctrl *guiController
}

func (w guiLogWriter) Write(p []byte) (int, error) {
	w.ctrl.appendLog(string(p))
	return len(p), nil
}

func (c *guiController) appendLog(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.logs.WriteString(text)
	const max = 80_000
	if c.logs.Len() > max {
		trimmed := c.logs.String()[c.logs.Len()-max:]
		c.logs.Reset()
		c.logs.WriteString(trimmed)
	}
}

func (c *guiController) snapshot() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	machineID := ""
	enrolled := false
	earning := config.EvaluateEarning(nil, time.Now())
	scheduleEnabled := false
	startHour := 18
	endHour := 23
	timezone := ""
	windows := config.ScheduleWindows(types.ScheduleConfig{StartHour: 18, EndHour: 23})
	paused := false
	if cfg, err := config.Load(config.DefaultConfigPath()); err == nil && cfg.APIKey != "" && cfg.MachineID != "" {
		enrolled = true
		machineID = cfg.MachineID
		earning = config.EvaluateEarning(cfg, time.Now())
		paused = cfg.Paused
		normalized := config.NormalizeSchedule(cfg.Schedule)
		scheduleEnabled = cfg.Schedule.Enabled
		startHour = normalized.StartHour
		endHour = normalized.EndHour
		timezone = cfg.Schedule.Timezone
		windows = normalized.Windows
	}
	if !c.running {
		earning = config.EarningStatus{Mode: config.EarningStopped, Summary: "Ready — click Start to earn"}
	}
	return map[string]any{
		"version":           version,
		"enrolled":          enrolled,
		"machine_id":        machineID,
		"running":           c.running,
		"busy":              c.busy,
		"paused":            paused,
		"earning_mode":      earning.Mode,
		"earning":           earning.Summary,
		"earning_next":      earning.NextLabel,
		"schedule_enabled":  scheduleEnabled,
		"schedule_start":    startHour,
		"schedule_end":      endHour,
		"schedule_timezone": timezone,
		"schedule_windows":  windows,
		"logs":              c.logs.String(),
	}
}

func (c *guiController) writeState(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(c.snapshot())
}

func (c *guiController) writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func (c *guiController) shouldExit() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.running || c.busy != "" {
		return false
	}
	return time.Since(c.lastPing) > 20*time.Second
}
