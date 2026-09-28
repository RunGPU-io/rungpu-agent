package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
	"github.com/gorilla/websocket"
)

const credential = "simulated-agent-local-only"

type simulator struct {
	mu                sync.Mutex
	conn              *websocket.Conn
	ready             bool
	gpuID             string
	active            map[string]chan struct{}
	seen              map[string]bool
	pending           map[string]types.JobResult
	attempts          map[string]string
	probeCancellation map[string]bool
}

func loopbackURL(raw string, schemes ...string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, err
	}
	allowed := false
	for _, scheme := range schemes {
		allowed = allowed || u.Scheme == scheme
	}
	ip := net.ParseIP(u.Hostname())
	if !allowed || ip == nil || !ip.IsLoopback() || u.User != nil || u.Port() == "" || u.Fragment != "" {
		return nil, fmt.Errorf("only literal loopback URLs with an explicit port are allowed")
	}
	return u, nil
}

func event(kind, jobID string) {
	b, _ := json.Marshal(map[string]string{"event": kind, "job_id": jobID})
	fmt.Println(string(b))
}

func (s *simulator) sendLocked(message interface{}) {
	if s.conn != nil && s.ready {
		_ = s.conn.SetWriteDeadline(time.Now().Add(2 * time.Second))
		if err := s.conn.WriteJSON(message); err != nil {
			_ = s.conn.Close()
		}
	}
}

func (s *simulator) finish(result types.JobResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, running := s.active[result.JobID]; !running {
		return
	}
	delete(s.active, result.JobID)
	s.pending[result.JobID] = result
	s.sendLocked(result)
	event("result", result.JobID)
}

func (s *simulator) execute(job types.JobAssignment, stop <-chan struct{}) {
	scenario, _ := job.Input["scenario"].(string)
	s.mu.Lock()
	if replayJobID, _ := job.Input["legacy_result_job"].(string); replayJobID != "" {
		s.sendLocked(types.JobResult{
			Type: "job_result", JobID: replayJobID, GPUID: s.gpuID, Success: true,
			Result: map[string]interface{}{"text": "Historical terminal replay"},
		})
	}
	if stale, _ := job.Input["stale_result"].(bool); stale {
		s.sendLocked(types.JobResult{
			Type: "job_result", JobID: job.JobID, GPUID: s.gpuID,
			DispatchToken: "stale-result-attempt", Success: true,
			Result: map[string]interface{}{"text": "Must never be accepted"},
		})
	}
	if stale, _ := job.Input["stale_progress"].(bool); stale {
		s.sendLocked(types.JobProgress{
			Type: "job_progress", JobID: job.JobID, GPUID: s.gpuID,
			DispatchToken: "stale-attempt", Stage: "wrong-attempt", Progress: 0.99,
			Message: "Must never be accepted",
		})
	}
	s.sendLocked(types.JobProgress{
		Type: "job_progress", JobID: job.JobID, GPUID: s.gpuID,
		DispatchToken: job.DispatchToken, Stage: "running", Progress: 0.5,
		Message: "Synthetic execution; no GPU runtime",
	})
	s.mu.Unlock()
	event("progress", job.JobID)
	if scenario == "hold" {
		<-stop
		return
	}
	select {
	case <-stop:
		return
	case <-time.After(700 * time.Millisecond):
	}
	result := types.JobResult{
		Type: "job_result", JobID: job.JobID, GPUID: s.gpuID, Success: true,
		DispatchToken: job.DispatchToken,
		DurationMS:    700, Result: map[string]interface{}{"text": "synthetic:" + job.ModelName},
	}
	if scenario == "fail" {
		result.Success, result.Error, result.Result = false, "Synthetic runtime failure", nil
	} else if modality, _ := job.Parameters["modality"].(string); modality == "image" || modality == "video" || modality == "audio" {
		result.Result = map[string]interface{}{"uploaded": false, "modality": modality}
		if scenario == "missing-upload" {
			result.Result["uploaded"] = true
		} else if scenario != "missing-media" {
			u, err := loopbackURL(job.UploadURL, "http")
			if err == nil {
				payload := []byte("synthetic-" + modality + ":" + job.JobID)
				req, requestErr := http.NewRequest(http.MethodPut, u.String(), bytes.NewReader(payload))
				err = requestErr
				if err == nil {
					req.Header.Set("Content-Type", map[string]string{
						"image": "image/png", "video": "video/mp4", "audio": "audio/wav",
					}[modality])
					client := &http.Client{
						Timeout:   2 * time.Second,
						Transport: &http.Transport{Proxy: nil},
						CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
							return fmt.Errorf("simulator refuses upload redirects")
						},
					}
					response, uploadErr := client.Do(req)
					err = uploadErr
					if response != nil {
						response.Body.Close()
						if response.StatusCode < 200 || response.StatusCode >= 300 {
							err = fmt.Errorf("upload status %d", response.StatusCode)
						}
					}
					client.CloseIdleConnections()
				}
			}
			if err != nil {
				result.Success, result.Error, result.Result = false, err.Error(), nil
			} else {
				result.Result["uploaded"] = true
			}
		}
	}
	s.finish(result)
}

func (s *simulator) receive(data []byte) error {
	var envelope types.Envelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return err
	}
	switch envelope.Type {
	case "gpu_register_ack":
		var ack struct{ Success bool }
		if err := json.Unmarshal(data, &ack); err != nil || !ack.Success {
			return fmt.Errorf("registration rejected")
		}
		s.mu.Lock()
		s.ready = true
		for _, result := range s.pending {
			s.sendLocked(result)
			event("replay", result.JobID)
		}
		s.mu.Unlock()
		event("registered", "")
	case "job_assignment":
		var job types.JobAssignment
		if err := json.Unmarshal(data, &job); err != nil {
			return err
		}
		if err := job.Validate(); err != nil {
			return err
		}
		if job.UploadURL != "" {
			if _, err := loopbackURL(job.UploadURL, "http"); err != nil {
				return err
			}
		}
		s.mu.Lock()
		if s.seen[job.JobID] {
			event("duplicate", job.JobID)
			s.mu.Unlock()
			return nil
		}
		s.seen[job.JobID] = true
		s.attempts[job.JobID] = job.DispatchToken
		s.probeCancellation[job.JobID], _ = job.Input["probe_cancel_acks"].(bool)
		stop := make(chan struct{})
		s.active[job.JobID] = stop
		s.mu.Unlock()
		event("started", job.JobID)
		go s.execute(job, stop)
	case "job_result_ack":
		var ack struct {
			JobID         string `json:"job_id"`
			DispatchToken string `json:"dispatch_token"`
			Success       bool   `json:"success"`
		}
		if err := json.Unmarshal(data, &ack); err != nil {
			return err
		}
		if ack.Success {
			s.mu.Lock()
			pending, exists := s.pending[ack.JobID]
			if exists && (pending.DispatchToken == "" || pending.DispatchToken == ack.DispatchToken) {
				delete(s.pending, ack.JobID)
				event("acked", ack.JobID)
			} else {
				event("ack-ignored", ack.JobID)
			}
			s.mu.Unlock()
		}
	case "job_cancel", "job_stop", "stop_job":
		var control types.JobControl
		if err := json.Unmarshal(data, &control); err != nil {
			return err
		}
		s.mu.Lock()
		token, known := s.attempts[control.JobID]
		if !known || token != control.DispatchToken {
			s.sendLocked(types.JobCancelAck{
				Type: "job_cancel_ack", JobID: control.JobID, DispatchToken: control.DispatchToken,
				Success: false, Error: "Cancellation does not match a known attempt",
			})
			s.mu.Unlock()
			return nil
		}
		if s.probeCancellation[control.JobID] {
			s.sendLocked(types.JobCancelAck{
				Type: "job_cancel_ack", JobID: control.JobID,
				DispatchToken: "00000000-0000-4000-8000-000000000098", Success: true,
			})
			s.sendLocked(types.JobCancelAck{Type: "job_cancel_ack", JobID: control.JobID, Success: true})
			event("invalid-stop-acks", control.JobID)
		}
		if stop, ok := s.active[control.JobID]; ok {
			close(stop)
			delete(s.active, control.JobID)
		}
		s.mu.Unlock()
		event("stopped", control.JobID)
		time.Sleep(250 * time.Millisecond)
		s.mu.Lock()
		s.sendLocked(types.JobCancelAck{
			Type: "job_cancel_ack", JobID: control.JobID, DispatchToken: control.DispatchToken, Success: true,
		})
		s.mu.Unlock()
		event("stop-acked", control.JobID)
	}
	return nil
}

func main() {
	endpoint := flag.String("coordinator", "", "literal loopback ws:// endpoint")
	gpuID := flag.String("gpu-id", "simulated-gpu-1", "synthetic GPU identity")
	flag.Parse()
	u, err := loopbackURL(*endpoint, "ws")
	if err != nil {
		log.Fatal(err)
	}
	query := u.Query()
	query.Set("gpu_id", *gpuID)
	u.RawQuery = query.Encode()
	s := &simulator{
		gpuID: *gpuID, active: map[string]chan struct{}{}, seen: map[string]bool{},
		pending: map[string]types.JobResult{}, attempts: map[string]string{}, probeCancellation: map[string]bool{},
	}
	dialer := websocket.Dialer{HandshakeTimeout: 2 * time.Second}
	for {
		conn, _, err := dialer.Dial(u.String(), http.Header{"Authorization": {"Bearer " + credential}})
		if err != nil {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		s.mu.Lock()
		s.conn = conn
		s.ready = false
		_ = conn.SetWriteDeadline(time.Now().Add(2 * time.Second))
		_ = conn.WriteJSON(types.RegisterMessage{
			Type: "gpu_register", GPUID: *gpuID, MachineID: "machine-" + *gpuID,
			DetectedDeviceCount: 1, GPUType: "SIMULATED-NO-GPU", Backend: "cuda",
			VRAMGB: 32, PricePerMinute: 0.06, ModelsCached: []string{},
			Capabilities: []string{"ollama", "docker"}, DriverVersion: "synthetic",
		})
		s.mu.Unlock()
		for {
			_, data, readErr := conn.ReadMessage()
			if readErr != nil {
				break
			}
			if err := s.receive(data); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
		s.mu.Lock()
		s.conn = nil
		s.ready = false
		s.mu.Unlock()
		conn.Close()
		time.Sleep(100 * time.Millisecond)
	}
}
