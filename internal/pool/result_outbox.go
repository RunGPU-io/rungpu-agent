package pool

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const (
	resultRetryTick = time.Second
	resultRetryMin  = 2 * time.Second
	resultRetryMax  = 30 * time.Second
	resultBatchSize = 64
)

type pendingResult struct {
	key         string
	result      types.JobResult
	durable     bool
	nextAttempt time.Time
	retryDelay  time.Duration
}

func (c *Client) queueResult(result types.JobResult) {
	if !types.ValidJobID(result.JobID) {
		return
	}
	c.resultsMu.Lock()
	if c.results == nil {
		c.results = make(map[string]*pendingResult)
	}

	if pending := c.results[result.JobID]; pending != nil &&
		(pending.result.DispatchToken != "" || result.DispatchToken != "") {
		c.resultsMu.Unlock()
		return
	}
	if c.executionJournal != nil {
		if err := c.executionJournal.RememberResult(result); err != nil {
			log("result withheld because execution journal is fenced: %v", err)
			c.resultsMu.Unlock()
			return
		}
	}
	err := c.persistResult(result)
	if err != nil {
		log("could not persist result for job %s: %v", result.JobID, err)
	}
	c.results[result.JobID] = &pendingResult{key: result.JobID, result: result, durable: err == nil}
	c.resultsMu.Unlock()

	select {
	case c.resultsReady <- struct{}{}:
	default:
	}
}

func attemptResultKey(jobID, token string) string {
	return fmt.Sprintf("%s@%x", jobID, sha256.Sum256([]byte(token)))
}

func (c *Client) queueAttemptRejection(result types.JobResult) {
	if !types.ValidJobID(result.JobID) || result.DispatchToken == "" {
		return
	}
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	key := attemptResultKey(result.JobID, result.DispatchToken)
	if c.results == nil {
		c.results = make(map[string]*pendingResult)
	}
	if c.results[key] != nil {
		return
	}
	if c.executionJournal != nil {
		if err := c.executionJournal.RememberResult(result); err != nil {
			log("attempt rejection withheld because execution journal is fenced: %v", err)
			return
		}
	}
	err := c.persistResultAtKey(key, result)
	if err != nil {
		log("could not persist attempt rejection for job %s: %v", result.JobID, err)
	}
	c.results[key] = &pendingResult{key: key, result: result, durable: err == nil}
	select {
	case c.resultsReady <- struct{}{}:
	default:
	}
}

func (c *Client) ackResult(jobID, dispatchToken string) (accepted, canonical bool, token string) {
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	key := jobID
	pending := c.results[jobID]
	if dispatchToken != "" && c.results[attemptResultKey(jobID, dispatchToken)] != nil {
		key = attemptResultKey(jobID, dispatchToken)
		pending = c.results[key]
	}
	if pending == nil || (pending.result.DispatchToken != "" && pending.result.DispatchToken != dispatchToken) {
		return false, false, ""
	}
	if c.executionJournal != nil {
		if err := c.executionJournal.Accept(jobID, pending.result.DispatchToken); err != nil {
			log("result ACK withheld because execution journal is fenced: %v", err)
			return false, false, ""
		}
	}
	if err := os.Remove(c.resultPath(key)); err != nil && !os.IsNotExist(err) {
		log("could not remove acknowledged result for job %s: %v", jobID, err)
		return false, false, ""
	}
	delete(c.results, key)
	return true, key == jobID, pending.result.DispatchToken
}

func (c *Client) resultPath(jobID string) string {
	return filepath.Join(c.outboxDir, filepath.Base(jobID)+".json")
}

func (c *Client) persistResult(result types.JobResult) error {
	return c.persistResultAtKey(result.JobID, result)
}

func (c *Client) persistResultAtKey(key string, result types.JobResult) error {
	if !types.ValidJobID(result.JobID) ||
		(key != result.JobID && (result.DispatchToken == "" || key != attemptResultKey(result.JobID, result.DispatchToken))) {
		return fmt.Errorf("invalid result identity")
	}
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}
	path := c.resultPath(key)
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, data, 0o600); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func (c *Client) loadPendingResults() {
	c.resultsMu.Lock()
	defer c.resultsMu.Unlock()
	if c.results == nil {
		c.results = make(map[string]*pendingResult)
	}
	entries, err := os.ReadDir(c.outboxDir)
	if err != nil {
		log("could not read result outbox: %v", err)
	}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(c.outboxDir, entry.Name()))
		if err != nil {
			continue
		}
		var result types.JobResult
		if json.Unmarshal(data, &result) != nil || !types.ValidJobID(result.JobID) {
			continue
		}
		key := result.JobID
		if entry.Name() != key+".json" {
			key = attemptResultKey(result.JobID, result.DispatchToken)
			if result.DispatchToken == "" || entry.Name() != key+".json" {
				continue
			}
		}
		if _, exists := c.results[key]; !exists {
			c.results[key] = &pendingResult{key: key, result: result, durable: true}
		}
	}
	for _, pending := range c.results {
		pending.nextAttempt = time.Time{}
	}
}

func (c *Client) sendPendingResults(ctx context.Context, now time.Time, write func(interface{}) bool) bool {
	c.resultsMu.Lock()
	due := make([]*pendingResult, 0)
	for _, pending := range c.results {
		if c.executionJournal != nil {
			if record, exists := c.executionJournal.Find(pending.result.JobID, pending.result.DispatchToken); exists && record.ReplayBlocked {
				continue
			}
		}
		if !pending.nextAttempt.After(now) {
			due = append(due, pending)
		}
	}
	sort.Slice(due, func(i, j int) bool {
		if due[i].nextAttempt.Equal(due[j].nextAttempt) {
			return due[i].result.JobID < due[j].result.JobID
		}
		return due[i].nextAttempt.Before(due[j].nextAttempt)
	})
	if len(due) > resultBatchSize {
		due = due[:resultBatchSize]
	}
	c.resultsMu.Unlock()

	for _, pending := range due {
		if ctx.Err() != nil {
			return false
		}
		c.resultsMu.Lock()

		if c.results[pending.key] != pending {
			c.resultsMu.Unlock()
			continue
		}
		if !pending.durable {
			if err := c.persistResultAtKey(pending.key, pending.result); err != nil {
				log("could not persist result for job %s: %v", pending.result.JobID, err)
			} else {
				pending.durable = true
			}
		}
		if pending.retryDelay == 0 {
			pending.retryDelay = resultRetryMin
		} else {
			pending.retryDelay = min(pending.retryDelay*2, resultRetryMax)
		}
		pending.nextAttempt = now.Add(pending.retryDelay)
		c.resultsMu.Unlock()
		if !write(pending.result) {
			return false
		}
	}
	return true
}
