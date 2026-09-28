package job

import (
	"fmt"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func (j *AttemptJournal) AssignDockerDaemon(id, token, daemon string) (AttemptRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	record, exists := j.records[attemptKey(id, token)]
	if !exists || record.CancelRequested || record.Phase != "running" || daemon == "" {
		return record, fmt.Errorf("Docker execution cannot enter the assigned backend")
	}
	record.DockerDaemonID = daemon
	record.Resource = j.resource(record)
	return record, j.store(record)
}

func (j *AttemptJournal) recoverUnstarted(record AttemptRecord) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	current := j.records[attemptKey(record.JobID, record.DispatchToken)]
	if !current.KnownUnstarted && (!dockerRuntimeName(current.Runtime) || current.DockerDaemonID != "") {
		return fmt.Errorf("attempt has no durable pre-execution checkpoint")
	}
	current.KnownUnstarted, current.StopVerified, current.ReplayBlocked = true, true, false
	current.Phase = "finished"
	if current.Result == nil {
		current.Result = &types.JobResult{Type: "job_result", JobID: current.JobID, DispatchToken: current.DispatchToken,
			GPUID: current.GPUID, Error: "agent restarted before runtime entry; assignment was not executed"}
	}
	return j.store(current)
}
