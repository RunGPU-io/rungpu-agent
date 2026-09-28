package main

import (
	"fmt"

	"github.com/RunGPU-io/rungpu-agent/internal/config"
	"github.com/RunGPU-io/rungpu-agent/internal/job"
	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func prepareExecutionRecovery(cfg *types.Config, configPath string) error {
	id, err := job.PrepareJournalRoot(cfg.ModelCacheDir, cfg.ExecutionJournalID)
	if err != nil {
		return fmt.Errorf("execution recovery preflight: %w", err)
	}
	if cfg.ExecutionJournalID != id {
		previous := cfg.ExecutionJournalID
		cfg.ExecutionJournalID = id
		if err := config.Save(cfg, configPath); err != nil {
			cfg.ExecutionJournalID = previous
			return fmt.Errorf("persist execution journal identity before admission: %w", err)
		}
	}
	return nil
}
