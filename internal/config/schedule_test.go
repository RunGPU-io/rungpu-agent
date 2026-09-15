package config

import (
	"testing"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

func TestInScheduleWindow(t *testing.T) {
	tests := []struct {
		hour, start, end int
		want             bool
	}{
		{hour: 10, start: 9, end: 17, want: true},
		{hour: 17, start: 9, end: 17, want: false},
		{hour: 8, start: 9, end: 17, want: false},
		{hour: 22, start: 21, end: 6, want: true},
		{hour: 5, start: 21, end: 6, want: true},
		{hour: 6, start: 21, end: 6, want: false},
		{hour: 12, start: 0, end: 0, want: true},
	}
	for _, test := range tests {
		if got := inScheduleWindow(test.hour, test.start, test.end); got != test.want {
			t.Fatalf("inScheduleWindow(%d, %d, %d) = %v, want %v", test.hour, test.start, test.end, got, test.want)
		}
	}
}

func TestEvaluateEarningModes(t *testing.T) {
	now := time.Date(2026, 9, 3, 20, 0, 0, 0, time.UTC)
	paused := EvaluateEarning(&types.Config{Paused: true, Schedule: types.ScheduleConfig{Enabled: true, StartHour: 18, EndHour: 23, Timezone: "UTC"}}, now)
	if paused.Open || paused.Mode != EarningPaused {
		t.Fatalf("paused = %+v", paused)
	}

	earning := EvaluateEarning(&types.Config{Schedule: types.ScheduleConfig{Enabled: true, StartHour: 18, EndHour: 23, Timezone: "UTC"}}, now)
	if !earning.Open || earning.Mode != EarningOn {
		t.Fatalf("earning = %+v", earning)
	}

	scheduled := EvaluateEarning(&types.Config{Schedule: types.ScheduleConfig{Enabled: true, StartHour: 18, EndHour: 23, Timezone: "UTC"}}, now.Add(-4*time.Hour))
	if scheduled.Open || scheduled.Mode != EarningScheduledOff {
		t.Fatalf("scheduled = %+v", scheduled)
	}

	always := EvaluateEarning(&types.Config{}, now)
	if !always.Open || always.Mode != EarningOn {
		t.Fatalf("always = %+v", always)
	}
}

func TestEvaluateEarningWeeklyAndMultipleWindows(t *testing.T) {
	thursday := time.Date(2026, 9, 3, 20, 0, 0, 0, time.UTC) // Thursday
	weekday := types.ScheduleConfig{
		Enabled:  true,
		Timezone: "UTC",
		Windows: []types.ScheduleWindow{{
			StartHour: 9,
			EndHour:   17,
			Days:      []string{"mon", "tue", "wed", "thu", "fri"},
		}},
	}
	if got := EvaluateEarning(&types.Config{Schedule: weekday}, thursday); got.Open {
		t.Fatalf("Thursday 20:00 should be closed: %+v", got)
	}
	if got := EvaluateEarning(&types.Config{Schedule: weekday}, time.Date(2026, 9, 3, 10, 0, 0, 0, time.UTC)); !got.Open {
		t.Fatalf("Thursday 10:00 should be open: %+v", got)
	}
	if got := EvaluateEarning(&types.Config{Schedule: weekday}, time.Date(2026, 9, 5, 10, 0, 0, 0, time.UTC)); got.Open {
		t.Fatalf("Saturday should be closed: %+v", got)
	}

	overnight := types.ScheduleConfig{
		Enabled:  true,
		Timezone: "UTC",
		Windows: []types.ScheduleWindow{{
			StartHour: 22,
			EndHour:   6,
			Days:      []string{"fri"},
		}},
	}
	if got := EvaluateEarning(&types.Config{Schedule: overnight}, time.Date(2026, 9, 4, 23, 0, 0, 0, time.UTC)); !got.Open {
		t.Fatalf("Friday 23:00 should be open: %+v", got)
	}
	if got := EvaluateEarning(&types.Config{Schedule: overnight}, time.Date(2026, 9, 5, 3, 0, 0, 0, time.UTC)); !got.Open {
		t.Fatalf("Saturday 03:00 should stay open for Friday overnight: %+v", got)
	}
	if got := EvaluateEarning(&types.Config{Schedule: overnight}, time.Date(2026, 9, 5, 7, 0, 0, 0, time.UTC)); got.Open {
		t.Fatalf("Saturday 07:00 should be closed: %+v", got)
	}

	either := types.ScheduleConfig{
		Enabled:  true,
		Timezone: "UTC",
		Windows: []types.ScheduleWindow{
			{StartHour: 9, EndHour: 17, Days: []string{"mon", "tue", "wed", "thu", "fri"}},
			{StartHour: 18, EndHour: 23},
		},
	}
	if got := EvaluateEarning(&types.Config{Schedule: either}, thursday); !got.Open {
		t.Fatalf("daily evening window should open Thursday 20:00: %+v", got)
	}
}
