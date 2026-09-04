package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/RunGPU-io/rungpu-agent/internal/types"
)

const (
	EarningOn           = "earning"
	EarningPaused       = "paused"
	EarningScheduledOff = "scheduled_pause"
	EarningStopped      = "stopped"
)

var weekdayNames = []string{"sun", "mon", "tue", "wed", "thu", "fri", "sat"}

type EarningStatus struct {
	Mode      string
	Open      bool
	Summary   string
	NextLabel string
}

func EvaluateEarning(cfg *types.Config, now time.Time) EarningStatus {
	if cfg == nil {
		return EarningStatus{Mode: EarningStopped, Summary: "Stopped"}
	}
	if cfg.Paused {
		return EarningStatus{
			Mode:    EarningPaused,
			Summary: "Paused — process stays running, pool is disconnected",
		}
	}
	sched := NormalizeSchedule(cfg.Schedule)
	if !sched.Enabled {
		return EarningStatus{Mode: EarningOn, Open: true, Summary: "Earning"}
	}
	localNow, loc := scheduleNow(sched, now)
	windows := ScheduleWindows(sched)
	if scheduleOpen(windows, localNow) {
		until := nextTransition(localNow, loc, windows, false)
		return EarningStatus{
			Mode:      EarningOn,
			Open:      true,
			Summary:   fmt.Sprintf("Earning until %s", until.Format("Mon 15:04 MST")),
			NextLabel: "Pauses at " + until.Format("Mon 15:04 MST"),
		}
	}
	until := nextTransition(localNow, loc, windows, true)
	return EarningStatus{
		Mode:      EarningScheduledOff,
		Summary:   fmt.Sprintf("Scheduled pause until %s", until.Format("Mon 15:04 MST")),
		NextLabel: "Resumes at " + until.Format("Mon 15:04 MST"),
	}
}

func ScheduleWindows(sched types.ScheduleConfig) []types.ScheduleWindow {
	if len(sched.Windows) > 0 {
		out := make([]types.ScheduleWindow, 0, len(sched.Windows))
		for _, win := range sched.Windows {
			out = append(out, normalizeWindow(win))
		}
		return out
	}
	return []types.ScheduleWindow{normalizeWindow(types.ScheduleWindow{
		StartHour: sched.StartHour,
		EndHour:   sched.EndHour,
	})}
}

func NormalizeSchedule(sched types.ScheduleConfig) types.ScheduleConfig {
	sched.Windows = ScheduleWindows(sched)
	if len(sched.Windows) > 0 {
		sched.StartHour = sched.Windows[0].StartHour
		sched.EndHour = sched.Windows[0].EndHour
	} else {
		sched.StartHour = clampHour(sched.StartHour)
		sched.EndHour = clampHour(sched.EndHour)
	}
	return sched
}

func scheduleNow(sched types.ScheduleConfig, now time.Time) (time.Time, *time.Location) {
	loc := time.Local
	if sched.Timezone != "" {
		if loaded, err := time.LoadLocation(sched.Timezone); err == nil {
			loc = loaded
		}
	}
	return now.In(loc), loc
}

func scheduleOpen(windows []types.ScheduleWindow, now time.Time) bool {
	for _, win := range windows {
		if windowActive(win, now) {
			return true
		}
	}
	return false
}

func windowActive(win types.ScheduleWindow, now time.Time) bool {
	start := clampHour(win.StartHour)
	end := clampHour(win.EndHour)
	hour := now.Hour()
	day := now.Weekday()
	if start == end {
		return dayMatches(win.Days, day)
	}
	if start < end {
		return hour >= start && hour < end && dayMatches(win.Days, day)
	}
	if hour >= start {
		return dayMatches(win.Days, day)
	}
	if hour < end {
		return dayMatches(win.Days, previousWeekday(day))
	}
	return false
}

func dayMatches(days []string, weekday time.Weekday) bool {
	wanted := normalizeDays(days)
	if len(wanted) == 0 {
		return true
	}
	name := weekdayNames[int(weekday)]
	for _, day := range wanted {
		if day == name {
			return true
		}
	}
	return false
}

func previousWeekday(day time.Weekday) time.Weekday {
	if day == time.Sunday {
		return time.Saturday
	}
	return day - 1
}

func nextTransition(now time.Time, loc *time.Location, windows []types.ScheduleWindow, wantOpen bool) time.Time {
	cursor := time.Date(now.Year(), now.Month(), now.Day(), now.Hour(), 0, 0, 0, loc)
	if !cursor.After(now) {
		cursor = cursor.Add(time.Hour)
	}
	for i := 0; i < 24*8; i++ {
		if scheduleOpen(windows, cursor) == wantOpen {
			return cursor
		}
		cursor = cursor.Add(time.Hour)
	}
	return cursor
}

func normalizeWindow(win types.ScheduleWindow) types.ScheduleWindow {
	win.StartHour = clampHour(win.StartHour)
	win.EndHour = clampHour(win.EndHour)
	win.Days = normalizeDays(win.Days)
	return win
}

func normalizeDays(days []string) []string {
	if len(days) == 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, raw := range days {
		name := parseWeekday(raw)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	if len(out) == 7 {
		return nil
	}
	return out
}

func parseWeekday(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "sun", "sunday", "0", "7":
		return "sun"
	case "mon", "monday", "1":
		return "mon"
	case "tue", "tues", "tuesday", "2":
		return "tue"
	case "wed", "wednesday", "3":
		return "wed"
	case "thu", "thur", "thurs", "thursday", "4":
		return "thu"
	case "fri", "friday", "5":
		return "fri"
	case "sat", "saturday", "6":
		return "sat"
	default:
		return ""
	}
}

func inScheduleWindow(hour, start, end int) bool {
	return windowActive(types.ScheduleWindow{StartHour: start, EndHour: end}, time.Date(2026, 1, 1, hour, 0, 0, 0, time.UTC))
}

func clampHour(hour int) int {
	if hour < 0 {
		return 0
	}
	if hour > 23 {
		return 23
	}
	return hour
}

func FormatWindows(windows []types.ScheduleWindow) string {
	if len(windows) == 0 {
		windows = ScheduleWindows(types.ScheduleConfig{})
	}
	var b strings.Builder
	for _, win := range windows {
		win = normalizeWindow(win)
		b.WriteString(fmt.Sprintf("%02d:00–%02d:00", win.StartHour, win.EndHour))
		if len(win.Days) == 0 {
			b.WriteString(" daily")
		} else {
			b.WriteString(" " + strings.Join(win.Days, ","))
		}
		b.WriteString("; ")
	}
	return strings.TrimSuffix(b.String(), "; ")
}

func ParseDaysFlag(raw string) ([]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ' '
	})
	days := normalizeDays(parts)
	if len(days) == 0 && len(parts) > 0 {
		return nil, fmt.Errorf("days must be weekday names such as mon,tue,sat")
	}
	return days, nil
}
