package pool

import "testing"

func TestShortJobRefDoesNotExposeFullCoordinatorID(t *testing.T) {
	const full = "job_1234567890_sensitive_suffix"
	if got := shortJobRef(full); got != "job_12345678" {
		t.Fatalf("shortJobRef() = %q", got)
	}
}
