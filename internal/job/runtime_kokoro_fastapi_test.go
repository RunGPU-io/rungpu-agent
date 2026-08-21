package job

import (
	"strings"
	"testing"
)

func TestIsKokoroFastAPIImage(t *testing.T) {
	allowed := []string{
		"ghcr.io/remsky/kokoro-fastapi-gpu",
		"ghcr.io/remsky/kokoro-fastapi-gpu:v0.8.0",
		"ghcr.io/remsky/kokoro-fastapi-gpu:latest-cu128",
		"ghcr.io/remsky/kokoro-fastapi-gpu@sha256:abc",
	}
	for _, img := range allowed {
		if !isKokoroFastAPIImage(img) {
			t.Errorf("expected FastAPI image %q", img)
		}
	}

	rejected := []string{
		"ghcr.io/tokenize/kokoro-tts:v1",
		"ghcr.io/remsky/kokoro-fastapi-cpu:latest",
		"ghcr.io/remsky/kokoro-fastapi-gpu-evil:latest",
		"ghcr.io/remsky/unrelated:latest",
		"",
	}
	for _, img := range rejected {
		if isKokoroFastAPIImage(img) {
			t.Errorf("did not expect FastAPI image %q", img)
		}
	}
}

func TestKokoroFastAPIClientUsesOpenAISpeechAPI(t *testing.T) {
	for _, needle := range []string{
		"http://127.0.0.1:8880",
		"/health",
		"/v1/audio/speech",
		`"response_format": "wav"`,
		`"model": "kokoro"`,
		"speech.wav",
	} {
		if !strings.Contains(kokoroFastAPIClient, needle) {
			t.Errorf("client missing %q", needle)
		}
	}
}
