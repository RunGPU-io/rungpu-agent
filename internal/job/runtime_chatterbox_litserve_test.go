package job

import (
	"strings"
	"testing"
)

func TestIsChatterboxLitServeImage(t *testing.T) {
	allowed := []string{
		"bhimrazy/chatterbox-tts",
		"bhimrazy/chatterbox-tts:v0.1.0",
		"docker.io/bhimrazy/chatterbox-tts:v0.1.0",
		"docker.io/bhimrazy/chatterbox-tts@sha256:abc",
	}
	for _, img := range allowed {
		if !isChatterboxLitServeImage(img) {
			t.Errorf("expected Chatterbox image %q", img)
		}
	}
	rejected := []string{
		"ghcr.io/tokenize/chatterbox-tts:v1",
		"bhimrazy/unrelated:latest",
		"bhimrazy/chatterbox-tts-evil:latest",
		"ghcr.io/remsky/kokoro-fastapi-gpu:v0.8.0",
		"",
	}
	for _, img := range rejected {
		if isChatterboxLitServeImage(img) {
			t.Errorf("did not expect Chatterbox image %q", img)
		}
	}
}

func TestChatterboxClientUsesLitServeSpeechAPI(t *testing.T) {
	for _, needle := range []string{
		"http://127.0.0.1:8000",
		"/speech",
		`"text"`,
		"audio_prompt",
		"/custom/input",
		"speech.wav",
	} {
		if !strings.Contains(chatterboxLitServeClient, needle) {
			t.Errorf("client missing %q", needle)
		}
	}
}
