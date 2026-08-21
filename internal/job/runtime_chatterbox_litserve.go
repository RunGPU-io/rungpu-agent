package job

import (
	"context"
	"strings"
	"time"
)

const chatterboxLitServeImage = "docker.io/bhimrazy/chatterbox-tts"
const chatterboxLitServeShortImage = "bhimrazy/chatterbox-tts"

// Client talks to bhimrazy/chatterbox-tts (LitServe) over localhost inside the
// job container. POST /speech; optional /custom/input/reference.{wav,mp3} for cloning.
const chatterboxLitServeClient = `
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

READY_TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 180
SPEECH_TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BASE = "http://127.0.0.1:8000"

def wait_ready(timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        for path in ("/health", "/docs", "/speech"):
            try:
                urllib.request.urlopen(BASE + path, timeout=2)
                return
            except urllib.error.HTTPError as exc:
                if exc.code in (200, 400, 405, 415, 422):
                    return
                last = exc
            except Exception as exc:
                last = exc
        time.sleep(1)
    raise SystemExit("Chatterbox TTS did not become ready: %s" % last)

def main():
    wait_ready(READY_TIMEOUT)
    payload = json.loads(os.environ.get("INPUT_DATA") or "{}")
    prompt = str(payload.get("prompt") or payload.get("text") or "").strip()
    if not prompt:
        raise SystemExit("prompt is required")
    if len(prompt) > 500:
        raise SystemExit("Chatterbox prompts must be 500 characters or fewer")
    body = {
        "text": prompt,
        "exaggeration": 0.5,
        "cfg": 0.5,
        "temperature": 0.8,
    }
    for name in ("reference.wav", "reference.mp3"):
        ref = Path("/custom/input") / name
        if ref.is_file():
            body["audio_prompt"] = str(ref)
            break
    req = urllib.request.Request(
        BASE + "/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=SPEECH_TIMEOUT) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit("Chatterbox speech failed: %s %s" % (exc.code, detail))
    if not audio:
        raise SystemExit("Chatterbox returned empty audio")
    out = Path("/output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "speech.wav").write_bytes(audio)

if __name__ == "__main__":
    main()
`

func isChatterboxLitServeImage(image string) bool {
	trimmed := strings.TrimSpace(image)
	return imageNameOrTag(trimmed, chatterboxLitServeImage) ||
		imageNameOrTag(trimmed, chatterboxLitServeShortImage)
}

func imageNameOrTag(image, name string) bool {
	return image == name || strings.HasPrefix(image, name+":") || strings.HasPrefix(image, name+"@")
}

func (r *customDockerRuntime) driveChatterboxLitServe(ctx context.Context, name, outputDir string, timeout time.Duration) error {
	return r.execPythonClient(ctx, name, outputDir, ".rungpu_chatterbox_client.py", chatterboxLitServeClient, timeout)
}
