package job

import (
	"context"
	"strings"
	"time"
)

const kokoroFastAPIImage = "ghcr.io/remsky/kokoro-fastapi-gpu"

// Client talks to Kokoro-FastAPI over localhost inside the job container.
// The image is a long-running OpenAI-compatible server, not an INPUT_DATA
// batch worker, so the agent drives /health and /v1/audio/speech then stops it.
const kokoroFastAPIClient = `
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

READY_TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 180
SPEECH_TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BASE = "http://127.0.0.1:8880"

def wait_ready(timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        for path in ("/health", "/v1/audio/voices"):
            try:
                urllib.request.urlopen(BASE + path, timeout=2)
                return
            except Exception as exc:
                last = exc
        time.sleep(1)
    raise SystemExit("Kokoro-FastAPI did not become ready: %s" % last)

def main():
    wait_ready(READY_TIMEOUT)
    payload = json.loads(os.environ.get("INPUT_DATA") or "{}")
    prompt = str(payload.get("prompt") or payload.get("text") or "").strip()
    if not prompt:
        raise SystemExit("prompt is required")
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    voice = str(generation.get("voice") or "af_heart")
    body = json.dumps({
        "model": "kokoro",
        "input": prompt,
        "voice": voice,
        "response_format": "wav",
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=SPEECH_TIMEOUT) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit("Kokoro-FastAPI speech failed: %s %s" % (exc.code, detail))
    if not audio:
        raise SystemExit("Kokoro-FastAPI returned empty audio")
    out = Path("/output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "speech.wav").write_bytes(audio)

if __name__ == "__main__":
    main()
`

func isKokoroFastAPIImage(image string) bool {
	trimmed := strings.TrimSpace(image)
	return trimmed == kokoroFastAPIImage ||
		strings.HasPrefix(trimmed, kokoroFastAPIImage+":") ||
		strings.HasPrefix(trimmed, kokoroFastAPIImage+"@")
}

func (r *customDockerRuntime) driveKokoroFastAPI(ctx context.Context, name, outputDir string, timeout time.Duration) error {
	return r.execPythonClient(ctx, name, outputDir, ".rungpu_kokoro_client.py", kokoroFastAPIClient, timeout)
}
