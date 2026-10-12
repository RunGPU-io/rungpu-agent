# Contributing

Found a bug? Want to improve something? Pull requests are welcome.

1. Fork the repo
2. Create a branch
3. Make your change
4. Run `go test ./...`
5. Submit a PR

## Runtime and release verification

`go test -short ./...` runs unit and regression tests without real inference.
`node --test gui_browser_test.mjs` runs the GUI browser regressions using only
Node's built-in modules; shared CI and release verification run this on Node 22.

### Docker-only local check

#### Purpose and scope

**Question this test answers:** can the selected Docker image load a real model,
execute a prompt, and generate the expected answer on this machine's Docker
engine?

This is a regression check for the observed llama.cpp startup failure:
`no backends are loaded`, followed by model-load/server-startup failure.
Compiling the agent or running mocked tests cannot establish that this runtime
path works. The check exercises the pinned image with the corrected
`/app/llama-completion` entrypoint, not the broken interactive CLI path.

| Stage | What is checked | What failure means |
|---|---|---|
| Prerequisites | Go and a reachable Docker engine; the PowerShell runner also checks Linux-container mode | The test cannot start; this is not an inference result |
| Image and model download | Pull the pinned public CPU image and download the small public GGUF model | Network, registry access, storage, or download failure; inference is not verified |
| Container/model startup | Start the image, mount the model read-only, and load it through llama.cpp's completion entrypoint | Container/backend/model loading is not working for this check |
| Real generation | Ask `What is 2+2? Answer with only the number.` and generate up to eight tokens at temperature zero | The process must finish successfully and produce an answer |
| Acceptance | The named test passes, and generated stdout contains the standalone token `4` | Nonzero exit, wrong/empty output, `4` only in stderr logs, skipped test, or missing test is a failure |

**A PASS establishes only this CPU Docker inference path.** It does not prove:

- NVIDIA/CUDA acceleration, GPU discovery, VRAM capacity, or performance;
- other models, image/audio/video generation, or custom workloads;
- GUI setup/start/pause behavior, enrollment, or coordinator connectivity;
- the complete agent assignment/upload/retry/recovery lifecycle;
- customer billing, host earnings, or overall release/production readiness.

The model is deliberately small so the check can run without a GPU. A Windows
PASS means this CPU workload ran through Docker Desktop's Linux-container
engine; it does not mean a native Windows model process or CUDA was tested.

#### Run on Windows

Copy the **entire updated Go-agent source folder**, not just the script or an
older released executable, to the Windows machine. Install Go 1.22+ and Docker
Desktop, start Docker Desktop, and select **Linux containers**. Docker Desktop
must have its supported virtualization/WSL 2 backend configured, but this
PowerShell runner does not require opening a WSL shell.

In PowerShell, from that source folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\test-docker-inference.ps1
$LASTEXITCODE
```

PowerShell 7 can use `pwsh` instead of `powershell.exe`. The execution-policy
argument applies only to that child process; do not change the machine's
policy or bypass organization policy. If a downloaded script is blocked,
review its contents and follow your organization's procedure for trusted
scripts.

#### Run on Linux, macOS, or WSL

Copy this updated source checkout to the test machine, install Go 1.22 or newer,
and start Docker Desktop or Docker Engine in Linux-container mode. On Linux,
macOS, or Windows through WSL with Docker integration enabled, run:

```sh
bash scripts/test-docker-inference.sh
```

Both runners work from any current directory when invoked by their full path. Each
reuses the real Docker smoke test below; no Ollama, GPU, RunGPU credentials, or
enrollment is required. It downloads Go dependencies if missing, the pinned CPU
image, and a roughly 100 MB public model, then asks the model to answer `2+2`.
Allow internet access to Go module sources, GHCR, and Hugging Face, and keep
several hundred MB of disk space available. Docker Desktop must be able to
bind-mount the current user's home directory, where the temporary model lives.

Exit code 0 and the final `PASS` mean real inference produced the expected
answer. Missing Docker, failed downloads/inference, skipped tests, and absent
tests all fail. After prerequisite checks pass, a diagnostic log path is printed
and the log is kept for troubleshooting (`%TEMP%` for PowerShell). The test
removes its container and temporary model on normal completion/failure; the
pulled Docker image and Go dependency cache remain.
It does not install runtimes, enroll a host, contact customer jobs, or bill
anyone. This CPU check does **not** certify CUDA/GPU execution, the whole agent
job lifecycle, or the desktop GUI.

When reporting a result, include the source commit (or state that it is an
uncommitted source copy), Windows/Docker/Go versions, the final PASS/FAIL and exit
code, and the printed diagnostic log. A failure before generation is
**unverified inference**, not proof that the model answered incorrectly.
Review logs for machine-specific details before sharing them.

### Native and Docker checks together

On a machine with Docker running and Ollama installed, run:

```sh
RUNGPU_RUNTIME_SMOKE=1 go test -v -count=1 -run '^TestRuntimeSmoke_' ./internal/job -timeout 20m
```

This opt-in check downloads small public models and performs native Ollama and
CPU Docker inference. Docker uses the digest-pinned llama.cpp `light` image in
`internal/job/runtime_smoke_test.go`, with `/app/llama-completion` explicitly
selected. The interactive `llama-cli` entrypoint in affected upstream builds
starts its embedded server without initializing dynamic backends, producing
`no backends are loaded`. Completion initializes the backends before loading
the model. Only generated stdout, not version/backend logs on stderr, satisfies
the Docker answer check. When updating the image digest, run real Docker
inference; unit tests and image metadata inspection are not substitutes.

Tag releases reuse the entire CI workflow at the tagged commit's SHA.
Publishing requires both the release builds and CI to succeed, including the
clean Linux service-user install, Linux native/Docker inference, and macOS native
inference. A successful build alone or an unrelated green `main` run cannot
authorize publication. macOS release binaries use Go 1.24, matching the macOS
runtime job's toolchain and its macOS 26 `LC_UUID` requirement. Linux ARM64,
Intel macOS, and Windows receive build checks, not native runtime smoke coverage.

We won't accept changes that weaken security or bypass platform billing.

By submitting a PR, you agree your contribution is licensed under the same [Source Available license](LICENSE).

Report security issues privately to support@rungpu.io.
