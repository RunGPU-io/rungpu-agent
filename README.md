# RunGPU Agent

[![License](https://img.shields.io/badge/license-Source%20Available-orange)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)]()

The GPU agent for the [RunGPU](https://www.rungpu.io) P2P GPU marketplace.
List your GPU, earn money when others rent it. Source available so you can audit every line running on your machine.

## Quick Start

1. Go to [rungpu.io/marketplace/host](https://www.rungpu.io/marketplace/host) and list your GPU
2. Create a one-time enrollment token for this machine
3. Download the agent for your platform from [Releases](https://github.com/RunGPU-io/rungpu-agent/releases)
4. Run:

```bash
./rungpu-agent init --enrollment-token YOUR_ONE_TIME_TOKEN
./rungpu-agent setup
./rungpu-agent version
./rungpu-agent start
```

`version` must match the release tag shown on GitHub. `setup` installs runtime
requirements; it does not rebuild or update the agent binary. To upgrade,
download and extract the newer GitHub release, then run its `setup` and `start`
commands with the existing `~/.tokenize` enrollment configuration.

That's it. Your GPU is now earning money.

## Install as a Service

Run in the background with auto-restart on boot:

```bash
# macOS
./scripts/install-macos.sh

# Linux
sudo ./scripts/install.sh

# Windows
.\scripts\install.ps1
```

## Commands

| Command | What it does |
|---|---|
| `rungpu-agent init --enrollment-token TOKEN` | Enroll this machine |
| `rungpu-agent setup` | Install supported runtime requirements |
| `rungpu-agent start` | Start earning |
| `rungpu-agent status` | Check GPU status |
| `rungpu-agent cleanup` | Remove agent data |
| `rungpu-agent version` | Print the embedded GitHub release version |

## Requirements

- **Docker** — [Install](https://docs.docker.com/get-docker/)
- **NVIDIA GPU** (recommended) — any GPU with 8+ GB VRAM
- **10 GB disk space**

Apple Silicon Macs are supported via Ollama.

## Troubleshooting

### Start with `status`

Run `rungpu-agent status` from the extracted agent folder. It checks whether
the machine is enrolled, detects the GPU, lists available capabilities, and
reports whether the required Docker or Ollama runtime is ready.

If it reports `Runtime: Setup required`, use this sequence:

```bash
rungpu-agent setup
rungpu-agent start
rungpu-agent status
```

On macOS and Linux, use the downloaded executable name with the `./` prefix.
Complete any installer or restart prompts before starting the agent again.

### Windows: virtualization or WSL 2 not detected

Docker Desktop requires hardware virtualization and WSL 2. Open **Task Manager
> Performance > CPU** and confirm **Virtualization** is enabled. If it is
disabled, enable Intel VT-x/VT-d or AMD-V/SVM in BIOS/UEFI.

Then open PowerShell as Administrator and run:

```powershell
wsl --install
wsl --update
wsl --set-default-version 2
```

Restart Windows, open Docker Desktop, and verify:

```powershell
wsl --status
docker version
docker run --rm hello-world
```

If Windows is itself running inside a VM, the VM host must enable nested
virtualization.

### Windows: `start` or `status` prints nothing

Run the executable directly from its extracted folder in PowerShell. Starting
the installed Scheduled Task does not open a console and is silent by design.

```powershell
Unblock-File .\rungpu-agent-windows-amd64.exe
.\rungpu-agent-windows-amd64.exe status *>&1 | Tee-Object .\rungpu-status.log
.\rungpu-agent-windows-amd64.exe start
```

The command now prints before checking NVIDIA, Docker, or Ollama, and each
external probe has a 10-second deadline. If an older release remains blank,
press Ctrl+C and test `nvidia-smi -L` and `docker version` separately; whichever
command hangs is the blocked driver/runtime. Start Docker Desktop and install
current NVIDIA drivers before retrying.

### Connected, but setup is incomplete

CUDA hosts require a running Docker engine. Apple Silicon hosts require Ollama.
Run `rungpu-agent setup`, start Docker Desktop if applicable, then restart the
agent and run `rungpu-agent status` again. A connected machine is shown in the
marketplace only after its declared hardware matches and its required runtime
is ready.

### Security software blocks the agent

Norton or another security product may flag a newly released, low-prevalence
binary. Download only from the official RunGPU GitHub release and verify the
published checksum. Review the source and the security product's details before
allowing the exact verified file. Do not disable endpoint protection to install
the agent.

### Enrollment is rejected

Enrollment commands are single-use and expire after seven days. Do not rerun a
consumed command. If enrollment already succeeded, continue with `setup` and
`start` using the saved credentials. Otherwise, generate a replacement command
from the host dashboard.

### Permission denied or command not found

- Extract the downloaded archive before running the agent.
- On macOS or Linux, run `chmod +x rungpu-agent-*` and keep the `./` prefix.
- On Windows, run the executable from its extracted folder in PowerShell.
- If `winget` is missing, install **App Installer** from the Microsoft Store.
- Confirm outbound HTTPS and WebSocket traffic is allowed by the firewall.

## Asset Retention

Downloaded safe model assets and workflows are cached for seven days after
their last use, with a default 20 GB cache budget. Reusing an asset refreshes
its age. When over budget, least-recently-used assets are removed first.
Automatic cleanup runs only while every GPU on the machine is idle and never
removes outputs, workspaces, configuration, or the durable result outbox.

Configure the policy in `~/.tokenize/config.yaml`:

```yaml
cleanup_interval_hours: 24
custom_asset_ttl_days: 7 # -1 disables age-based removal
max_custom_asset_cache_gb: 20 # -1 disables size-based LRU eviction
```

## Security

Every workload runs in a sandboxed container. You control what's allowed via the [web dashboard](https://www.rungpu.io/marketplace/host) or config file.

RunGPU's coordinator selects each workload runtime and its approved assets. The
agent does not infer runtimes from model names; incomplete assignments are
rejected locally.

## Uninstall

```bash
rungpu-agent cleanup --all
```

## License

Source Available — see [LICENSE](LICENSE). You may view, audit, and run this code to participate as a GPU host on RunGPU. You may not copy, redistribute, or use it to build competing products.

---

[RunGPU](https://www.rungpu.io) · [List Your GPU](https://www.rungpu.io/marketplace/host) · [Contact](https://www.rungpu.io/contact)
