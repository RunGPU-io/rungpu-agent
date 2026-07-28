# RunGPU Agent

[![License](https://img.shields.io/badge/license-Source%20Available-orange)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)]()

The GPU agent for the [RunGPU](https://www.rungpu.io) P2P GPU marketplace.
List your GPU, earn money when others rent it. Source available so you can audit every line running on your machine.

## Quick Start

1. Go to [rungpu.io/marketplace/host](https://www.rungpu.io/marketplace/host) and list your GPU
2. Copy your API key
3. Download the agent for your platform from [Releases](https://github.com/RunGPU-io/rungpu-agent/releases)
4. Run:

```bash
./rungpu-agent init --api-key YOUR_API_KEY
./rungpu-agent start
```

That's it. Your GPU is now earning money.

## Install as a Service

Run in the background with auto-restart on boot:

```bash
# macOS
./scripts/install-macos.sh YOUR_API_KEY

# Linux
sudo ./scripts/install.sh

# Windows
.\scripts\install.ps1 -ApiKey "YOUR_API_KEY"
```

## Commands

| Command | What it does |
|---|---|
| `rungpu-agent init --api-key KEY` | Set up the agent |
| `rungpu-agent start` | Start earning |
| `rungpu-agent status` | Check GPU status |
| `rungpu-agent cleanup` | Remove agent data |

## Requirements

- **Docker** — [Install](https://docs.docker.com/get-docker/)
- **NVIDIA GPU** (recommended) — any GPU with 8+ GB VRAM
- **10 GB disk space**

Apple Silicon Macs are supported via Ollama.

## Security

Every workload runs in a sandboxed container. You control what's allowed via the [web dashboard](https://www.rungpu.io/marketplace/host) or config file.

## Uninstall

```bash
rungpu-agent cleanup --all
```

## License

Source Available — see [LICENSE](LICENSE). You may view, audit, and run this code to participate as a GPU host on RunGPU. You may not copy, redistribute, or use it to build competing products.

---

[RunGPU](https://www.rungpu.io) · [List Your GPU](https://www.rungpu.io/marketplace/host) · [Contact](https://www.rungpu.io/contact)
