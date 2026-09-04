# RunGPU Agent

The app you run on your computer to earn money with a spare GPU on
[RunGPU](https://www.rungpu.io).

## Quick Start

1. List your GPU at [rungpu.io/marketplace/host](https://www.rungpu.io/marketplace/host)
2. Download the agent from [Releases](https://github.com/RunGPU-io/rungpu-agent/releases)
3. Enroll this computer, set it up, and start

**Command**

```bash
./rungpu-agent init --enrollment-token YOUR_TOKEN
./rungpu-agent setup
./rungpu-agent start
```

**GUI**

Open the window, paste the token, then click Enroll, Setup, and Start.

- Windows: double-click the `.exe`
- macOS: open `RunGPU Agent.app`
- Linux: `./rungpu-agent gui`

That's it. Your GPU can start earning.

## Keep it running

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
| `rungpu-agent` or `gui` | Open the window |
| `init --enrollment-token TOKEN` | Add this computer |
| `setup` | Install what the agent needs |
| `start` | Start earning |
| `pause` / `resume` | Stop or start earning |
| `status` | See if it is working |
| `cleanup` | Remove agent data |
| `version` | Show the version |

## You need

- A GPU with about 8 GB of memory or more
- [Docker](https://docs.docker.com/get-docker/) on Windows or Linux
- About 10 GB of free disk space

Apple Macs with Apple Silicon work too.

## Common problems

**Setup required.** Run `setup`, then `start`, then `status`.

**Windows Docker issues.** Turn on virtualization in BIOS, install WSL 2, restart, then open Docker Desktop.

**Mac says the app is blocked.** Right-click `RunGPU Agent.app` and choose Open.

**Antivirus warning.** Download only from this GitHub Releases page. Check the checksum. Do not turn off your antivirus.

**Enrollment failed.** Tokens work once and expire in 7 days. Get a new one from the host page.

**Permission denied.** Extract the download first. On Mac or Linux, run `chmod +x` and keep the `./` in front.

## Uninstall

```bash
rungpu-agent cleanup --all
```

## License

You may view, audit, and run this code to host a GPU on RunGPU. You may not copy it or use it to build a competing product. See [LICENSE](LICENSE).

---

[RunGPU](https://www.rungpu.io) · [List Your GPU](https://www.rungpu.io/marketplace/host) · [Contact](https://www.rungpu.io/contact)
