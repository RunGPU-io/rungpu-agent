# Installs the Tokenize GPU Agent (Go) on Windows as a Scheduled Task.
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 [-ApiKey KEY]
param(
    [string]$ApiKey = "",
    [string]$InstallDir = "$env:ProgramFiles\Tokenize"
)

$ErrorActionPreference = "Stop"
$BinName = "tokenize-gpu-agent.exe"
$TaskName = "TokenizeGpuAgent"

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Error "Go 1.22+ not found. Install from https://go.dev/dl or drop a prebuilt binary into $InstallDir."
    exit 1
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Warning "nvidia-smi not found; agent will run CPU-only."
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Warning "docker not found; container jobs unavailable. GPU jobs need Docker Desktop + WSL2 backend."
}

Push-Location (Join-Path $PSScriptRoot "..")
Write-Host "Building agent..."
$env:CGO_ENABLED = "0"
go build -ldflags "-s -w" -o $BinName .

Write-Host "Installing to $InstallDir..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Force $BinName (Join-Path $InstallDir $BinName)
Pop-Location

$BinPath = Join-Path $InstallDir $BinName

if ($ApiKey -ne "") {
    & $BinPath init --api-key $ApiKey
}

Write-Host "Registering Scheduled Task '$TaskName'..."
$action  = New-ScheduledTaskAction -Execute $BinPath -Argument "start"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -DontStopOnIdleEnd
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Host ""
Write-Host "Installed. Manage with: Get-ScheduledTask -TaskName $TaskName"
if ($ApiKey -eq "") {
    Write-Host "Next: `"$BinPath`" init --api-key YOUR_API_KEY"
}
