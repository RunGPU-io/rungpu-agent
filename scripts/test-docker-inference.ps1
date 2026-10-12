[CmdletBinding()]
param([switch]$Help)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if ($Help) {
    Write-Output "Usage: .\scripts\test-docker-inference.ps1"
    Write-Output "Requires this source checkout, Go 1.22+, and Docker Desktop using Linux containers."
    Write-Output "Checks real CPU model inference in Docker, not GPU/CUDA or the full agent."
    exit 0
}

$originalGoFlags = $env:GOFLAGS
$originalSmoke = $env:RUNGPU_RUNTIME_SMOKE
$logFile = $null
$locationPushed = $false
$exitCode = 1

try {
    $go = Get-Command go -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $docker = Get-Command docker -CommandType Application -ErrorAction Stop | Select-Object -First 1
    Push-Location (Join-Path $PSScriptRoot "..")
    $locationPushed = $true

    & $docker.Source version
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not reachable. Start Docker Desktop and ensure this user can access it."
    }
    $containerOS = & $docker.Source info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0 -or "$containerOS".Trim() -ne "linux") {
        throw "This check requires Linux containers. Switch Docker Desktop to Linux containers."
    }
    & $go.Source version
    if ($LASTEXITCODE -ne 0) {
        throw "Go could not run. Install Go 1.22 or newer and reopen PowerShell."
    }

    $logFile = Join-Path ([IO.Path]::GetTempPath()) ("rungpu-docker-inference-" + [guid]::NewGuid().ToString("N") + ".log")
    New-Item -Path $logFile -ItemType File -ErrorAction Stop | Out-Null
    Write-Host "Checking real CPU inference inside Docker with a small public model."
    Write-Host "May download Go dependencies, the pinned image, and a roughly 100 MB model."
    Write-Host "No enrollment, customer jobs, billing, or runtime installation is performed."
    Write-Host "Log: $logFile"

    $env:GOFLAGS = ""
    $env:RUNGPU_RUNTIME_SMOKE = "1"
    # Windows PowerShell 5.1 treats redirected native stderr as error records.
    # Keep diagnostics, but use the native exit code to decide success.
    $ErrorActionPreference = "Continue"
    & $go.Source test -v -count=1 -short=false `
        -run '^TestRuntimeSmoke_DockerInference$' ./internal/job -timeout 20m `
        2>&1 | Tee-Object -FilePath $logFile -ErrorAction Stop
    $testExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($testExitCode -ne 0) {
        throw "Docker inference test failed (exit code $testExitCode)."
    }
    if (-not (Select-String -Path $logFile -Pattern '^--- PASS: TestRuntimeSmoke_DockerInference \(' -Quiet)) {
        throw "The real Docker test did not report a pass; skipped or absent tests do not count."
    }

    Write-Host "PASS: a real model ran in Docker and produced the expected answer."
    Write-Host "This does not verify GPU/CUDA execution or the full RunGPU agent."
    $exitCode = 0
}
catch {
    Write-Host ("FAIL: " + $_.Exception.Message)
    if ($logFile) {
        Write-Host "Diagnostic log: $logFile"
    }
}
finally {
    $env:GOFLAGS = $originalGoFlags
    $env:RUNGPU_RUNTIME_SMOKE = $originalSmoke
    if ($locationPushed) {
        Pop-Location
    }
}

exit $exitCode
