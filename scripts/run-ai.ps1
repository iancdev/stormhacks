param(
    # Motor = the AI turns the TMX (the demo). Vjoy = fallback: AI steers Forza through vJoy, wheel stays still.
    [ValidateSet("Motor", "Vjoy")]
    [string]$Mode = "Motor",
    [string]$InferenceHost = "169.254.72.151",
    [ValidateRange(0, 127)]
    [int]$ArmButton = 10,
    [ValidateRange(0, 127)]
    [int]$TakeoverButton = 11,
    # Default: the Ethernet cable address, so only the laptop on the cable can open the dashboard.
    # 0.0.0.0 also exposes it (including its ARM button) on Wi-Fi; use only on a network you trust.
    [string]$DashboardHost = "169.254.218.1",
    [ValidateRange(1024, 65535)]
    [int]$DashboardPort = 8090,   # 8080 is taken on this PC (AgentService)
    [ValidateRange(10, 7200)]
    [double]$Duration = 900,
    [ValidateRange(0.01, 0.3)]
    [double]$TorqueLimit = 0.15
)

# Run on the game PC from anywhere: .\scripts\run-ai.ps1 [-Mode Vjoy] [-DashboardHost 0.0.0.0]
# Setup: Forza on the vJoy wheel layout, TMX hidden from Forza by HidHide, laptop inference server running.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) { throw "Missing $python; create the venv first." }

if (-not $env:FORZA_LINK_KEY) {
    # Hidden prompt; the key stays in this process only, never in files or command lines.
    $env:FORZA_LINK_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Shared LAN key' -AsSecureString)).Password
}

$runs = Join-Path $repo "runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$status = Join-Path $runs ("ai-" + $Mode.ToLower() + "-" + $stamp + ".csv")

$runtimeArgs = @("-m", "forza_ai.runtime", "--backend", "windows",
    "--inference-host", $InferenceHost, "--capture-config", (Join-Path $repo "config/capture.json"),
    "--assist", "--auto-pedals", "--arm-button", $ArmButton, "--takeover-button", $TakeoverButton,
    "--torque-limit", $TorqueLimit, "--duration", $Duration, "--status-csv", $status,
    "--dashboard-host", $DashboardHost, "--dashboard-port", $DashboardPort)
if ($Mode -eq "Vjoy") { $runtimeArgs += "--direct-vjoy" }

if ($DashboardHost -eq "0.0.0.0") {
    Write-Warning "Dashboard on ALL networks: anyone who can reach this PC can open it and press ARM."
}
Write-Host "Mode: $Mode | ARM button $ArmButton | takeover button $TakeoverButton (or touch a pedal)"
Write-Host "Dashboard: http://${DashboardHost}:$DashboardPort  (from the laptop: http://169.254.218.1:$DashboardPort)"
Write-Host "Log: $status"
Push-Location $repo
try {
    & $python @runtimeArgs
} finally {
    Pop-Location
}
