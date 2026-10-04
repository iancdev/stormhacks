param(
    # Mirror = Forza follows the AI through vJoy AND the motor turns the TMX to mirror it (recommended demo:
    #          the car never depends on how well the motor tracks; grab the wheel to take over).
    # Motor  = the AI turns the TMX and Forza reads the wheel's measured angle (needs a well-tuned motor).
    # Vjoy   = fallback: AI steers Forza through vJoy, wheel stays still.
    [ValidateSet("Mirror", "Motor", "Vjoy")]
    [string]$Mode = "Mirror",
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
    # Motor gains, measured on this TMX: it needs ~0.2 torque to break free, then moves very freely,
    # so: modest pull (Kp), strong braking (Kd), a breakaway push (Friction), low cap (TorqueLimit).
    [ValidateRange(0.01, 0.3)]
    [double]$TorqueLimit = 0.22,
    [double]$Kp = 0.01,
    [double]$Kd = 0.005,
    [double]$Friction = 0.15,
    # Steering booster: the model steers about half as much as a human in corners (shadow test).
    [ValidateRange(0.5, 3.0)]
    [double]$SteerGain = 1.6,
    # How fast the steering target may change (deg/s); the runtime default of 60 lags quick corners.
    [ValidateRange(30, 400)]
    [double]$TargetRate = 150,
    # AI throttle damping (it was spinning the wheels): cap, max rise per second (drops instantly),
    # optional speed limit (0 = off). Ignored with -HumanPedals.
    [ValidateRange(0.05, 1.0)]
    [double]$ThrottleCap = 0.7,
    [ValidateRange(0, 10)]
    [double]$ThrottleRate = 0.5,
    [ValidateRange(0, 400)]
    [double]$MaxSpeedKmh = 0,
    # AI steers only; you drive the gas and brake (pressing them does not take over).
    [switch]$HumanPedals
)

# Run on the game PC from anywhere: .\scripts\run-ai.ps1 [-Mode Mirror|Motor|Vjoy] [-DashboardHost 0.0.0.0]
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
    "--assist", "--arm-button", $ArmButton, "--takeover-button", $TakeoverButton,
    "--torque-limit", $TorqueLimit, "--duration", $Duration, "--status-csv", $status,
    "--steer-gain", $SteerGain, "--target-rate", $TargetRate,
    "--dashboard-host", $DashboardHost, "--dashboard-port", $DashboardPort)
if ($HumanPedals) {
    $runtimeArgs += "--human-pedals"
} else {
    $runtimeArgs += @("--auto-pedals", "--throttle-cap", $ThrottleCap, "--throttle-rate", $ThrottleRate,
                      "--max-speed-kmh", $MaxSpeedKmh)
}
if ($Mode -eq "Vjoy") { $runtimeArgs += "--direct-vjoy" }
if ($Mode -eq "Mirror") { $runtimeArgs += @("--direct-vjoy", "--mirror-wheel") }
# --motor-update-ms/--torque-step: the TMX queues commands sent every 10 ms and falls further behind.
if ($Mode -ne "Vjoy") { $runtimeArgs += @("--kp", $Kp, "--kd", $Kd, "--friction", $Friction,
                                          "--motor-update-ms", 30, "--torque-step", 0.02) }

if ($DashboardHost -eq "0.0.0.0") {
    Write-Warning "Dashboard on ALL networks: anyone who can reach this PC can open it and press ARM."
}
$pedalHow = if ($HumanPedals) { "" } else { "or touch a pedal, " }
$how = if ($Mode -eq "Mirror") { "${pedalHow}or hold the wheel away from the AI" } else { "${pedalHow}".TrimEnd(", ".ToCharArray()) }
Write-Host "Mode: $Mode | ARM button $ArmButton | takeover button $TakeoverButton ($how)"
Write-Host "Steering gain x$SteerGain, target rate $TargetRate deg/s"
if ($HumanPedals) { Write-Host "Pedals: YOU drive gas and brake; the AI steers only" }
else { Write-Host "AI pedals: throttle cap $ThrottleCap, ramp $ThrottleRate/s, speed cap $(if ($MaxSpeedKmh) { "$MaxSpeedKmh km/h" } else { 'off' })" }
Write-Host "Dashboard: http://${DashboardHost}:$DashboardPort  (from the laptop: http://169.254.218.1:$DashboardPort)"
Write-Host "Log: $status"
Push-Location $repo
try {
    & $python @runtimeArgs
} finally {
    Pop-Location
}
