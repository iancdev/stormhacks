param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(0, 127)]
    [int]$TakeoverButton,
    [ValidateRange(1, 15)]
    [double]$Angle = 5,
    [ValidateRange(0.01, 0.3)]
    [double]$TorqueLimit = 0.15
)

# Run in the wheel PC's existing Python environment. This script deliberately
# commands small physical movement; no game, model, or dataset is needed.
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Run this test on the Windows wheel PC." }
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskOldPythonPath = $env:PYTHONPATH
$taskOldLocation = Get-Location
try {
    Set-Location $taskRepo
    $env:PYTHONPATH = Join-Path $taskRepo "src"
    $taskRun = Join-Path $taskRepo ("runs/wheel-test-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
    New-Item -ItemType Directory -Path $taskRun -ErrorAction Stop | Out-Null
    python -m forza_ai.preflight --json --takeover-button $TakeoverButton |
        Tee-Object -FilePath (Join-Path $taskRun "preflight.json")
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed. Inspect $taskRun/preflight.json." }
    Write-Host "Stationary test: centre, right, centre, left, centre. Close Forza for this first run."
    Write-Host "Opening the wheel driver can engage its built-in centering. Keep the wheel clear."
    Write-Host "Press the selected takeover button to remove AI torque; Ctrl+C stops the run."
    python -m forza_ai.runtime --backend windows --assist --sweep --target-angle $Angle `
        --duration 12 --torque-limit $TorqueLimit --takeover-button $TakeoverButton `
        --interactive --status-csv (Join-Path $taskRun "control.csv")
    if ($LASTEXITCODE -ne 0) { throw "Wheel test failed; inspect diagnostics in $taskRun." }
    Write-Host "Test exited. Review actual movement and control.csv/control.json in $taskRun."
    Write-Host "An exit code alone does not verify the physical wheel's direction or tracking."
} finally {
    $env:PYTHONPATH = $taskOldPythonPath
    Set-Location $taskOldLocation
}
