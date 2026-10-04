param(
    [Parameter(Mandatory=$true)]
    [string]$Profile,
    [switch]$Check,
    [switch]$Assist
)

$ErrorActionPreference = "Stop"
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskProfile = (Resolve-Path -LiteralPath $Profile -ErrorAction Stop).Path
$taskPython = Join-Path $taskRepo ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
$taskArgs = @("-m", "forza_ai.launch", "--profile", $taskProfile, "--role", "game")
if ($Check) { $taskArgs += "--check" }
if ($Assist) { $taskArgs += "--assist" }
# Uses the installed project in this Python environment; no activation, policy,
# firewall, CUDA, or persistent environment changes are performed here.
& $taskPython @taskArgs
exit $LASTEXITCODE
