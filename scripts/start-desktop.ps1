param(
    [Parameter(Mandatory=$true)]
    [string]$Profile,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskProfile = (Resolve-Path -LiteralPath $Profile -ErrorAction Stop).Path
$taskPython = Join-Path $taskRepo ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
$taskArgs = @("-m", "forza_ai.launch", "--profile", $taskProfile, "--role", "desktop")
if ($Check) { $taskArgs += "--check" }
# The shared key is inherited from this process's environment, never an argument.
& $taskPython @taskArgs
exit $LASTEXITCODE
