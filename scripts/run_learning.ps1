param(
    [int]$Steps = 240000,
    [int]$Session = 1,
    [int]$Episodes = 1000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runnerPath = Join-Path $PSScriptRoot "run_learning_session.py"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python is missing at $pythonPath. Run the project setup first."
}

Push-Location $projectRoot
try {
    & $pythonPath $runnerPath --session $Session --steps $Steps --episodes $Episodes
    $runnerExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $runnerExitCode
