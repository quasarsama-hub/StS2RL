param(
    [int]$Session = 1,
    [int]$Steps = 240000,
    [int]$Episodes = 1000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trialPath = Join-Path $PSScriptRoot "choice_credit_trial.py"
$seriesPath = Join-Path $projectRoot "experiments\ironclad-a0-learning-sessions"
$sourcePath = Join-Path $seriesPath ("session-{0:D4}" -f $Session)
$outputPath = Join-Path $seriesPath "session-choice-features-trial"

if ($Session -lt 1 -or $Steps -lt 1 -or $Episodes -lt 1) {
    throw "Session, Steps and Episodes must be positive integers."
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python is missing at $pythonPath. Run the project setup first."
}
if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw "The completed source session is missing at $sourcePath."
}

Push-Location $projectRoot
try {
    & $pythonPath $trialPath --source $sourcePath --experiment $outputPath --steps $Steps --episodes $Episodes
    $trialExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $trialExitCode
