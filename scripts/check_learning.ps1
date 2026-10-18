param(
    [int]$Session = 1,
    [int]$Episodes = 100
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$auditPath = Join-Path $PSScriptRoot "audit_learning_session.py"
$sourcePath = Join-Path $projectRoot ("experiments\ironclad-a0-learning-sessions\session-{0:D4}" -f $Session)

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python is missing at $pythonPath. Run the project setup first."
}
if ($Session -lt 1 -or $Episodes -lt 1) {
    throw "Session and Episodes must be positive integers."
}

Push-Location $projectRoot
try {
    & $pythonPath $auditPath --source $sourcePath --episodes $Episodes
    $auditExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $auditExitCode
