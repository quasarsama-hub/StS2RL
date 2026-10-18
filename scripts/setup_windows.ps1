[CmdletBinding()]
param(
    [string]$GameDirectory = $env:STS2_GAME_DIR,
    [string]$PythonExecutable = $env:SPIRE2_PYTHON,
    [string]$GitExecutable = $env:SPIRE2_GIT,
    [switch]$BuildNative
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ExternalRoot = Join-Path $ProjectRoot '.external'
$EmulatorRoot = Join-Path $ExternalRoot 'sts2-rl-agent'
$NativeRoot = Join-Path $ExternalRoot 'sts2-cli'
$EmulatorCommit = '04a62c35b0ee264f5e4a3b5c6587f7438f1a7e0c'
$NativeCommit = '084d1aa3d8e118ca7ce8d8774ad16d6be9c92367'
$TorchVersion = '2.8.0'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    # Stream subprocess diagnostics to the console without contaminating function return values.
    & $Executable @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw ('Command failed with exit code ' + $LASTEXITCODE + ': ' + $Executable + ' ' + ($Arguments -join ' ')) }
}

function Get-PinnedRepository {
    param([string]$Url, [string]$Destination, [string]$Commit)
    if (-not (Test-Path -LiteralPath $Destination)) {
        Invoke-Checked $GitPath @('clone', '--no-checkout', '--filter=blob:none', $Url, $Destination)
        Invoke-Checked $GitPath @('-C', $Destination, 'checkout', '--detach', $Commit)
    } else {
        if (-not (Test-Path -LiteralPath (Join-Path $Destination '.git'))) { throw "Existing dependency folder is not a Git checkout: $Destination" }
        $CurrentCommit = (& $GitPath -C $Destination rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Cannot read dependency revision: $Destination" }
        if ($CurrentCommit -ne $Commit) { throw "Dependency revision mismatch at $Destination. Expected $Commit; found $CurrentCommit. Preserve your changes and move that folder before rerunning." }
        $Changes = & $GitPath -C $Destination status --porcelain --untracked-files=no
        if ($LASTEXITCODE -ne 0) { throw "Cannot inspect dependency checkout: $Destination" }
        # Native localization is refreshed from the game; all other tracked edits need review.
        $Unexpected = @($Changes | Where-Object { $_ -notmatch '^.. localization_(eng|zhs)/' })
        if ($Unexpected.Count -gt 0) { throw "Dependency has tracked local edits at $Destination. Preserve them before rerunning setup." }
    }
}

function Build-NativeRunner {
    param([hashtable]$Diagnostics, [string]$PythonExecutable)
    if (-not $Diagnostics.has_dotnet_9_sdk) { throw 'Native build needs the .NET 9+ SDK. Install it from https://dotnet.microsoft.com/download and rerun with -BuildNative.' }
    if (-not $Diagnostics.game) { throw 'Native build cannot locate sts2.dll. Rerun with -GameDirectory "C:\path\to\Slay the Spire 2".' }
    $Game = $Diagnostics.game
    $LibRoot = Join-Path $NativeRoot 'lib'
    New-Item -ItemType Directory -Force -Path $LibRoot | Out-Null
    $DllNames = @('sts2.dll', 'SmartFormat.dll', 'SmartFormat.ZString.dll', 'Sentry.dll', 'Sentry.Godot.dll', 'Steamworks.NET.dll', 'MonoMod.Backports.dll', 'MonoMod.ILHelpers.dll', '0Harmony.dll', 'System.IO.Hashing.dll')
    foreach ($Name in $DllNames) {
        $Source = Join-Path $Game.data_directory $Name
        if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
            $Found = Get-ChildItem -LiteralPath $Game.install_directory -Filter $Name -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($Found) { $Source = $Found.FullName }
            elseif ($Name -in @('Sentry.Godot.dll', 'System.IO.Hashing.dll')) {
                Write-Warning "Optional game dependency $Name not found; continuing as upstream setup does."
                continue
            }
            else { throw "Game dependency $Name not found. Native setup is incomplete; no game files were modified." }
        }
        Copy-Item -LiteralPath $Source -Destination (Join-Path $LibRoot $Name) -Force
    }
    $CopiedDll = Join-Path $LibRoot 'sts2.dll'
    Copy-Item -LiteralPath $CopiedDll -Destination ($CopiedDll + '.original') -Force
    $Pck = Get-ChildItem -LiteralPath $Game.install_directory -Filter '*.pck' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Pck) { Invoke-Checked $PythonExecutable @((Join-Path $NativeRoot 'scripts\extract_localization.py'), $Pck.FullName) }
    else { Write-Warning 'No game PCK found. Native runner uses the localization bundled in its pinned checkout.' }

    # Use the exact patcher source in pinned upstream setup.sh; Git Bash is unnecessary.
    $SetupText = Get-Content -LiteralPath (Join-Path $NativeRoot 'setup.sh') -Raw
    $ProjectMatch = [regex]::Match($SetupText, '(?s)<< ''PROJ''\r?\n(.*?)\r?\nPROJ')
    $SourceMatch = [regex]::Match($SetupText, '(?s)<< ''CSHARP''\r?\n(.*?)\r?\nCSHARP')
    if (-not $ProjectMatch.Success -or -not $SourceMatch.Success) { throw 'Pinned native setup patcher blocks were not found. Refusing to build an unpatched runner.' }
    $PatchRoot = Join-Path $ExternalRoot 'native-patcher'
    New-Item -ItemType Directory -Force -Path $PatchRoot | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $PatchRoot 'Patcher.csproj'), $ProjectMatch.Groups[1].Value)
    [System.IO.File]::WriteAllText((Join-Path $PatchRoot 'Program.cs'), $SourceMatch.Groups[1].Value)
    $PreviousRollForward = $env:DOTNET_ROLL_FORWARD
    try {
        $env:DOTNET_ROLL_FORWARD = 'LatestMajor'
        Invoke-Checked $Diagnostics.dotnet @('run', '--project', (Join-Path $PatchRoot 'Patcher.csproj'), '--', $CopiedDll)
    } finally { $env:DOTNET_ROLL_FORWARD = $PreviousRollForward }
    Invoke-Checked $Diagnostics.dotnet @('build', (Join-Path $NativeRoot 'src\Sts2Headless\Sts2Headless.csproj'), '--configuration', 'Debug')
    $RunnerDll = Join-Path $NativeRoot 'src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.dll'
    if (-not (Test-Path -LiteralPath $RunnerDll -PathType Leaf)) { throw "Native build did not produce expected runner: $RunnerDll" }
    $GameHashAfter = (Get-FileHash -LiteralPath $Game.dll -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($GameHashAfter -ne $Game.dll_sha256) { throw 'Installed game DLL changed during setup. Stop and inspect the installation before using the runner.' }
    return @{
        native_status = 'built_not_probed'
        native_command = @($Diagnostics.dotnet, $RunnerDll)
        native_cwd = $NativeRoot
        native_original_dll_sha256 = $Game.dll_sha256
        native_patched_dll_sha256 = (Get-FileHash -LiteralPath $CopiedDll -Algorithm SHA256).Hash.ToLowerInvariant()
        game_hashes = @{
            installed_sts2_sha256 = $Game.dll_sha256
            copied_original_sts2_sha256 = (Get-FileHash -LiteralPath ($CopiedDll + '.original') -Algorithm SHA256).Hash.ToLowerInvariant()
            patched_sts2_sha256 = (Get-FileHash -LiteralPath $CopiedDll -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        native_reference_game_version = '0.111.0'
        native_reference_steam_build = '24724944'
        native_parity = 'unverified'
    }
}

try {
    $Diagnostics = & (Join-Path $PSScriptRoot 'doctor_windows.ps1') -GameDirectory $GameDirectory -PythonExecutable $PythonExecutable -GitExecutable $GitExecutable -RequireNative:$BuildNative -AsObject
    if ($Diagnostics.missing_prerequisites.Count -gt 0) {
        foreach ($Attempt in $Diagnostics.python_discovery_attempts) { Write-Host ('Python candidate: ' + $Attempt.executable + ' - ' + $Attempt.reason) }
        throw ('Required tools were not detected: ' + ($Diagnostics.missing_prerequisites -join ', ') + '. Already installed? Pass -PythonExecutable and -GitExecutable with their full paths.')
    }
    $GitPath = $Diagnostics.git
    # Python backend provenance also invokes Git; expose this executable to children.
    $env:Path = (Split-Path -Parent $GitPath) + [System.IO.Path]::PathSeparator + $env:Path
    Write-Host ('Python: ' + $Diagnostics.python.version)
    foreach ($Gpu in $Diagnostics.gpu) { Write-Host ('Detected GPU: ' + $Gpu.name) }
    Write-Host 'Installing an isolated CPU training environment.'
    New-Item -ItemType Directory -Force -Path $ExternalRoot | Out-Null
    Get-PinnedRepository 'https://github.com/vvsuby/sts2-rl-agent.git' $EmulatorRoot $EmulatorCommit
    Get-PinnedRepository 'https://github.com/wuhao21/sts2-cli.git' $NativeRoot $NativeCommit
    $VenvRoot = Join-Path $ProjectRoot '.venv'
    $VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) { Invoke-Checked $Diagnostics.python.executable @('-m', 'venv', $VenvRoot) }
    Invoke-Checked $VenvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked $VenvPython @('-m', 'pip', 'install', ('torch==' + $TorchVersion), '--index-url', 'https://download.pytorch.org/whl/cpu')
    Invoke-Checked $VenvPython @('-m', 'pip', 'install', '-e', $EmulatorRoot)
    Invoke-Checked $VenvPython @('-m', 'pip', 'install', '-e', ($ProjectRoot + '[dev]'))
    Invoke-Checked $VenvPython @('-c', "import torch; from sb3_contrib import MaskablePPO; print('Training dependencies OK; torch=' + torch.__version__)")
    $Report = @{
        schema_version = 1
        created_at_utc = [DateTime]::UtcNow.ToString('o')
        project_root = $ProjectRoot
        training_device = 'cpu'
        emulator_path = $EmulatorRoot
        emulator_commit = $EmulatorCommit
        native_path = $NativeRoot
        native_commit = $NativeCommit
        native_status = 'not_built'
        diagnostics = $Diagnostics
    }
    if ($BuildNative) {
        Write-Host 'Building native runner from local copies of installed game DLLs.'
        $NativeReport = Build-NativeRunner $Diagnostics $VenvPython
        foreach ($Key in $NativeReport.Keys) { $Report[$Key] = $NativeReport[$Key] }
    }
    $ArtifactsRoot = Join-Path $ProjectRoot 'artifacts'
    New-Item -ItemType Directory -Force -Path $ArtifactsRoot | Out-Null
    $ReportPath = Join-Path $ArtifactsRoot 'windows-install.json'
    # UTF-8 without BOM lets Python json.load consume this on Windows PowerShell 5.1.
    $Encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($ReportPath, ($Report | ConvertTo-Json -Depth 10), $Encoding)
    Write-Host ''
    Write-Host ('Emulator training setup complete. Report: ' + $ReportPath)
    if ($BuildNative) {
        Write-Host 'Native build completed; live protocol and mechanics parity still need checking.'
        Write-Host '.\.venv\Scripts\spire2-rl.exe native-probe --config artifacts/windows-install.json --seed 7'
    } else {
        Write-Host 'Native game runner has not been built. Add -BuildNative when the game and .NET 9+ SDK are installed.'
    }
    Write-Host '.\.venv\Scripts\spire2-rl.exe train --experiment experiments/ironclad-a0 --upstream .external/sts2-rl-agent --steps 20000'
} catch {
    Write-Error ('Setup failed: ' + $_.Exception.Message) -ErrorAction Continue
    exit 1
}
