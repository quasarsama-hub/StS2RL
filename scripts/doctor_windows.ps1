[CmdletBinding()]
param(
    [string]$GameDirectory = $env:STS2_GAME_DIR,
    [string]$PythonExecutable = $env:SPIRE2_PYTHON,
    [string]$GitExecutable = $env:SPIRE2_GIT,
    [switch]$AsObject,
    [switch]$AsJson,
    [switch]$RequireNative,
    [switch]$FunctionsOnly
)

$ErrorActionPreference = 'Stop'

function Get-PythonDetails {
    param([string]$ExplicitExecutable)
    $script:PythonDiscoveryAttempts = @()
    $Candidates = @()
    if ($ExplicitExecutable) {
        $Candidates += @{ executable = $ExplicitExecutable; arguments = @() }
    } else {
        if (Get-Command py -CommandType Application -ErrorAction SilentlyContinue) {
            foreach ($Version in @('-3.12', '-3.13', '-3.11')) {
                $Candidates += @{ executable = 'py'; arguments = @($Version) }
            }
        }
        if (Get-Command python -CommandType Application -ErrorAction SilentlyContinue) {
            $Candidates += @{ executable = 'python'; arguments = @() }
        }
        foreach ($Base in @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
            if (-not $Base) { continue }
            foreach ($Minor in @('312', '313', '311')) {
                foreach ($Relative in @("Programs\Python\Python$Minor\python.exe", "Python$Minor\python.exe")) {
                    $Path = Join-Path $Base $Relative
                    if (Test-Path -LiteralPath $Path -PathType Leaf) {
                        $Candidates += @{ executable = $Path; arguments = @() }
                    }
                }
            }
        }
    }
    foreach ($Candidate in $Candidates) {
        # Windows PowerShell 5.1 strips embedded double quotes in native argv.
        # Single quotes INSIDE Python code survive that legacy argument parser.
        $Probe = "import json,sys,struct,importlib.util; print(json.dumps({'executable':sys.executable,'version':list(sys.version_info[:3]),'bits':struct.calcsize('P')*8,'venv':importlib.util.find_spec('venv') is not None,'ensurepip':importlib.util.find_spec('ensurepip') is not None}))"
        $Arguments = @($Candidate.arguments) + @('-c', $Probe)
        # Missing py-launcher versions and Store aliases are normal discovery failures.
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $Output = $null
        $Code = -1
        try { $Output = & $Candidate.executable @Arguments 2>$null; $Code = $LASTEXITCODE }
        catch { $Code = -1 }
        finally { $ErrorActionPreference = $PreviousPreference }
        if ($Code -ne 0 -or -not $Output) {
            $script:PythonDiscoveryAttempts += @{ executable = $Candidate.executable; reason = 'Probe could not run; executable/launcher may be unavailable.' }
            continue
        }
        try { $Details = ($Output -join "`n") | ConvertFrom-Json }
        catch {
            $script:PythonDiscoveryAttempts += @{ executable = $Candidate.executable; reason = 'Probe did not return valid JSON.' }
            continue
        }
        $Reason = $null
        if ($Details.bits -ne 64) { $Reason = '64-bit Python is required.' }
        elseif ($Details.version[0] -ne 3 -or $Details.version[1] -lt 11 -or $Details.version[1] -gt 13) { $Reason = 'Supported Python versions are 3.11-3.13; choose an installed supported version.' }
        elseif (-not $Details.venv -or -not $Details.ensurepip) { $Reason = 'This Python is missing venv/ensurepip; choose a full Python installation instead of an embedded runtime.' }
        if (-not $Reason) {
            return @{ executable = $Details.executable; version = ($Details.version -join '.'); bits = $Details.bits }
        }
        $script:PythonDiscoveryAttempts += @{ executable = $Details.executable; version = ($Details.version -join '.'); reason = $Reason }
    }
    return $null
}

function Refresh-ProcessPath {
    $Parts = @($env:Path)
    foreach ($Scope in @('Machine', 'User')) {
        $Value = [Environment]::GetEnvironmentVariable('Path', $Scope)
        if ($Value) { $Parts += [Environment]::ExpandEnvironmentVariables($Value) }
    }
    $env:Path = ($Parts | Where-Object { $_ } | Select-Object -Unique) -join [System.IO.Path]::PathSeparator
}

function Get-GitExecutable {
    param([string]$ExplicitExecutable)
    if ($ExplicitExecutable) {
        $Resolved = Get-Command $ExplicitExecutable -CommandType Application -ErrorAction SilentlyContinue
        if (-not $Resolved) { return $null }
        $Candidates = @($Resolved.Source)
    } else {
        $Candidates = @()
        $OnPath = Get-Command git -CommandType Application -ErrorAction SilentlyContinue
        if ($OnPath) { $Candidates += $OnPath.Source }
        foreach ($Base in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
            if (-not $Base) { continue }
            foreach ($Relative in @('Git\cmd\git.exe', 'Programs\Git\cmd\git.exe')) {
                $Path = Join-Path $Base $Relative
                if (Test-Path -LiteralPath $Path -PathType Leaf) { $Candidates += $Path }
            }
        }
    }
    foreach ($Candidate in @($Candidates | Select-Object -Unique)) {
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $Code = -1
        try { $Version = & $Candidate --version 2>$null; $Code = $LASTEXITCODE }
        catch { $Code = -1 }
        finally { $ErrorActionPreference = $PreviousPreference }
        if ($Code -eq 0 -and ($Version -join '') -match '^git version ') { return $Candidate }
    }
    return $null
}

function Get-SteamLibraries {
    $Roots = New-Object 'System.Collections.Generic.List[string]'
    foreach ($RegistryPath in @('HKCU:\Software\Valve\Steam', 'HKLM:\SOFTWARE\WOW6432Node\Valve\Steam', 'HKLM:\SOFTWARE\Valve\Steam')) {
        $Properties = Get-ItemProperty -LiteralPath $RegistryPath -ErrorAction SilentlyContinue
        if ($Properties) {
            foreach ($Name in @('SteamPath', 'InstallPath')) {
                if ($Properties.PSObject.Properties[$Name] -and $Properties.$Name) { $Roots.Add([string]$Properties.$Name) }
            }
        }
    }
    if (${env:ProgramFiles(x86)}) { $Roots.Add((Join-Path ${env:ProgramFiles(x86)} 'Steam')) }
    if ($env:ProgramFiles) { $Roots.Add((Join-Path $env:ProgramFiles 'Steam')) }
    $Libraries = New-Object 'System.Collections.Generic.List[string]'
    foreach ($SteamRoot in @($Roots | Select-Object -Unique)) {
        $Libraries.Add($SteamRoot)
        $Vdf = Join-Path $SteamRoot 'steamapps\libraryfolders.vdf'
        if (-not (Test-Path -LiteralPath $Vdf -PathType Leaf)) { continue }
        $Text = Get-Content -LiteralPath $Vdf -Raw
        # Current and older Steam libraryfolders formats; decode doubled backslashes only.
        foreach ($Match in [regex]::Matches($Text, '"(?:path|[0-9]+)"\s*"((?:\\.|[^"\\])*)"')) {
            $Library = $Match.Groups[1].Value.Replace('\\', '\')
            if ([System.IO.Path]::IsPathRooted($Library)) { $Libraries.Add($Library) }
        }
    }
    return @($Libraries | Select-Object -Unique)
}

function Find-Game {
    param([string]$ExplicitDirectory, [string[]]$Libraries)
    $Candidates = New-Object 'System.Collections.Generic.List[string]'
    if ($ExplicitDirectory) { $Candidates.Add($ExplicitDirectory) }
    else {
        foreach ($Library in $Libraries) { $Candidates.Add((Join-Path $Library 'steamapps\common\Slay the Spire 2')) }
    }
    foreach ($Candidate in $Candidates) {
        if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) { continue }
        $Resolved = (Resolve-Path -LiteralPath $Candidate).Path
        $DirectDll = Join-Path $Resolved 'sts2.dll'
        if (Test-Path -LiteralPath $DirectDll -PathType Leaf) { $Dll = Get-Item -LiteralPath $DirectDll }
        else { $Dll = Get-ChildItem -LiteralPath $Resolved -Filter 'sts2.dll' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 }
        if ($Dll) {
            $Manifest = $null
            foreach ($Library in $Libraries) {
                $ManifestPath = Join-Path $Library 'steamapps\appmanifest_2868840.acf'
                if (Test-Path -LiteralPath $ManifestPath -PathType Leaf) {
                    $ManifestText = Get-Content -LiteralPath $ManifestPath -Raw
                    $InstallDirectory = [regex]::Match($ManifestText, '"installdir"\s*"([^"]+)"').Groups[1].Value
                    $InstalledPath = Join-Path $Library (Join-Path 'steamapps\common' $InstallDirectory)
                    if ($Resolved.StartsWith($InstalledPath, [StringComparison]::OrdinalIgnoreCase)) {
                        $Manifest = @{ path = $ManifestPath; build_id = [regex]::Match($ManifestText, '"buildid"\s*"([^"]+)"').Groups[1].Value }
                        break
                    }
                }
            }
            return @{
                install_directory = $Resolved
                data_directory = $Dll.DirectoryName
                dll = $Dll.FullName
                dll_sha256 = (Get-FileHash -LiteralPath $Dll.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                file_version = $Dll.VersionInfo.FileVersion
                steam_manifest = $Manifest
            }
        }
    }
    return $null
}

if ($FunctionsOnly) { return }
Refresh-ProcessPath
$Libraries = @(Get-SteamLibraries)
$Python = Get-PythonDetails -ExplicitExecutable $PythonExecutable
$Git = Get-GitExecutable -ExplicitExecutable $GitExecutable
$Dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
$Sdks = @()
if ($Dotnet) { $Sdks = @(& $Dotnet.Source --list-sdks) }
$HasSdk9 = @($Sdks | Where-Object { $_ -match '^(9|[1-9][0-9])\.' }).Count -gt 0
$Gpu = @()
try {
    $Gpu = @(Get-CimInstance Win32_VideoController | ForEach-Object {
        @{ name = $_.Name; driver_version = $_.DriverVersion; adapter_ram_bytes = $_.AdapterRAM }
    })
} catch { $Gpu = @(@{ name = 'GPU detection unavailable'; reason = $_.Exception.Message }) }
$Game = Find-Game -ExplicitDirectory $GameDirectory -Libraries $Libraries
$Missing = New-Object 'System.Collections.Generic.List[string]'
if (-not $Python) { $Missing.Add('Python 3.11-3.13 (64-bit)') }
if (-not $Git) { $Missing.Add('Git for Windows') }
if ($RequireNative -and -not $HasSdk9) { $Missing.Add('.NET 9+ SDK') }
if ($RequireNative -and -not $Game) { $Missing.Add('Slay the Spire 2 game DLL (use -GameDirectory if Steam detection fails)') }
$Report = @{
    python = $Python
    git = $Git
    python_discovery_attempts = @($script:PythonDiscoveryAttempts)
    dotnet = $(if ($Dotnet) { $Dotnet.Source } else { $null })
    dotnet_sdks = $Sdks
    has_dotnet_9_sdk = $HasSdk9
    gpu = $Gpu
    steam_libraries = $Libraries
    game = $Game
    missing_prerequisites = @($Missing)
}
if ($AsObject) { return $Report }
if ($AsJson) { $Report | ConvertTo-Json -Depth 8 }
else {
    Write-Host 'Spire 2 RL Windows diagnostics'
    Write-Host ('Python: ' + $(if ($Python) { $Python.version + ' at ' + $Python.executable } else { 'missing; install Python 3.12 x64' }))
    Write-Host ('Git: ' + $(if ($Git) { $Git } else { 'missing; specify -GitExecutable with the full path' }))
    if (-not $Python) {
        foreach ($Attempt in $script:PythonDiscoveryAttempts) { Write-Host ('Python candidate: ' + $Attempt.executable + ' - ' + $Attempt.reason) }
    }
    Write-Host ('Native SDK: ' + $(if ($HasSdk9) { $Sdks -join ', ' } else { '.NET 9+ SDK missing; emulator training still works' }))
    foreach ($Device in $Gpu) { Write-Host ('GPU: ' + $Device.name) }
    Write-Host 'Training defaults to CPU; no GPU setup is required for the first run.'
    Write-Host ('Game DLL: ' + $(if ($Game) { $Game.dll } else { 'not found; pass -GameDirectory with your Steam game folder' }))
    if ($Game) {
        Write-Host ('Game DLL SHA256: ' + $Game.dll_sha256)
        if ($Game.steam_manifest) { Write-Host ('Steam build: ' + $Game.steam_manifest.build_id) }
    }
    foreach ($Item in $Missing) { Write-Host ('Missing prerequisite: ' + $Item) }
}
if ($Missing.Count -gt 0) { exit 1 }
