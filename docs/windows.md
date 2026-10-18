# Windows setup

The first training run uses CPU. The scripts find the Steam game folder and list your GPU automatically; you do not need to know either before starting. Training uses a pinned Python emulator until the native game runner passes its own checks. Emulator results do not establish performance in the installed game.

Install [Python 3.12 for Windows, 64-bit](https://www.python.org/downloads/windows/) and [Git for Windows](https://git-scm.com/download/win) if needed. Open PowerShell in the extracted project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/doctor_windows.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv\Scripts\spire2-rl.exe train --experiment experiments/ironclad-a0 --upstream .external/sts2-rl-agent --steps 20000
```

`-ExecutionPolicy Bypass` applies to this PowerShell process only. Setup writes inside this project: `.venv`, `.external`, and `artifacts/windows-install.json`. It does not change the Steam installation. Internet access is needed for dependencies. Downloading PyTorch may take a few minutes. If setup fails, it returns a nonzero exit code and reports the failed step; rerunning reuses the pinned checkouts and environment.

The setup supports Python 3.11–3.13 and starts with CPU PyTorch 2.8.0. GPU detection is informational. The Windows driver can report an inaccurate `AdapterRAM` value, so do not use that field as a reliable VRAM measurement.

If Python/Git are installed but detection fails, pass their full executable
paths. This also selects a supported Python when the default is 3.14:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -PythonExecutable "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -GitExecutable "C:\Program Files\Git\cmd\git.exe"
```

The doctor records rejected Python candidates and their reasons, including
unsupported versions or embedded runtimes missing `venv`/`ensurepip`.
Tool probes use arguments compatible with Windows PowerShell's legacy quoting.
Executable overrides are also available through `SPIRE2_PYTHON` and `SPIRE2_GIT`.
PATH refresh applies to the current process; setup does not edit the stored
user or system PATH. Full-path Git calls are used throughout setup.

## Native game connection

Install the [.NET 9+ SDK](https://dotnet.microsoft.com/download) for your Windows architecture. Then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -BuildNative
.\.venv\Scripts\spire2-rl.exe native-probe --config artifacts/windows-install.json --seed 7
```

Game detection checks Steam registry entries, the usual Program Files folders, and Steam's `libraryfolders.vdf`. If Steam is in an unusual location, supply the game folder explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -BuildNative -GameDirectory 'D:\SteamLibrary\steamapps\common\Slay the Spire 2'
```

The native runner copies installed DLLs into `.external/sts2-cli/lib`, keeps `sts2.dll.original`, runs the exact IL patcher from the pinned upstream `setup.sh`, and builds its headless .NET executable. It substitutes Godot stubs and alters asynchronous scheduling in the copied engine. This is a native mechanics backend with integration changes that require testing. The installed game's DLL hash is checked again afterward.

The pinned native project reports compatibility with game **v0.111.0**, Steam public-beta build **24724944**. A newer or older installed game can require adapter changes. The diagnostics record your installed build ID and DLL hash; a successful build or initial-decision probe does not prove that full runs work or that emulator policies transfer.

This Windows integration was authored in a Linux workspace without the game DLL or Windows runtime. It has not been validated against your installed game. `native-probe` starts an Ironclad A0 run and checks the initial protocol response; it does not train or complete a run. Native and emulator observations are separate interfaces, so the emulator policy is not a drop-in native policy.

The dependencies are pinned at:

| Dependency | Commit |
| --- | --- |
| `vvsuby/sts2-rl-agent` | `04a62c35b0ee264f5e4a3b5c6587f7438f1a7e0c` |
| `wuhao21/sts2-cli` | `084d1aa3d8e118ca7ce8d8774ad16d6be9c92367` |

The project does not redistribute the game DLLs. Keep the Steam game build fixed for experiments. After an intentional game update, rerun native setup and use a new experiment identity; previous checkpoints do not establish compatibility with the new build.
