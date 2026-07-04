# Build the ptcg_engine pybind module with MSVC + Ninja.
# Locates vcvars64 via vswhere, configures with the active python's pybind11,
# then copies the built .pyd next to this script for easy `import ptcg_engine`.
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$build = Join-Path $here "build"

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found at $vswhere" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$py = (Get-Command python).Source
$pybindDir = (& python -m pybind11 --cmakedir).Trim()

$cfg = "cmake -S `"$here`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=`"$pybindDir`" -DPython_EXECUTABLE=`"$py`""
$bld = "cmake --build `"$build`""
# vcvars64.bat calls bare `vswhere.exe`; put the Installer dir on PATH so it resolves.
$installerDir = Split-Path $vswhere
& cmd /c "set `"PATH=$installerDir;%PATH%`" && call `"$vcvars`" && $cfg && $bld"
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

Get-ChildItem -Path $build -Filter "ptcg_engine*.pyd" -Recurse | ForEach-Object {
  Copy-Item $_.FullName -Destination $here -Force
  Write-Output "copied $($_.Name) -> $here"
}
Write-Output "OK"
