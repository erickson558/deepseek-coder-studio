# Build the GUI entrypoint into a standalone Windows executable.
param(
    [string]$PythonExe = "python"
)

# Resolve all paths from the repository root so the script works from CI and locally.
$root = Split-Path -Parent $PSScriptRoot
$entryScript = Join-Path $root "llmstudio.py"
$iconFile = Get-ChildItem -Path $root -Filter *.ico | Select-Object -First 1

if (-not (Test-Path -LiteralPath $entryScript)) {
    throw "Entry script not found: $entryScript"
}

if (-not $iconFile) {
    throw "No .ico file was found in $root"
}

$missingTrainingDeps = & $PythonExe -c "import importlib.util; required=('torch','datasets','transformers','peft'); missing=[name for name in required if importlib.util.find_spec(name) is None]; print(','.join(missing))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not verify Python dependencies before packaging."
}

if ($missingTrainingDeps) {
    throw "Missing training dependencies for a complete build: $missingTrainingDeps. Run '$PythonExe -m pip install -e .' before packaging."
}

$pyInstallerArgs = @(
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name",
    "llmstudio",
    "--distpath",
    $root,
    "--workpath",
    (Join-Path $root "build\\pyinstaller"),
    "--specpath",
    (Join-Path $root "build\\pyinstaller"),
    "--icon",
    $iconFile.FullName,
    "--hidden-import",
    "torch",
    "--hidden-import",
    "datasets",
    "--hidden-import",
    "transformers",
    "--hidden-import",
    "peft",
    $entryScript
)

& $PythonExe @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

# Build directly into the repository root so the .exe sits next to llmstudio.py.
$targetExe = Join-Path $root "llmstudio.exe"

if (-not (Test-Path -LiteralPath $targetExe)) {
    throw "PyInstaller did not generate the executable."
}
Write-Host "Executable created at $targetExe"
