# Build the CLI entrypoint into a standalone Windows executable.
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

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --name "llmstudio" `
    --icon $iconFile.FullName `
    $entryScript

# Copy the built file next to the Python entrypoint for the user's preferred layout.
$builtExe = Join-Path $root "dist\\llmstudio.exe"
$targetExe = Join-Path $root "llmstudio.exe"

if (-not (Test-Path -LiteralPath $builtExe)) {
    throw "PyInstaller did not generate the executable."
}

Copy-Item -LiteralPath $builtExe -Destination $targetExe -Force
Write-Host "Executable copied to $targetExe"
