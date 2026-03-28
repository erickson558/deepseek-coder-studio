param(
    [string]$PythonExe = "python"
)

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
    --name "llmstudio" `
    --icon $iconFile.FullName `
    $entryScript

$builtExe = Join-Path $root "dist\\llmstudio.exe"
$targetExe = Join-Path $root "llmstudio.exe"

if (-not (Test-Path -LiteralPath $builtExe)) {
    throw "PyInstaller did not generate the executable."
}

Copy-Item -LiteralPath $builtExe -Destination $targetExe -Force
Write-Host "Executable copied to $targetExe"
