@echo off
:: ============================================================
:: install.bat  —  Infield Apple Sorting System
:: Michigan State University | ASABE AIM26 | 2026
::
:: Double-click this file to set up the app on a new machine.
:: After this runs once, use launch.bat to start the app.
::
:: What this does:
::   1. Creates the 'applegui' conda environment
::   2. Installs the JAI eBUS SDK wheel (if found)
::   3. Creates a Desktop shortcut for launch.bat
:: ============================================================

title Apple Sorting System — Installer
cd /d "%~dp0"

:: Run the embedded PowerShell block with ExecutionPolicy Bypass
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
"$ProjectDir = '%~dp0'.TrimEnd('\'); ^
$ErrorActionPreference = 'Stop'; ^
Write-Host ''; ^
Write-Host '============================================================' -ForegroundColor Cyan; ^
Write-Host '  Infield Apple Sorting System - Setup' -ForegroundColor Cyan; ^
Write-Host '  Michigan State University  |  ASABE AIM26  |  2026' -ForegroundColor Cyan; ^
Write-Host '============================================================' -ForegroundColor Cyan; ^
Write-Host ''; ^
Write-Host '[1/4] Checking for Conda...' -ForegroundColor Yellow; ^
try { $v = & conda --version 2>&1; Write-Host ('      Found: ' + $v) -ForegroundColor Green } ^
catch { Write-Host '  ERROR: Conda not found. Install Miniconda first:' -ForegroundColor Red; ^
        Write-Host '  https://docs.conda.io/en/latest/miniconda.html' -ForegroundColor Red; ^
        Read-Host 'Press Enter to exit'; exit 1 }; ^
Write-Host ''; ^
Write-Host '[2/4] Creating applegui conda environment...' -ForegroundColor Yellow; ^
Write-Host '      (This may take 5-15 minutes on first run)'; ^
$envFile = Join-Path $ProjectDir 'environment.yml'; ^
$envExists = conda env list | Select-String 'applegui'; ^
if ($envExists) { Write-Host '      Already exists - updating...' -ForegroundColor Cyan; conda env update -f $envFile --prune } ^
else { conda env create -f $envFile }; ^
if ($LASTEXITCODE -ne 0) { Write-Host '  ERROR: conda env create failed.' -ForegroundColor Red; Read-Host 'Press Enter to exit'; exit 1 }; ^
Write-Host '      Environment ready.' -ForegroundColor Green; ^
Write-Host ''; ^
Write-Host '[3/4] Looking for JAI eBUS SDK wheel...' -ForegroundColor Yellow; ^
$ebus = 'C:\Program Files\Common Files\Pleora\eBUS SDK\Python'; ^
if (Test-Path $ebus) { ^
    $whl = Get-ChildItem $ebus -Filter 'ebus_python*.whl' -EA SilentlyContinue | Select-Object -First 1; ^
    if ($whl) { Write-Host ('      Found: ' + $whl.Name) -ForegroundColor Green; ^
                conda run -n applegui pip install $whl.FullName; ^
                if ($LASTEXITCODE -eq 0) { Write-Host '      eBUS SDK installed.' -ForegroundColor Green } ^
                else { Write-Host '      WARNING: eBUS install failed - app will use mock camera.' -ForegroundColor DarkYellow } } ^
    else { Write-Host '      No wheel found - app will use mock camera mode.' -ForegroundColor DarkYellow } } ^
else { Write-Host '      eBUS SDK not installed - app will use mock camera mode.' -ForegroundColor DarkYellow }; ^
Write-Host ''; ^
Write-Host '[4/4] Creating Desktop shortcut...' -ForegroundColor Yellow; ^
$launchBat = Join-Path $ProjectDir 'launch.bat'; ^
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Apple Sorter.lnk'); ^
$shortcut.TargetPath = $launchBat; ^
$shortcut.WorkingDirectory = $ProjectDir; ^
$shortcut.Description = 'Infield Apple Sorting System'; ^
$shortcut.Save(); ^
Write-Host '      Shortcut created on Desktop.' -ForegroundColor Green; ^
Write-Host ''; ^
Write-Host '============================================================' -ForegroundColor Green; ^
Write-Host '  Setup complete!' -ForegroundColor Green; ^
Write-Host '  Launch the app by double-clicking launch.bat' -ForegroundColor White; ^
Write-Host '  or the Apple Sorter shortcut on your Desktop.' -ForegroundColor White; ^
Write-Host '============================================================' -ForegroundColor Green; ^
Write-Host ''; ^
Read-Host 'Press Enter to close'"

pause
