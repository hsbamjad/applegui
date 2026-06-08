@echo off
:: ============================================================
:: install.bat  —  Launcher for install.ps1
:: Double-click THIS file to run the installer.
::
:: Windows blocks .ps1 scripts by default (ExecutionPolicy).
:: This .bat bypasses that restriction safely for this one script.
:: ============================================================

title Apple Sorting System — Installer

cd /d "%~dp0"

echo Starting installer...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"

:: Keep window open if PowerShell exits with an error
if errorlevel 1 (
    echo.
    echo  Installer encountered an error. See output above.
    pause
)
