@echo off
:: ============================================================
:: launch.bat  —  Infield Apple Sorting System
:: Michigan State University | ASABE AIM26 | 2026
::
:: Double-click this file to start the application.
:: Requires the 'applegui' conda environment to be set up first.
:: Run install.ps1 if this is a new machine.
:: ============================================================

title Infield Apple Sorting System

:: Move to the directory where this .bat file lives (the project root).
:: This ensures all relative paths (models/, config/) resolve correctly.
cd /d "%~dp0"

:: Check if the conda env exists
call conda activate applegui 2>nul
if errorlevel 1 (
    echo.
    echo  ERROR: The 'applegui' conda environment was not found.
    echo  Please run install.ps1 first to set up the environment.
    echo.
    pause
    exit /b 1
)

echo  Starting Infield Apple Sorting System...
python main.py

:: If python exits with an error, keep the window open so the user can read it
if errorlevel 1 (
    echo.
    echo  Application exited with an error. See output above.
    pause
)
