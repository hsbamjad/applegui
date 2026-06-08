@echo off
:: ============================================================
:: install.bat  —  Infield Apple Sorting System
::
:: Run this ONCE on a new machine, then use launch.bat every time.
:: ============================================================
title Apple Sorting System -- Installer
cd /d "%~dp0"

echo.
echo ============================================================
echo   Infield Apple Sorting System - Setup

echo ============================================================
echo.

:: ── Step 1: Find conda ───────────────────────────────────────
echo [1/4] Looking for conda installation...

set CONDA_ACTIVATE=
set CONDA_BASE=
set CONDA_EXE=

for %%P in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\Anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\miniconda3"
    "C:\anaconda3"
) do (
    if exist "%%~P\Scripts\activate.bat" (
        set CONDA_BASE=%%~P
        set CONDA_ACTIVATE=%%~P\Scripts\activate.bat
        set CONDA_EXE=%%~P\Scripts\conda.exe
        goto found_conda
    )
)

echo.
echo   ERROR: Could not find a conda installation.
echo   Please install Miniconda from:
echo   https://docs.conda.io/en/latest/miniconda.html
echo.
pause
exit /b 1

:found_conda
echo        Found conda at: %CONDA_BASE%

:: ── Step 2: Create or update the conda environment ───────────
echo.
echo [2/4] Setting up 'applegui' conda environment...
echo        (First run may take 5-15 minutes to download packages)
echo.

:: Initialize conda for use in this CMD session
call "%CONDA_ACTIVATE%"

:: Check if env already exists
"%CONDA_EXE%" env list | findstr /C:"applegui" >nul 2>&1
if not errorlevel 1 (
    echo        Environment already exists. Updating...
    "%CONDA_EXE%" env update -f environment.yml --prune
) else (
    "%CONDA_EXE%" env create -f environment.yml
)

if errorlevel 1 (
    echo.
    echo   ERROR: Failed to create the conda environment.
    echo   See output above for details.
    echo.
    pause
    exit /b 1
)
echo.
echo        Environment ready.

:: ── Step 3: Check / install eBUS SDK ────────────────────────────
echo.
echo [3/4] Checking JAI eBUS SDK...

:: First: check if ebus_python is already installed in the env
"%CONDA_EXE%" run -n applegui pip show ebus_python >nul 2>&1
if not errorlevel 1 (
    echo        Already installed in applegui env. Nothing to do.
    goto ebus_done
)

:: Not installed — look for the .whl file and install it
set EBUS_DIR=C:\Program Files\Common Files\Pleora\eBUS SDK\Python
if exist "%EBUS_DIR%" (
    for %%f in ("%EBUS_DIR%\ebus_python*.whl") do (
        echo        Found wheel: %%~nxf
        echo        Installing into applegui environment...
        "%CONDA_EXE%" run -n applegui pip install "%%f"
        if errorlevel 1 (
            echo        WARNING: eBUS install failed. App will use mock camera mode.
        ) else (
            echo        eBUS SDK installed successfully.
        )
        goto ebus_done
    )
    echo        No .whl found in SDK folder. App will use mock camera mode.
) else (
    echo        eBUS SDK folder not found. App will use mock camera mode.
    echo        If you need live camera, install JAI eBUS SDK 6.x then re-run.
)
:ebus_done

:: ── Step 4: Create a Desktop shortcut ─────────────────────────
echo.
echo [4/4] Creating Desktop shortcut...

set LAUNCH=%~dp0launch.bat
set SHORTCUT=%USERPROFILE%\Desktop\Apple Sorter.lnk

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT%');$s.TargetPath='%LAUNCH%';$s.WorkingDirectory='%~dp0';$s.Save()"

if exist "%SHORTCUT%" (
    echo        Shortcut created on Desktop.
) else (
    echo        Could not create shortcut. Use launch.bat directly.
)

:: ── Done ──────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To start the app, double-click:  launch.bat
echo   or the "Apple Sorter" shortcut on your Desktop.
echo ============================================================
echo.
pause
