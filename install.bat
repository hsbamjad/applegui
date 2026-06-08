@echo off
title Apple Sorting System -- Installer
cd /d "%~dp0"

echo.
echo ============================================================
echo   Infield Apple Sorting System - Setup
echo   Michigan State University  ^|  ASABE AIM26  ^|  2026
echo ============================================================
echo.

:: ── Step 1: Check conda is available ─────────────────────────
echo [1/4] Checking for Conda...
where conda >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ERROR: Conda was not found.
    echo   Please install Miniconda first, then re-run this installer.
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)
echo        Conda found.

:: ── Step 2: Create or update the conda environment ───────────
echo.
echo [2/4] Creating 'applegui' conda environment...
echo        This may take 5-15 minutes on the first run.
echo.

conda env create -f environment.yml
if errorlevel 1 (
    echo.
    echo        Environment may already exist. Trying update instead...
    conda env update -f environment.yml --prune
    if errorlevel 1 (
        echo.
        echo   ERROR: Failed to create or update the conda environment.
        echo   See output above for details.
        echo.
        pause
        exit /b 1
    )
)
echo.
echo        Environment ready.

:: ── Step 3: Install eBUS SDK wheel if present ─────────────────
echo.
echo [3/4] Looking for JAI eBUS SDK wheel...

set EBUS_DIR=C:\Program Files\Common Files\Pleora\eBUS SDK\Python
if exist "%EBUS_DIR%" (
    for %%f in ("%EBUS_DIR%\ebus_python*.whl") do (
        echo        Found: %%~nxf
        echo        Installing into applegui environment...
        conda run -n applegui pip install "%%f"
        if errorlevel 1 (
            echo        WARNING: eBUS install failed. App will use mock camera mode.
        ) else (
            echo        eBUS SDK installed successfully.
        )
        goto ebus_done
    )
    echo        No .whl file found. App will use mock camera mode.
) else (
    echo        eBUS SDK folder not found. App will use mock camera mode.
    echo        Install JAI eBUS SDK 6.x if you need the live camera.
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
echo   To start the app:
echo     - Double-click "Apple Sorter" on your Desktop, OR
echo     - Double-click launch.bat in this folder
echo ============================================================
echo.
pause
