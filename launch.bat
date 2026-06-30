@echo off
:: ============================================================
:: launch.bat  -  Infield Apple Sorting System
:: Michigan State University | ASABE AIM26 | 2026
:: ============================================================
title Infield Apple Sorting System
cd /d "%~dp0"

:: ── Find conda installation ───────────────────────────────────
:: conda is not on PATH in a plain CMD window (only in Anaconda Prompt).
:: We search common install locations and call its activate script directly.

set CONDA_ACTIVATE=
set CONDA_BASE=

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
        goto found_conda
    )
)

echo.
echo   ERROR: Could not find a conda installation.
echo   Searched: %%USERPROFILE%%\miniconda3, %%USERPROFILE%%\anaconda3, C:\miniconda3, etc.
echo   Please make sure Miniconda or Anaconda is installed.
echo.
pause
exit /b 1

:found_conda
echo   Found conda at: %CONDA_BASE%

:: ── Activate the applegui environment ────────────────────────
call "%CONDA_ACTIVATE%" applegui
if errorlevel 1 (
    echo.
    echo   ERROR: Could not activate the 'applegui' environment.
    echo   If this is a new machine, run install.bat first.
    echo.
    pause
    exit /b 1
)

:: ── Launch the app ────────────────────────────────────────────
echo   Starting Infield Apple Sorting System...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo   Application exited with an error. See output above.
    pause
)
