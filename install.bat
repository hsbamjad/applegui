@echo off
:: ============================================================
:: install.bat  --  Infield Apple Sorting System
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

:: -- Step 1: Find conda ---------------------------------------
echo [1/5] Looking for conda installation...

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

:: -- Step 2: Create or update the conda environment ----------
echo.
echo [2/5] Setting up 'applegui' conda environment...
echo        (Installs all packages except PyTorch)
echo.

call "%CONDA_ACTIVATE%"

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

:: -- Step 3: Install PyTorch CUDA build ----------------------
::
:: torch is NOT in environment.yml because conda's pip subprocess
:: uses --extra-index-url which still hits PyPI first and grabs
:: the CPU-only build. Using --index-url here (primary index)
:: forces pip to fetch ONLY from the PyTorch CUDA wheel server.
echo.
echo [3/5] Installing PyTorch with CUDA support...

:: Check if a CUDA-enabled torch is already present
"%CONDA_EXE%" run -n applegui python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 (
    echo        PyTorch CUDA already installed and working. Skipping.
    goto torch_done
)

echo        Downloading PyTorch CUDA 12.8 build (~2.5 GB)...
echo        This will take several minutes - you will see a progress bar below.
echo.
"%CONDA_EXE%" run -n applegui pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

if errorlevel 1 (
    echo.
    echo   WARNING: PyTorch CUDA install failed.
    echo   The app may fall back to CPU mode. Check your connection and retry.
) else (
    echo.
    echo        PyTorch CUDA installed successfully.
)
:torch_done

:: -- Step 4: Check / install eBUS SDK ------------------------
echo.
echo [4/5] Checking JAI eBUS SDK...

"%CONDA_EXE%" run -n applegui pip show ebus_python >nul 2>&1
if not errorlevel 1 (
    echo        Already installed. Nothing to do.
    goto ebus_done
)

set EBUS_DIR=C:\Program Files\Common Files\Pleora\eBUS SDK\Python
if exist "%EBUS_DIR%" (
    for %%f in ("%EBUS_DIR%\ebus_python*.whl") do (
        echo        Found wheel: %%~nxf
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
    echo        eBUS SDK not found. App will use mock camera mode.
    echo        Install JAI eBUS SDK 6.x and re-run if you need the live camera.
)
:ebus_done

:: -- Step 5: Create Desktop shortcut -------------------------
echo.
echo [5/5] Creating Desktop shortcut...

set LAUNCH=%~dp0launch.bat
set SHORTCUT=%USERPROFILE%\Desktop\Apple Sorter.lnk

powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT%');$s.TargetPath='%LAUNCH%';$s.WorkingDirectory='%~dp0';$s.Save()"

if exist "%SHORTCUT%" (
    echo        Shortcut created on Desktop.
) else (
    echo        Could not create shortcut. Use launch.bat directly.
)

:: -- Done ----------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To start the app, double-click:  launch.bat
echo   or the "Apple Sorter" shortcut on your Desktop.
echo ============================================================
echo.
pause
