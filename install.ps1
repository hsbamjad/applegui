# =============================================================================
# install.ps1  —  One-time setup for the Infield Apple Sorting System
# Michigan State University | ASABE AIM26 | 2026
#
# Run this ONCE on any new machine. After this, just double-click launch.bat.
#
# What this script does:
#   1. Checks that Conda (Miniconda/Anaconda) is installed
#   2. Creates the 'applegui' conda environment from environment.yml
#   3. Installs the eBUS SDK Python wheel if the SDK is found on this machine
#   4. Creates a desktop shortcut for launch.bat
#
# Requirements on the target machine:
#   - Miniconda or Anaconda  (https://docs.conda.io/en/latest/miniconda.html)
#   - NVIDIA driver >= 525.x  (for CUDA 12.x support)
#   - JAI eBUS SDK 6.x        (for live JAI camera — optional, app works in mock mode without it)
# =============================================================================

param(
    [string]$ProjectDir = $PSScriptRoot   # defaults to the folder containing this script
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Infield Apple Sorting System — Setup" -ForegroundColor Cyan
Write-Host "  Michigan State University | ASABE AIM26 | 2026" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check Conda ───────────────────────────────────────────────────────
Write-Host "[1/4] Checking for Conda..." -ForegroundColor Yellow
try {
    $condaVersion = & conda --version 2>&1
    Write-Host "      Found: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "  ERROR: Conda not found on this machine." -ForegroundColor Red
    Write-Host "  Please install Miniconda first:" -ForegroundColor Red
    Write-Host "  https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Red
    Write-Host ""
    Read-Host "  Press Enter to exit"
    exit 1
}

# ── Step 2: Create conda environment ─────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] Creating 'applegui' conda environment..." -ForegroundColor Yellow
Write-Host "      This will download PyTorch (CUDA 12.4) and other packages."
Write-Host "      This may take 5–15 minutes depending on your connection."
Write-Host ""

$envFile = Join-Path $ProjectDir "environment.yml"
if (-not (Test-Path $envFile)) {
    Write-Host "  ERROR: environment.yml not found at: $envFile" -ForegroundColor Red
    exit 1
}

# Check if env already exists
$envExists = conda env list | Select-String "applegui"
if ($envExists) {
    Write-Host "      'applegui' environment already exists. Updating..." -ForegroundColor Cyan
    conda env update -f $envFile --prune
} else {
    conda env create -f $envFile
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  ERROR: conda env create failed. See output above." -ForegroundColor Red
    exit 1
}
Write-Host "      Environment ready." -ForegroundColor Green

# ── Step 3: Install eBUS SDK wheel ───────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Looking for JAI eBUS SDK Python wheel..." -ForegroundColor Yellow

$ebusSdkPath = "C:\Program Files\Common Files\Pleora\eBUS SDK\Python"
if (Test-Path $ebusSdkPath) {
    $wheel = Get-ChildItem $ebusSdkPath -Filter "ebus_python*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($wheel) {
        Write-Host "      Found wheel: $($wheel.Name)" -ForegroundColor Green
        Write-Host "      Installing into 'applegui' environment..."
        conda run -n applegui pip install $wheel.FullName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "      eBUS SDK installed successfully." -ForegroundColor Green
        } else {
            Write-Host "      WARNING: eBUS wheel install failed. App will use mock camera mode." -ForegroundColor DarkYellow
        }
    } else {
        Write-Host "      WARNING: No .whl file found in $ebusSdkPath" -ForegroundColor DarkYellow
        Write-Host "      App will use mock camera mode (no live JAI camera)." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "      eBUS SDK not found on this machine." -ForegroundColor DarkYellow
    Write-Host "      App will run in mock/simulation camera mode." -ForegroundColor DarkYellow
    Write-Host "      Install JAI eBUS SDK 6.x if you need the live camera." -ForegroundColor DarkYellow
}

# ── Step 4: Create desktop shortcut ──────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Creating desktop shortcut..." -ForegroundColor Yellow

$launchBat = Join-Path $ProjectDir "launch.bat"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Apple Sorter.lnk"

try {
    $WshShell = New-Object -ComObject WScript.Shell
    $shortcut = $WshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launchBat
    $shortcut.WorkingDirectory = $ProjectDir
    $shortcut.Description = "Infield Apple Sorting System — MSU ASABE AIM26"
    $shortcut.Save()
    Write-Host "      Shortcut created: $shortcutPath" -ForegroundColor Green
} catch {
    Write-Host "      Could not create shortcut. You can still run launch.bat directly." -ForegroundColor DarkYellow
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To start the app:" -ForegroundColor White
Write-Host "    - Double-click 'Apple Sorter' on your Desktop" -ForegroundColor White
Write-Host "    - Or double-click launch.bat in this folder" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close"
