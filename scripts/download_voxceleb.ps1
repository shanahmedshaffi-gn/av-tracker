<#
.SYNOPSIS
  Automate download/preparation of VoxCeleb audio using clovaai/voxceleb_trainer dataprep.py

.DESCRIPTION
  This PowerShell helper clones the official `clovaai/voxceleb_trainer` repo
  and runs its `dataprep.py` script to download and prepare VoxCeleb1/2 audio
  (concatenate parts, extract archives, convert AAC->WAV 16k mono).

  The original dataprep expects `wget` and `ffmpeg`. On Windows it's
  recommended to run this script under WSL (Ubuntu) or install `wget` and
  `ffmpeg` (Chocolatey). The script will attempt to use WSL automatically
  if available.

.PARAMETER SavePath
  Target directory where VoxCeleb data will be placed (default: ..\data)

.PARAMETER UseWSL
  Force use of WSL even if native tools are available.

.EXAMPLE
  # Run using WSL (recommended on Windows)
  .\scripts\download_voxceleb.ps1 -SavePath C:\data\voxceleb -UseWSL

#>

param(
    [string]$SavePath = "..\data",
    [switch]$UseWSL
)

function Ensure-CommandExists($cmd) {
    return $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

# Check for WSL
$wslAvailable = $false
try { if (Get-Command wsl -ErrorAction SilentlyContinue) { $wslAvailable = $true } } catch { }

if ($UseWSL -and -not $wslAvailable) {
    Write-Error "WSL not found on this system. Remove -UseWSL or install WSL/Ubuntu."; exit 1
}

if ($wslAvailable -and -not $UseWSL) {
    Write-Host "WSL detected. This script will use WSL (recommended). To force native PowerShell mode, re-run with -UseWSL omitted." -ForegroundColor Cyan
}

if ($wslAvailable) {
    # Run inside WSL: clone repo and run dataprep.py using bash
    $pwdPath = $PWD.Path -replace ':','' -replace '\\','/'
    $wslSave = "/mnt/" + $pwdPath.Substring(0,1).ToLower() + $pwdPath.Substring(1) + "/../data"
    Write-Host "Cloning voxceleb_trainer inside WSL and running dataprep.py..." -ForegroundColor Green

    $commands = @(
        "set -e",
        "cd /tmp",
        "rm -rf voxceleb_trainer",
        "git clone https://github.com/clovaai/voxceleb_trainer.git",
        "cd voxceleb_trainer",
        "python3 -m pip install -r requirements.txt --user || true",
        "sudo apt-get update -y || true",
        "sudo apt-get install -y wget ffmpeg || true",
        "python3 dataprep.py --save_path $wslSave --download --user user --password pass",
        "python3 dataprep.py --save_path $wslSave --extract",
        "python3 dataprep.py --save_path $wslSave --convert"
    )

    $wslCmd = $commands -join "; "
    wsl bash -lc $wslCmd

    Write-Host "VoxCeleb data should be in: $($PWD.Path)\..\data (inside Windows path)." -ForegroundColor Green
    exit 0
}

# Native PowerShell path (if WSL not available)
Write-Host "WSL not available — running natively. Ensure 'wget' and 'ffmpeg' are installed and on PATH." -ForegroundColor Yellow

if (-not (Ensure-CommandExists wget)) {
    Write-Host "wget not found. Install it (e.g., Chocolatey: choco install wget) or install WSL." -ForegroundColor Red
    exit 1
}

if (-not (Ensure-CommandExists ffmpeg)) {
    Write-Host "ffmpeg not found. Install it (e.g., Chocolatey: choco install ffmpeg) or install WSL." -ForegroundColor Red
    exit 1
}

$repoDir = Join-Path $PWD "voxceleb_trainer"
if (Test-Path $repoDir) { Remove-Item -Recurse -Force $repoDir }

git clone https://github.com/clovaai/voxceleb_trainer.git
Set-Location voxceleb_trainer

python -m pip install -r requirements.txt

# Run dataprep steps (note: the original script expects --user and --password arguments)
python dataprep.py --save_path $SavePath --download --user user --password pass
python dataprep.py --save_path $SavePath --extract
python dataprep.py --save_path $SavePath --convert

Write-Host "Completed. Prepared VoxCeleb data under: $SavePath" -ForegroundColor Green
