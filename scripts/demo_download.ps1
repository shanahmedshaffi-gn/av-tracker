<#
.SYNOPSIS
  Demo downloader: fetch a small set of VoxCeleb audio samples for quick tests.

.DESCRIPTION
  Downloads up to N entries from the VoxCeleb train_list (official metadata),
  extracts audio via `yt-dlp`, converts to WAV 16 kHz mono and trims to a short
  segment (default 5s). Output is saved under `data/demo/<speaker_id>/`.

.REQUIREMENTS
  - PowerShell
  - `yt-dlp` on PATH (pip install -U yt-dlp)
  - `ffmpeg` on PATH

.EXAMPLE
  # Download 10 demo samples
  .\scripts\demo_download.ps1 -Count 10
#>

param(
    [int]$Count = 10,
    [int]$TrimSeconds = 5,
    [string]$OutRoot = "data\demo",
    [switch]$UseTrainListUrl
)

function Ensure-CommandExists($cmd) {
    return $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

if (-not (Ensure-CommandExists yt-dlp)) {
    Write-Error "yt-dlp not found on PATH. Install with: pip install -U yt-dlp"
    exit 1
}
if (-not (Ensure-CommandExists ffmpeg)) {
    Write-Error "ffmpeg not found on PATH. Install it (choco install ffmpeg) or via WSL."; exit 1
}

$trainListUrl = 'http://www.robots.ox.ac.uk/~joon/data/voxceleb/meta/train_list.txt'

if ($UseTrainListUrl) { Write-Host "Forcing fetch from train_list URL: $trainListUrl" }

Write-Host "Fetching train_list (official VoxCeleb metadata)..." -ForegroundColor Cyan
try {
    $content = (Invoke-WebRequest -UseBasicParsing -Uri $trainListUrl -ErrorAction Stop).Content
    $lines = $content -split "\r?\n" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
} catch {
    Write-Warning "Failed to download train_list.txt from $trainListUrl. Trying local file 'train_list.txt' or built-in demo IDs..."

    # Try local train_list.txt in repo root
    $localPath = Join-Path $PWD 'train_list.txt'
    if (Test-Path $localPath) {
        Write-Host "Using local file: $localPath" -ForegroundColor Cyan
        $content = Get-Content -Raw -Path $localPath
        $lines = $content -split "\r?\n" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
    } else {
        Write-Warning "Local train_list.txt not found. Falling back to built-in demo YouTube IDs (for quick test)."

        # Built-in fallback: 10 public YouTube IDs (short demo only)
        $builtinIds = @(
            '21Uxsk56VDQ', '2Z4m4lnjxkY', '3JZ_D3ELwOQ', 'M3mJkSqZbX4', 'kXYiU_JCYtU',
            'e-ORhEE9VVg', 'DLzxrzFCyOs', '9bZkp7q19f0', 'dQw4w9WgXcQ', 'hY7m5jjJ9mM'
        )

        # Create pseudo 'lines' in VoxCeleb train_list format: 'id00000 id00000/VIDEOID/00001.wav'
        $lines = @()
        $i = 0
        foreach ($vid in $builtinIds) {
            $spkid = "demo$( '{0:D5}' -f $i )"
            $lines += "$spkid $spkid/$vid/00001.wav"
            $i++
        }
    }
}

if (-not $lines -or $lines.Count -eq 0) {
    Write-Error "No entries available for demo. Provide a local 'train_list.txt' file or enable network access."; exit 1
}
if ($lines.Count -eq 0) { Write-Error "Empty train_list"; exit 1 }

Write-Host "Selecting $Count samples from train_list..."

$selected = $lines | Select-Object -First $Count

foreach ($line in $selected) {
    # line format: "id00012 id00012/21Uxsk56VDQ/00001.wav"
    $parts = $line.Trim().Split()
    if ($parts.Length -lt 2) { continue }
    $speaker = $parts[0]
    $path = $parts[1]
    $pathParts = $path.Split('/')
    if ($pathParts.Length -lt 2) { continue }
    $videoId = $pathParts[1]

    $outDir = Join-Path $PWD $OutRoot
    $speakerDir = Join-Path $outDir $speaker
    if (-not (Test-Path $speakerDir)) { New-Item -ItemType Directory -Path $speakerDir -Force | Out-Null }

    $outTemplate = Join-Path $speakerDir ("$videoId.%(ext)s")
    $url = "https://www.youtube.com/watch?v=$videoId"

    Write-Host "Downloading audio for speaker=$speaker video=$videoId ..." -ForegroundColor Green
    # Extract audio as wav and force sample rate conversion via ffmpeg postprocessor args
    yt-dlp $url -x --audio-format wav --audio-quality 0 --postprocessor-args "-ar 16000 -ac 1" -o $outTemplate

    # Trim produced wav to TrimSeconds to keep samples small
    $inWav = Join-Path $speakerDir ("$videoId.wav")
    if (Test-Path $inWav) {
        $trimmed = Join-Path $speakerDir ("${videoId}_trim.wav")
        ffmpeg -y -i $inWav -ss 0 -t $TrimSeconds -ac 1 -ar 16000 $trimmed >/dev/null 2>&1
        if (Test-Path $trimmed) {
            Move-Item -Force $trimmed $inWav
            Write-Host "Saved: $inWav"
        }
    } else {
        Write-Warning "Expected output not found: $inWav"
    }
}

Write-Host "Demo download finished. Files saved under: $(Resolve-Path $OutRoot)" -ForegroundColor Green
