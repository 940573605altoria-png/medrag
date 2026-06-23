# GitHub push helper (run in YOUR OWN interactive PowerShell terminal).
#
# Output is ASCII-only on purpose: PowerShell 5.1 reads scripts as GBK and would
# garble UTF-8 Chinese. Keep messages in English to avoid mojibake.
#
# Why run it yourself: the first push opens a GitHub login window (Git Credential
# Manager), which a background/sandbox shell cannot display.
#
# China network note: github.com over HTTPS is often reset without a proxy. If you
# get "Connection was reset", pass your local proxy via -Proxy, e.g.:
#     -Proxy http://127.0.0.1:7890     (Clash default)
#     -Proxy http://127.0.0.1:10809    (V2RayN HTTP default)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1 -Proxy http://127.0.0.1:7890
#   powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1 -Token ghp_xxx -Proxy http://127.0.0.1:7890

param(
    [string]$RemoteUrl = "https://github.com/940573605altoria-png/medrag.git",
    [string]$Branch    = "main",
    [string]$Token     = "",
    [string]$Proxy     = "",
    [int]$Retries      = 3
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Put the medrag conda env's git on PATH (git is installed inside that env).
$r = "F:\miniconda\envs\medrag"
$env:PATH = "$r;$r\Library\mingw-w64\bin;$r\Library\usr\bin;$r\Library\bin;$r\Scripts;" + $env:PATH

# cd to repo root (parent of scripts\)
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "[repo] $repoRoot" -ForegroundColor Cyan

git rev-parse --is-inside-work-tree > $null
if (-not $?) { throw "Not a git repository: $repoRoot" }

# Ensure origin points to the right URL
$existing = (git remote get-url origin) 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[remote] add origin -> $RemoteUrl" -ForegroundColor Yellow
    git remote add origin $RemoteUrl
} elseif ($existing -ne $RemoteUrl) {
    Write-Host "[remote] set origin: $existing -> $RemoteUrl" -ForegroundColor Yellow
    git remote set-url origin $RemoteUrl
}

Write-Host "`n[commits to push]" -ForegroundColor Cyan
git log $Branch --oneline -n 5

# Build push args. Apply proxy per-command (no global config side effects).
$pushUrl = "origin"
if ($Token) {
    $u = [System.Uri]$RemoteUrl
    $pushUrl = "https://$Token@$($u.Host)$($u.AbsolutePath)"
}
$gitArgs = @()
if ($Proxy) {
    Write-Host "[proxy] $Proxy" -ForegroundColor Yellow
    $gitArgs += @("-c", "http.proxy=$Proxy", "-c", "https.proxy=$Proxy")
}

Write-Host "`n[push] --force-with-lease (safely overwrites the empty Initial commit)..." -ForegroundColor Green

$ok = $false
for ($i = 1; $i -le $Retries; $i++) {
    Write-Host "  attempt $i/$Retries ..." -ForegroundColor DarkGray
    if ($Token) {
        git @gitArgs push --force-with-lease $pushUrl "${Branch}:${Branch}"
    } else {
        git @gitArgs push --force-with-lease -u origin $Branch
    }
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Start-Sleep -Seconds 2
}

if ($ok) {
    Write-Host "`n[OK] pushed to $RemoteUrl ($Branch)" -ForegroundColor Green
    Write-Host "[AutoDL] git clone $RemoteUrl" -ForegroundColor Cyan
} else {
    Write-Host "`n[FAIL] push failed (exit=$LASTEXITCODE). Common causes:" -ForegroundColor Red
    Write-Host "  - 'Connection was reset' -> network/proxy. Run with -Proxy http://127.0.0.1:<port>" -ForegroundColor Red
    Write-Host "    (Clash 7890 / V2RayN 10809). Or enable your VPN's system/TUN proxy and retry." -ForegroundColor Red
    Write-Host "  - login window not completed -> rerun, finish the GitHub login popup" -ForegroundColor Red
    Write-Host "  - private repo / no popup -> use -Token <your PAT>" -ForegroundColor Red
    exit $LASTEXITCODE
}
