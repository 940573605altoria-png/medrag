# 推送到 GitHub 的便捷脚本（在你自己的 PowerShell 终端里运行）。
#
# 为什么要你自己跑：第一次推送会弹 GitHub 登录窗口（Git Credential Manager），
# 后台/沙箱进程弹不出窗口，所以必须在你的交互式终端里执行一次。
#
# 用法：
#   普通（弹窗登录）：
#     powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1
#   用个人令牌（不弹窗，token 别外发/别提交）：
#     powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1 -Token ghp_xxx
#   覆盖远程地址/分支：
#     ... -RemoteUrl https://github.com/<user>/<repo>.git -Branch main

param(
    [string]$RemoteUrl = "https://github.com/940573605altoria-png/medrag.git",
    [string]$Branch    = "main",
    [string]$Token     = ""
)

$ErrorActionPreference = "Stop"

# 控制台用 UTF-8，避免中文乱码
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 把 medrag conda 环境里的 git 加进 PATH（本机 git 装在这个环境里）
$r = "F:\miniconda\envs\medrag"
$env:PATH = "$r;$r\Library\mingw-w64\bin;$r\Library\usr\bin;$r\Library\bin;$r\Scripts;" + $env:PATH

# 切到脚本所在仓库根目录（scripts 的上一级）
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "仓库目录: $repoRoot" -ForegroundColor Cyan

# 确认是 git 仓库
git rev-parse --is-inside-work-tree > $null
if (-not $?) { throw "当前目录不是 git 仓库" }

# 确保 origin 远程存在且指向正确地址
$existing = (git remote get-url origin) 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "添加 origin -> $RemoteUrl" -ForegroundColor Yellow
    git remote add origin $RemoteUrl
} elseif ($existing -ne $RemoteUrl) {
    Write-Host "更新 origin: $existing -> $RemoteUrl" -ForegroundColor Yellow
    git remote set-url origin $RemoteUrl
}

# 展示将要推送的提交
Write-Host "`n本地 $Branch 最新提交：" -ForegroundColor Cyan
git log $Branch --oneline -n 5

# 工作区有未提交改动则提醒（不自动提交，交给你决定）
$dirty = git status --porcelain
if ($dirty) {
    Write-Host "`n⚠️ 工作区有未提交改动，本次只推已提交内容：" -ForegroundColor Yellow
    git status --short
}

Write-Host "`n开始推送（--force-with-lease：安全覆盖远程的空 Initial commit）..." -ForegroundColor Green

if ($Token) {
    # 令牌方式：把 token 临时拼进 URL 推送，不写进 remote 配置（避免落盘）
    $u = [System.Uri]$RemoteUrl
    $authUrl = "https://$Token@$($u.Host)$($u.AbsolutePath)"
    git push --force-with-lease $authUrl "${Branch}:${Branch}"
} else {
    # 弹窗登录方式
    git push --force-with-lease -u origin $Branch
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ 推送成功！远程： $RemoteUrl ($Branch)" -ForegroundColor Green
    Write-Host "AutoDL 上克隆： git clone $RemoteUrl" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ 推送失败（exit=$LASTEXITCODE）。常见原因：" -ForegroundColor Red
    Write-Host "  - 登录窗口未完成 → 重跑本脚本，完成 GitHub 登录" -ForegroundColor Red
    Write-Host "  - 私有库/无弹窗 → 用令牌： scripts\push_github.ps1 -Token <你的token>" -ForegroundColor Red
    exit $LASTEXITCODE
}
