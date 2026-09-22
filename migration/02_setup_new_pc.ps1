# 02_setup_new_pc.ps1 -- run on the NEW PC (normal user PowerShell, not elevated; UAC prompts from winget
# are expected). Restores a bundle made by 01_export_old_pc.ps1 and builds the whole runtime. ASCII-only.
# Safe to re-run. If the banei-keiba PC setup already ran on this PC, winget/gh steps are skipped
# automatically (already-installed / already-authenticated checks).
#
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\02_setup_new_pc.ps1 -Bundle E:\keirin_bundle
#
# Best results: same Windows user name (C:\Users\yoshi) and the same project path as the old PC
# (scripts\start_live_scheduler.bat falls back to a hard-coded path if the new .venv is missing --
# see MIGRATION.md).
param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [switch]$SkipInstall,
    [switch]$SkipPython,
    [switch]$SkipVerify
)
$ErrorActionPreference = "Stop"
$UserHome = $env:USERPROFILE
$env:PYTHONIOENCODING = "utf-8"
function Step($m) { Write-Host ("`n=== {0}" -f $m) -ForegroundColor Cyan }
function Warn($m) { Write-Host ("WARN: {0}" -f $m) -ForegroundColor Yellow }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") }

Step "0. Preflight + manifest"
$mf = Get-Content (Join-Path $Bundle "manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$Root = Join-Path $UserHome $mf.repo_rel
Write-Host ("bundle from : {0}\{1} ({2})" -f $mf.source_host, $mf.source_user, $mf.created)
Write-Host ("project root: {0}" -f $Root)
if (-not $mf.repo_git_url) { throw "manifest.json has no repo_git_url" }
if ($mf.repo_dirty)    { Warn "keirin_girls had uncommitted changes on the old PC -- they are NOT in this clone" }
if ($mf.repo_unpushed) { Warn "keirin_girls had commits not pushed to origin on the old PC -- they are NOT in this clone" }

Step "1. Applications (winget) -- shared with the banei-keiba kit, skipped if already installed"
if (-not $SkipInstall) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "winget not found. Update 'App Installer' from the Microsoft Store." }
    foreach ($id in "Git.Git", "GitHub.cli", "Python.Python.3.10") {
        winget list --id $id -e --accept-source-agreements 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host ("already installed: {0}" -f $id); continue }
        Write-Host ("installing {0}" -f $id)
        winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { Warn ("winget install {0} exit {1}" -f $id, $LASTEXITCODE) }
    }
    Refresh-Path
}

Step "2. GitHub auth (skipped if already logged in, e.g. from setting up banei-keiba first)"
Refresh-Path
if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh auth status 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "GitHub login required (interactive: browser + one-time code)"; gh auth login --web --git-protocol https }
    gh auth setup-git 2>&1 | Out-Null
} else { Warn "gh not found -- 'git clone' will prompt for credentials interactively" }

Step "3. Restore files (repo via 'git clone', DB/secrets/Startup stub from the bundle)"
if (Test-Path (Join-Path $Root ".git")) {
    Write-Host "repo already present -- pulling latest instead of cloning"
    Push-Location $Root; try { git pull --ff-only } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "git pull failed in $Root" }
} elseif (Test-Path $Root) {
    throw "$Root exists but is not a git repo. Move it aside or delete it, then re-run."
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $Root) | Out-Null
    git clone $mf.repo_git_url $Root
    if ($LASTEXITCODE -ne 0) { throw "git clone failed ($($mf.repo_git_url)). Check 'gh auth status' above." }
}
Write-Host "repo cloned/updated"
$dbSrc = Join-Path $Bundle "db"
if (Test-Path $dbSrc) {
    $dataDir = Join-Path $Root "data"; New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    Get-ChildItem $dbSrc -Filter *.sqlite | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $dataDir $_.Name) -Force
        Write-Host ("  restored {0}" -f $_.Name)
    }
} else { Warn "no db\ in the bundle (rehearsal bundle?)" }
$sd = Join-Path $Bundle "secrets"
if (Test-Path (Join-Path $sd ".env")) { Copy-Item (Join-Path $sd ".env") $Root -Force } else { Warn "secret missing in bundle: .env" }
foreach ($f in "push_subs.json", "notified.json") {
    $p = Join-Path $sd $f
    if (Test-Path $p) { Copy-Item $p (Join-Path $Root "data\$f") -Force }
}
$homeBat = Join-Path $Bundle "home\KeirinGirlsLive.bat"
$startupDir = Join-Path $UserHome "AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
if (Test-Path $homeBat) { Copy-Item $homeBat $startupDir -Force; Write-Host "Startup launcher installed (KeirinGirlsLive.bat)" }
else { Warn "no Startup launcher stub in the bundle -- create $startupDir\KeirinGirlsLive.bat manually (see MIGRATION.md)" }

Step "4. Python: dedicated venv (KEIRIN\.venv), matching the old PC's base-Python packages"
if ($SkipPython) { Write-Host "skipped" }
else {
    Refresh-Path
    $py310 = (& py -3.10 -c "import sys;print(sys.executable)" 2>$null)
    if (-not $py310) { throw "Python 3.10 not found (py -3.10). Install Python.Python.3.10 and re-run." }
    $venvDir = Join-Path $Root ".venv"
    if (-not (Test-Path (Join-Path $venvDir "Scripts\python.exe"))) { & $py310 -m venv $venvDir }
    $vpy = Join-Path $venvDir "Scripts\python.exe"
    & $vpy -m pip install --upgrade pip
    & $vpy -m pip install -r (Join-Path $Root "migration\requirements.lock.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    & $vpy -m pip check
}

if (-not $SkipVerify) { Step "5. Verify"; & (Join-Path $Root "migration\04_verify.ps1") -Root $Root }

Write-Host "`n=== Manual steps that cannot be automated (details: MIGRATION.md) ===" -ForegroundColor Green
Write-Host " 1. Log on (or run the Startup shortcut manually) and confirm KeirinGirlsLive.bat launches live_scheduler.py."
Write-Host " 2. Confirm the GitHub Pages dashboard auto-deploys via this repo's Actions workflow."
Write-Host " 3. Cutover day: remove KeirinGirlsLive.bat from the OLD PC's Startup folder before enabling it on the new PC."
