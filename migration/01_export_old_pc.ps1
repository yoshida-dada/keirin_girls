# 01_export_old_pc.ps1 -- run on the OLD PC. Builds a migration bundle for KEIRIN (girls keirin AI).
# See MIGRATION.md. ASCII-only. Code lives in GitHub (yoshida-dada/keirin_girls, private) -- the new PC
# gets it via 'git clone' in 02_setup_new_pc.ps1. This script carries only what git cannot: the 6 SQLite
# DBs (all gitignored), .env / push subscription state, and the Startup-folder launcher stub.
#
#   .\migration\01_export_old_pc.ps1 -Dest E:\keirin_bundle
#   .\migration\01_export_old_pc.ps1 -Dest E:\keirin_bundle -DryRun
#   .\migration\01_export_old_pc.ps1 -Dest E:\keirin_bundle -SkipDb     # rehearsal without the ~1.9GB DB backup
#
# Before running: commit and push any pending repo changes (this script warns if dirty/unpushed, but
# does not push for you).
#
# Bundle layout:
#   db\        6x SQLite (consistent online backup) + per-file manifest
#   secrets\   .env / push_subs.json / notified.json          -> SENSITIVE (encrypted media, delete afterwards)
#   home\      KeirinGirlsLive.bat (the Startup-folder stub; tiny, no secrets)
#   manifest.json
param(
    [Parameter(Mandatory = $true)][string]$Dest,
    [switch]$DryRun,
    [switch]$SkipDb
)
$ErrorActionPreference = "Stop"
$Repo     = Split-Path $PSScriptRoot -Parent
$UserHome = $env:USERPROFILE
$PyBase   = "C:\Users\yoshi\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PyBase)) { $PyBase = "python" }
$StartupBat = Join-Path $UserHome "AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\KeirinGirlsLive.bat"
$env:PYTHONIOENCODING = "utf-8"

function Step($m) { Write-Host ("`n=== {0}" -f $m) -ForegroundColor Cyan }
function Do-It($desc, [scriptblock]$sb) {
    if ($DryRun) { Write-Host ("[dry-run] {0}" -f $desc) -ForegroundColor DarkGray } else { Write-Host $desc; & $sb }
}

Step "0. Preflight"
Write-Host ("repo: {0}" -f $Repo)
Write-Host ("dest: {0}" -f $Dest)
$live = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'live_scheduler\.py' }
if ($live) { Write-Host "NOTE: live_scheduler.py is running (DB may be mid-write; the backup API is still consistent):" -ForegroundColor Yellow; $live | ForEach-Object { Write-Host ("  pid {0}" -f $_.ProcessId) } }
else { Write-Host "NOTE: live_scheduler.py is not currently running (Startup auto-launch may need checking -- see MIGRATION.md)." -ForegroundColor Yellow }

Push-Location $Repo
try {
    $repoDirty = (git status --porcelain 2>$null | Out-String).Trim()
    git fetch origin -q 2>$null
    $repoUnpushed = (git log '@{u}..HEAD' --oneline 2>$null | Out-String).Trim()
    $repoUrl = (git remote get-url origin 2>$null)
} finally { Pop-Location }
if ($repoDirty) { Write-Host "WARNING: keirin_girls repo has uncommitted changes (git will not carry them):" -ForegroundColor Yellow; Write-Host $repoDirty }
if ($repoUnpushed) { Write-Host "WARNING: keirin_girls repo has commits not pushed to origin:" -ForegroundColor Yellow; Write-Host $repoUnpushed }
if (-not $repoUrl) { throw "no git remote 'origin' found in $Repo" }

Do-It "create $Dest" { New-Item -ItemType Directory -Force -Path $Dest | Out-Null }

Step "1. Repo code -- via GitHub, not copied here"
Write-Host ("repo: {0} (private). 02_setup_new_pc.ps1 does 'git clone'." -f $repoUrl)

Step "2. SQLite (6 files, consistent online backup + integrity + row counts)"
if ($SkipDb) { Write-Host "skipped (-SkipDb)" -ForegroundColor Yellow }
else {
    Do-It "db_tool.py backup x6" {
        $dbd = Join-Path $Dest "db"; New-Item -ItemType Directory -Force -Path $dbd | Out-Null
        foreach ($name in "keirin.sqlite", "keirin_men.sqlite", "keirin_men_probe.sqlite",
                           "keirin_v1_backup.sqlite", "keirin_v3.sqlite", "odds_snapshots.sqlite") {
            $src = Join-Path $Repo "data\$name"
            if (-not (Test-Path $src)) { Write-Host ("  skip (not found): {0}" -f $name) -ForegroundColor Yellow; continue }
            Write-Host ("  backing up {0} ..." -f $name)
            & $PyBase (Join-Path $PSScriptRoot "db_tool.py") backup $src (Join-Path $dbd $name) (Join-Path $dbd "$name.manifest.json")
            if ($LASTEXITCODE -ne 0) { throw ("DB backup failed integrity check: {0}" -f $name) }
        }
    }
}

Step "3. Secrets + per-device state (NOT in git; keep the bundle on encrypted media)"
Do-It "copy .env / push_subs.json / notified.json" {
    $sd = Join-Path $Dest "secrets"; New-Item -ItemType Directory -Force -Path $sd | Out-Null
    if (Test-Path (Join-Path $Repo ".env")) { Copy-Item (Join-Path $Repo ".env") $sd -Force } else { Write-Host "  missing: .env" -ForegroundColor Yellow }
    foreach ($f in "push_subs.json", "notified.json") {
        $p = Join-Path $Repo "data\$f"
        if (Test-Path $p) { Copy-Item $p $sd -Force }
    }
}

Step "4. Startup-folder launcher stub"
Do-It "copy KeirinGirlsLive.bat" {
    $hd = Join-Path $Dest "home"; New-Item -ItemType Directory -Force -Path $hd | Out-Null
    if (Test-Path $StartupBat) { Copy-Item $StartupBat $hd -Force } else { Write-Host "  missing: $StartupBat" -ForegroundColor Yellow }
}

Step "5. Manifest"
Do-It "write manifest.json" {
    $rel = $Repo.Substring($UserHome.Length).TrimStart('\')
    $files = 0; $bytes = 0
    Get-ChildItem $Dest -Recurse -File -Force | ForEach-Object { $files++; $bytes += $_.Length }
    [ordered]@{
        created = (Get-Date -Format "yyyy-MM-dd HH:mm:ss"); source_host = $env:COMPUTERNAME; source_user = $env:USERNAME
        source_userprofile = $UserHome; repo_rel = $rel; repo_git_url = $repoUrl
        repo_dirty = [bool]$repoDirty; repo_unpushed = [bool]$repoUnpushed; files = $files; bytes = $bytes
    } | ConvertTo-Json | Out-File (Join-Path $Dest "manifest.json") -Encoding UTF8
    Write-Host ("bundle: {0} files, {1:N0} MB" -f $files, ($bytes / 1MB))
}
Write-Host "`nDone. Next: copy the bundle to the new PC and run 02_setup_new_pc.ps1 -Bundle <path>." -ForegroundColor Green
Write-Host "The bundle contains secrets (.env / push subscription endpoints). Delete it after cutover." -ForegroundColor Yellow
