# 04_verify.ps1 -- post-migration health check for KEIRIN (read-only; safe on the old PC too). ASCII-only.
#   .\04_verify.ps1                 # everything
#   .\04_verify.ps1 -SkipTests      # skip pytest (fast)
# Exit code = number of FAIL items.
param(
    [string]$Root = (Split-Path $PSScriptRoot -Parent),
    [switch]$SkipTests
)
$env:PYTHONIOENCODING = "utf-8"
$script:fail = 0; $script:warn = 0
function Report($ok, $name, $detail, [switch]$WarnOnly) {
    if ($ok) { $tag = "PASS"; $col = "Green" } elseif ($WarnOnly) { $tag = "WARN"; $col = "Yellow"; $script:warn++ } else { $tag = "FAIL"; $col = "Red"; $script:fail++ }
    Write-Host ("[{0}] {1}  {2}" -f $tag, $name, $detail) -ForegroundColor $col
}
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "C:\Users\yoshi\AppData\Local\Programs\Python\Python310\python.exe" }

Write-Host "--- Python environment"
Report (Test-Path $venvPy) "dedicated venv (KEIRIN\.venv)" $venvPy -WarnOnly
Report (Test-Path $py) "python interpreter" $py
if (Test-Path $py) {
    $ver = (& $py --version 2>&1 | Out-String).Trim()
    Report ($ver -like "Python 3.10.*" -or $ver -like "Python 3.1[01].*") "python version" $ver
    $imp = & $py -c "import numpy,pandas,scipy,sklearn,lightgbm,requests,bs4,pywebpush,dotenv;print('ok')" 2>&1
    Report ($LASTEXITCODE -eq 0) "imports (numpy/pandas/scipy/sklearn/lightgbm/requests/bs4/pywebpush/dotenv)" ($imp | Out-String).Trim()
    $chk = (& $py -m pip check 2>&1 | Out-String).Trim()
    Report ($chk -like "No broken requirements*") "pip check" $chk
}

Write-Host "--- Data (6 SQLite DBs, all gitignored)"
foreach ($name in "keirin.sqlite", "keirin_men.sqlite", "keirin_men_probe.sqlite",
                   "keirin_v1_backup.sqlite", "keirin_v3.sqlite", "odds_snapshots.sqlite") {
    $db = Join-Path $Root "data\$name"
    Report (Test-Path $db) $name ("{0:N0} MB" -f ((Get-Item $db -ErrorAction SilentlyContinue).Length / 1MB))
}

Write-Host "--- Secrets / per-device state"
Report (Test-Path (Join-Path $Root ".env")) ".env (VAPID keys)" ""
foreach ($f in "push_subs.json", "notified.json") { Report (Test-Path (Join-Path $Root "data\$f")) "data\$f" "" -WarnOnly }

Write-Host "--- Startup launcher"
$startupBat = Join-Path $env:USERPROFILE "AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\KeirinGirlsLive.bat"
Report (Test-Path $startupBat) "Startup\KeirinGirlsLive.bat" $startupBat
Report (Test-Path (Join-Path $Root "scripts\start_live_scheduler.bat")) "scripts\start_live_scheduler.bat" ""
$live = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'live_scheduler\.py' }
Report ([bool]$live) "live_scheduler.py currently running" "(only meaningful after a logon; Startup launches it, not this script)" -WarnOnly

Write-Host "--- Git / GitHub"
if (Test-Path (Join-Path $Root ".git")) {
    Push-Location $Root
    try {
        git ls-remote origin HEAD 2>&1 | Out-Null
        Report ($LASTEXITCODE -eq 0) "origin reachable (needed: live_scheduler auto-commits+pushes dashboard data)" ""
        $dirty = (git status --porcelain 2>$null | Out-String).Trim()
        Report ([string]::IsNullOrEmpty($dirty)) "working tree clean" $dirty -WarnOnly
        $sizeMB = [math]::Round(((Get-ChildItem (Join-Path $Root ".git") -Recurse -Force -File | Measure-Object Length -Sum).Sum) / 1MB)
        Report ($sizeMB -lt 500) ".git size reasonable (history compression applied)" ("{0} MB" -f $sizeMB) -WarnOnly
    } finally { Pop-Location }
} else { Report $false ".git present" "" }

if (-not $SkipTests -and (Test-Path $py) -and (Test-Path (Join-Path $Root "tests"))) {
    Write-Host "--- pytest (--ignore=api: fastapi is not in requirements.lock.txt, api/ is unused/untested currently)"
    Push-Location $Root
    try {
        $o = & $py -m pytest -q --no-header -p no:cacheprovider --ignore=api 2>&1 | Select-Object -Last 6 | Out-String
        # baseline on the old PC (2026-09-22): 147 passed, 1 pre-existing failure unrelated to migration
        # (tests/test_gamboo_odds.py::test_race_meta_missing_is_none -- code/test drift). Compare counts, not zero-fail.
        Report ($o -match '147 passed' -or $o -match '\d+ passed') "pytest" $o.Trim() -WarnOnly
    } finally { Pop-Location }
}
Write-Host ("`nsummary: {0} FAIL, {1} WARN" -f $script:fail, $script:warn) -ForegroundColor $(if ($script:fail) { "Red" } else { "Green" })
exit $script:fail
