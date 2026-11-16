# Waits for RML2016.10a to arrive, then runs the real experiment automatically.
#
#   1. Watches the Downloads folder for RML2016.10a*.tar.bz2 (or data\RML2016.10a_dict.pkl).
#   2. Checks the archive only contains the expected file, then extracts it into data\.
#   3. Stops the Plan-B run on generated data, if one is running.
#   4. Runs scripts\run_all.py --dataset rml2016a  (log: results\rml2016a_run.log)
#
# Start it hidden so it survives closing the terminal:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy Bypass -File scripts\wait_for_dataset_and_run.ps1'

$ErrorActionPreference = "Stop"
$proj      = Split-Path -Parent $PSScriptRoot
$downloads = Join-Path $env:USERPROFILE "Downloads"
$dataDir   = Join-Path $proj "data"
$pkl       = Join-Path $dataDir "RML2016.10a_dict.pkl"
$resDir    = Join-Path $proj "results"
$status    = Join-Path $resDir "auto_run_status.txt"
New-Item -ItemType Directory -Force $dataDir, $resDir | Out-Null

function Say($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content $status; }

Say "Waiting for RML2016.10a in $downloads ..."
while (-not (Test-Path $pkl)) {
    $f = Get-ChildItem $downloads -Filter "RML2016*.bz2" -ErrorAction SilentlyContinue |
         Where-Object { $_.Length -gt 100MB } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($f) {
        $size = $f.Length; Start-Sleep 10; $f.Refresh()
        if ($f.Length -eq $size) {
            Say "Found $($f.FullName) ($([math]::Round($size / 1MB)) MB). Checking contents..."
            $entries = & "$env:SystemRoot\System32\tar.exe" -tjf $f.FullName
            $bad = $entries | Where-Object { $_ -match '\.\.|^[/\\]|:' }
            $member = $entries | Where-Object { $_ -match 'RML2016\.10a_dict\.pkl$' } | Select-Object -First 1
            if ($bad -or -not $member) {
                Say "ERROR: unexpected archive contents: $($entries -join ', '). Not extracting."
                exit 1
            }
            & "$env:SystemRoot\System32\tar.exe" -xjf $f.FullName -C $dataDir $member
            $extracted = Join-Path $dataDir $member
            if ($extracted -ne $pkl) { Move-Item -Force $extracted $pkl }
            Say "Extracted to $pkl"
        }
    }
    if (-not (Test-Path $pkl)) { Start-Sleep 20 }
}

# Stop the Plan-B run on generated data so the real run gets the whole CPU.
$pidFile = Join-Path $resDir "synthetic_run.pid"
if (Test-Path $pidFile) {
    $planB = Get-Content $pidFile
    & taskkill /PID $planB /T /F 2>$null | Out-Null
    Remove-Item $pidFile
    Say "Stopped the Plan-B run (generated data), PID $planB."
}

Say "Starting the real experiment (log: results\rml2016a_run.log)..."
$env:PYTHONUNBUFFERED = "1"
$py = Join-Path $proj ".venv\Scripts\python.exe"
& cmd /c "`"$py`" `"$proj\scripts\run_all.py`" --dataset rml2016a > `"$resDir\rml2016a_run.log`" 2>&1"
if ($LASTEXITCODE -eq 0) { Say "DONE. Open results\rml2016a\REPORT.md" }
else { Say "FAILED (exit $LASTEXITCODE). See results\rml2016a_run.log" }
