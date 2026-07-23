$ErrorActionPreference = 'Continue'
$webots = 'F:\webots2025a\Webots\msys64\mingw64\bin\webots.exe'
$python = 'F:\.conda\envs\py312\python.exe'
$worlds = @(Get-ChildItem 'webots\worlds\*.wbt' | Sort-Object Name)
$combos = @('ekf_on_cluster_on', 'ekf_on_cluster_off', 'ekf_off_cluster_on', 'ekf_off_cluster_off')
$report = Join-Path $PWD 'logs\webots_batch_report.csv'
if (-not (Test-Path $report)) { 'world,combination,run_dir,stop_reason,collision,elapsed_s' | Set-Content $report -Encoding utf8 }
foreach ($worldFile in $worlds) {
    foreach ($combo in $combos) {
        $existing = @(Get-ChildItem logs -Directory -Filter 'run_*' | ForEach-Object {
            $summary = Join-Path $_.FullName 'run_summary.json'
            if (Test-Path $summary) {
                try { Get-Content $summary -Raw | ConvertFrom-Json } catch {}
            }
        } | Where-Object { $_.webots_environment -eq $worldFile.Name -and $_.switch_combination -eq $combo })
        if ($existing.Count -gt 0) {
            Write-Output "SKIP world=$($worldFile.Name) combo=$combo existing=$($existing.Count)"
            continue
        }
        Write-Output "START world=$($worldFile.Name) combo=$combo"
        $beforeDirs = @(Get-ChildItem logs -Directory -Filter 'run_*' | Select-Object -ExpandProperty FullName)
        $beforePython = @(Get-Process python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
        $beforeWebots = @(Get-Process webots,webots-bin -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
        $webotsProcess = Start-Process -FilePath $webots -ArgumentList @($worldFile.FullName) -PassThru
        Start-Sleep -Seconds 8
        $env:SWITCH_COMBINATION = $combo
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $stdout = Join-Path $PWD "logs\batch_${stamp}_stdout.txt"
        $stderr = Join-Path $PWD "logs\batch_${stamp}_stderr.txt"
        $launcher = Start-Process -FilePath $python -ArgumentList @('src/show_laptop.py', '--simulation') -WorkingDirectory $PWD -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $deadline = (Get-Date).AddSeconds(200)
        $newDir = $null
        $summaryObject = $null
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            $newDir = @(Get-ChildItem logs -Directory -Filter 'run_*' | Where-Object { $beforeDirs -notcontains $_.FullName } | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
            if ($newDir) {
                $summaryPath = Join-Path $newDir.FullName 'run_summary.json'
                if (Test-Path $summaryPath) {
                    try { $summaryObject = Get-Content $summaryPath -Raw | ConvertFrom-Json } catch {}
                    if ($summaryObject) { break }
                }
            }
        }
        $stopReason = if ($summaryObject) { $summaryObject.stop_reason } else { 'timeout_200s' }
        if (-not $summaryObject) { Stop-Process -Id $launcher.Id -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
        $newPython = @(Get-Process python -ErrorAction SilentlyContinue | Where-Object { $beforePython -notcontains $_.Id } | Select-Object -ExpandProperty Id)
        foreach ($id in $newPython) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
        $newWebots = @(Get-Process webots,webots-bin -ErrorAction SilentlyContinue | Where-Object { $beforeWebots -notcontains $_.Id } | Select-Object -ExpandProperty Id)
        foreach ($id in $newWebots) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
        if ($summaryObject) {
            $line = '{0},{1},{2},{3},{4},{5}' -f $summaryObject.webots_environment,$summaryObject.switch_combination,$newDir.Name,$summaryObject.stop_reason,$summaryObject.collision_detected,$summaryObject.elapsed_s
            Add-Content $report $line -Encoding utf8
            Write-Output "DONE world=$($summaryObject.webots_environment) combo=$($summaryObject.switch_combination) collision=$($summaryObject.collision_detected) stop=$($summaryObject.stop_reason) elapsed=$($summaryObject.elapsed_s)"
        } else {
            Add-Content $report "$($worldFile.Name),$combo,,timeout_200s,unknown," -Encoding utf8
            Write-Output "DONE world=$($worldFile.Name) combo=$combo collision=unknown stop=timeout_200s"
        }
    }
}
Write-Output "BATCH_COMPLETE report=$report"
