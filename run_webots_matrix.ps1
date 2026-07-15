param(
    [int]$StartAt = 1
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$webots = Join-Path $env:WEBOTS_HOME 'msys64\mingw64\bin\webots.exe'
$worlds = Get-ChildItem (Join-Path $root 'webots\worlds') -Filter '*.wbt' | Sort-Object Name
$combinations = @(
    'ekf_on_cluster_on',
    'ekf_on_cluster_off',
    'ekf_off_cluster_on',
    'ekf_off_cluster_off'
)
$total = $worlds.Count * $combinations.Count
$matrixDir = Join-Path $root ("logs\matrix_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $matrixDir | Out-Null

function Stop-ProcessTree([int]$ProcessId) {
    Get-CimInstance Win32_Process | Where-Object ParentProcessId -eq $ProcessId | ForEach-Object {
        Stop-ProcessTree $_.ProcessId
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

$case = 0
foreach ($world in $worlds) {
    foreach ($combination in $combinations) {
        $case++
        if ($case -lt $StartAt) { continue }

        $tag = '{0:D2}_{1}_{2}' -f $case, $world.BaseName, $combination
        $before = @(Get-ChildItem (Join-Path $root 'logs') -Directory | ForEach-Object FullName)
        $env:SWITCH_COMBINATION = $combination
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        Write-Output "START $case/$total world=$($world.Name) switch=$combination"

        $webotsProcess = Start-Process -FilePath $webots -ArgumentList @(
            '--mode=realtime', $world.FullName
        ) -WorkingDirectory $root -PassThru `
          -RedirectStandardOutput (Join-Path $matrixDir "$tag.webots.out.txt") `
          -RedirectStandardError (Join-Path $matrixDir "$tag.webots.err.txt")

        Start-Sleep -Seconds 5
        $webotsProcess.Refresh()
        if ($webotsProcess.HasExited) {
            throw "Webots exited before case $case started"
        }
        $env:QT_QPA_PLATFORM = 'offscreen'
        $laptopProcess = Start-Process -FilePath 'python' -ArgumentList @(
            'src/show_laptop.py', '--simulation'
        ) -WorkingDirectory $root -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput (Join-Path $matrixDir "$tag.laptop.out.txt") `
          -RedirectStandardError (Join-Path $matrixDir "$tag.laptop.err.txt")

        $finished = $laptopProcess.WaitForExit(150000)
        if (-not $finished) {
            Stop-ProcessTree $laptopProcess.Id
        }
        Stop-ProcessTree $webotsProcess.Id
        Start-Sleep -Seconds 5

        $runDir = Get-ChildItem (Join-Path $root 'logs') -Directory |
            Where-Object FullName -notin $before |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        $summaryPath = if ($runDir) { Join-Path $runDir.FullName 'run_summary.json' }
        $collisionPath = if ($runDir) { Join-Path $runDir.FullName 'webots_collision.json' }
        if ($runDir -and (Test-Path $summaryPath)) {
            $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
            [pscustomobject]@{
                case = $case
                total = $total
                world = $world.Name
                switch = $combination
                collision = [bool]$summary.collision_detected
                stop_reason = $summary.stop_reason
                mission_complete = [bool]$summary.mission_complete
                elapsed_s = $summary.elapsed_s
                run_dir = $runDir.Name
            } | ConvertTo-Json -Compress | Tee-Object -FilePath (Join-Path $matrixDir 'results.jsonl') -Append
        } else {
            $collision = if ($collisionPath -and (Test-Path $collisionPath)) {
                [bool](Get-Content $collisionPath -Raw | ConvertFrom-Json).detected
            } else {
                $null
            }
            [pscustomobject]@{
                case = $case
                total = $total
                world = $world.Name
                switch = $combination
                collision = $collision
                stop_reason = if ($finished) { 'missing_summary' } else { 'timeout' }
                mission_complete = $false
                elapsed_s = $null
                run_dir = if ($runDir) { $runDir.Name } else { $null }
            } | ConvertTo-Json -Compress | Tee-Object -FilePath (Join-Path $matrixDir 'results.jsonl') -Append
        }
    }
}
