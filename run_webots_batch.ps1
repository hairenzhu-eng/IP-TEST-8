$ErrorActionPreference = 'Stop'
$world = $args[0]
$combo = $args[1]
$webots = 'F:\webots2025a\Webots\msys64\mingw64\bin\webots.exe'
$python = 'F:\.conda\envs\py312\python.exe'
$before = @(Get-ChildItem logs -Directory -Filter 'run_*' | Select-Object -ExpandProperty FullName)
$webotsProcess = Start-Process -FilePath $webots -ArgumentList @('--mode=fast', $world) -PassThru
Start-Sleep -Seconds 8
$env:SWITCH_COMBINATION = $combo
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $PWD "logs\batch_${stamp}_stdout.txt"
$stderr = Join-Path $PWD "logs\batch_${stamp}_stderr.txt"
$pythonProcess = Start-Process -FilePath $python -ArgumentList @('src/show_laptop.py', '--simulation') -WorkingDirectory $PWD -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$deadline = (Get-Date).AddSeconds(200)
while ((Get-Date) -lt $deadline -and -not $pythonProcess.HasExited) {
    Start-Sleep -Seconds 2
    $pythonProcess.Refresh()
}
if (-not $pythonProcess.HasExited) {
    Stop-Process -Id $pythonProcess.Id -Force -ErrorAction SilentlyContinue
    $stopReason = 'timeout_200s'
} else {
    $stopReason = 'process_exit'
}
Start-Sleep -Seconds 3
if (-not $webotsProcess.HasExited) {
    Stop-Process -Id $webotsProcess.Id -Force -ErrorAction SilentlyContinue
}
$after = @(Get-ChildItem logs -Directory -Filter 'run_*' | Where-Object { $before -notcontains $_.FullName } | Sort-Object LastWriteTime -Descending)
Write-Output "STOP_REASON=$stopReason"
Write-Output "WORLD=$world"
Write-Output "COMBINATION=$combo"
Get-Content $stdout -ErrorAction SilentlyContinue | Select-Object -Last 20
Get-Content $stderr -ErrorAction SilentlyContinue | Select-Object -Last 20
foreach ($directory in $after) {
    $summary = Join-Path $directory.FullName 'run_summary.json'
    if (Test-Path $summary) { Get-Content $summary -Raw }
}
