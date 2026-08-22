$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$player = Join-Path $root 'builds\performance\CGRLPerformance.exe'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$results = Join-Path $logs 'performance-results.json'
$log = Join-Path $logs 'performance-player.log'
Remove-Item -LiteralPath $results -Force -ErrorAction SilentlyContinue
$arguments = @(
    '-batchmode',
    '-logFile', $log,
    '--cgrl-performance-results', $results
)
$process = Start-Process `
    -FilePath $player `
    -ArgumentList $arguments `
    -WorkingDirectory (Split-Path -Parent $player) `
    -Wait `
    -PassThru `
    -WindowStyle Hidden
if ($process.ExitCode -ne 0) {
    exit $process.ExitCode
}
if (-not (Test-Path -LiteralPath $results)) {
    Write-Error "Performance Player exited without results. See $log"
    exit 1
}
