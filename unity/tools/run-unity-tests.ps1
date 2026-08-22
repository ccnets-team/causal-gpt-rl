param(
    # Omit to derive it from the project's own editor version. See unity-editor.ps1.
    [string]$Unity
)

. (Join-Path $PSScriptRoot 'unity-editor.ps1')
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$project = Join-Path $root 'test-project'
$Unity = Resolve-UnityEditor -Unity $Unity -ProjectPath $project
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$results = Join-Path $logs 'editmode-results.xml'
$log = Join-Path $logs 'editmode.log'
Remove-Item -LiteralPath $results -Force -ErrorAction SilentlyContinue

$arguments = @(
    '-batchmode',
    '-projectPath', $project,
    '-runTests',
    '-testPlatform', 'EditMode',
    '-testResults', $results,
    '-logFile', $log
)
$process = Start-Process `
    -FilePath $Unity `
    -ArgumentList $arguments `
    -Wait `
    -PassThru `
    -WindowStyle Hidden
$unityExitCode = $process.ExitCode
if ($unityExitCode -ne 0) {
    exit $unityExitCode
}
if (-not (Test-Path -LiteralPath $results)) {
    Write-Error "Unity exited without producing test results. See $log"
    exit 1
}
