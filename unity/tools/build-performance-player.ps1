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
$log = Join-Path $logs 'performance-build.log'
$arguments = @(
    '-batchmode',
    '-projectPath', $project,
    '-executeMethod', 'CCNets.CausalGPTRL.Editor.PerformanceBuild.BuildWindowsDevelopmentPlayer',
    '-quit',
    '-logFile', $log
)
$process = Start-Process -FilePath $Unity -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
exit $process.ExitCode
