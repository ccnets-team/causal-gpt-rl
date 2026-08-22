<#
.SYNOPSIS
Stage ONNX policies into the test project's performance Models folder.

.DESCRIPTION
Takes explicit "label=path" pairs so this helper carries no knowledge of where
policies live. The label becomes the staged file name and must match the label
configured on the PerformanceHarness scene component.

.EXAMPLE
./stage-performance-models.ps1 -Model 'crawler-b1-ctx32=D:\policies\crawler.onnx',
                                      'soccertwos-b16-ctx32=D:\policies\soccer.onnx'
#>
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Model,

    [string]$Destination
)

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Destination) {
    $Destination = Join-Path $root 'test-project\Assets\CGRLTests\Performance\Models'
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

foreach ($entry in $Model) {
    $separator = $entry.IndexOf('=')
    if ($separator -lt 1) {
        throw "Expected 'label=path', got '$entry'."
    }
    $label = $entry.Substring(0, $separator).Trim()
    $path = $entry.Substring($separator + 1).Trim()
    # The label becomes a file name under $Destination. Without this a label such
    # as '..\..\evil' would write outside the staging folder.
    if ($label -eq '' -or
        $label.IndexOfAny([System.IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
        $label -ne [System.IO.Path]::GetFileName($label)) {
        throw "Label must be a bare file name without path separators, got '$label'."
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Model not found for label '$label': $path"
    }
    if ([System.IO.Path]::GetExtension($path) -ne '.onnx') {
        throw "Model for label '$label' is not an .onnx file: $path"
    }
    $target = Join-Path $Destination ("{0}.onnx" -f $label)
    Copy-Item -LiteralPath (Resolve-Path -LiteralPath $path).Path -Destination $target -Force
    Write-Output "staged $label -> $target"
}
