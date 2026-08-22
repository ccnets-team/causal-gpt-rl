<#
.SYNOPSIS
Copy generated conformance fixtures into the Unity test project.

.DESCRIPTION
Unity's Asset Database only imports files inside the project, so fixtures are
generated once outside it and staged here before a test run.

The source holds one directory per model (`crawler-b1-ctx32/`, ...), each with a
policy.onnx and its *.json documents. The destination mirrors that layout so a
single test run can cover several models, and so a leftover fixture from an
earlier model is never validated against the current graph.

Both paths are required to stay inside the repository. The repository root is
found by searching upward for pyproject.toml rather than by assuming this
script's depth, so the helper keeps working after it is promoted out of the
staging area.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

function Find-RepositoryRoot {
    param([string]$Start)
    $current = (Resolve-Path -LiteralPath $Start).Path
    while ($current) {
        if (Test-Path -LiteralPath (Join-Path $current 'pyproject.toml')) {
            return $current
        }
        $parent = Split-Path -Parent $current
        if ($parent -eq $current) { break }
        $current = $parent
    }
    throw "Could not locate the repository root above $Start."
}

$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$workspacePath = Find-RepositoryRoot -Start $PSScriptRoot
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path

function Test-InsideDirectory {
    # A bare StartsWith would accept a sibling that merely shares a prefix
    # ("...\causal-gpt-rl-backup" under "...\causal-gpt-rl"). Compare against the
    # directory plus a separator, and allow the directory itself.
    param([string]$Path, [string]$Directory)
    $normalized = $Directory.TrimEnd([System.IO.Path]::DirectorySeparatorChar,
                                     [System.IO.Path]::AltDirectorySeparatorChar)
    if ($Path -eq $normalized) { return $true }
    $prefix = $normalized + [System.IO.Path]::DirectorySeparatorChar
    return $Path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

if (-not (Test-InsideDirectory -Path $sourcePath -Directory $workspacePath)) {
    throw "Refusing to stage from outside the repository: $sourcePath"
}
if (-not (Test-InsideDirectory -Path $destinationPath -Directory $workspacePath)) {
    throw "Refusing to stage outside the repository: $destinationPath"
}

$models = Get-ChildItem -LiteralPath $sourcePath -Directory
if (-not $models) {
    throw "No model directories found in $sourcePath. Generate with --out <fixtures>/<model-id>."
}

foreach ($model in $models) {
    if (-not (Test-Path -LiteralPath (Join-Path $model.FullName 'policy.onnx') -PathType Leaf)) {
        throw "Missing policy.onnx in $($model.FullName)"
    }
    if (-not (Get-ChildItem -LiteralPath $model.FullName -Filter '*.json' -File)) {
        throw "No fixture documents (*.json) found in $($model.FullName)"
    }
}

# Mirror rather than merge. A model directory the source no longer carries is
# removed along with its .meta files, which Unity regenerates on the next
# refresh. Files left directly under the destination root come from the earlier
# flat layout and are cleared the same way.
$staleRootFiles = Get-ChildItem -LiteralPath $destinationPath -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '\.(json|onnx)(\.meta)?$' }
foreach ($item in $staleRootFiles) {
    Remove-Item -LiteralPath $item.FullName -Force
    Write-Output "cleared $($item.Name)"
}

$staleModels = Get-ChildItem -LiteralPath $destinationPath -Directory -ErrorAction SilentlyContinue
foreach ($item in $staleModels) {
    Remove-Item -LiteralPath $item.FullName -Recurse -Force
    $meta = "$($item.FullName).meta"
    if (Test-Path -LiteralPath $meta) { Remove-Item -LiteralPath $meta -Force }
    Write-Output "cleared $($item.Name)/"
}

foreach ($model in $models) {
    $target = Join-Path $destinationPath $model.Name
    New-Item -ItemType Directory -Force -Path $target | Out-Null

    Copy-Item -LiteralPath (Join-Path $model.FullName 'policy.onnx') -Destination $target -Force
    Write-Output "staged $($model.Name)/policy.onnx"

    foreach ($document in Get-ChildItem -LiteralPath $model.FullName -Filter '*.json' -File) {
        Copy-Item -LiteralPath $document.FullName -Destination $target -Force
        Write-Output "staged $($model.Name)/$($document.Name)"
    }
}
