# Locates the Unity editor to run a batchmode command with.
#
# Unity Hub installs no shim on PATH, so every batchmode call needs a full path.
# The point of this helper is that the path is *derived* rather than written
# down: the version comes from the project itself, so upgrading the editor
# cannot leave a script pointing at the old one.
#
# Dot-source it:  . (Join-Path $PSScriptRoot 'unity-editor.ps1')

function Get-UnityProjectVersion {
    param([Parameter(Mandatory)][string]$ProjectPath)

    $versionFile = Join-Path $ProjectPath 'ProjectSettings\ProjectVersion.txt'
    if (-not (Test-Path -LiteralPath $versionFile)) {
        throw "No ProjectVersion.txt under $ProjectPath; cannot tell which editor this project wants."
    }

    # m_EditorVersion: 6000.0.40f1   (m_EditorVersionWithRevision adds the changeset,
    # which Hub does not put in the install path, so the plain field is the one to read.)
    foreach ($line in Get-Content -LiteralPath $versionFile) {
        if ($line -match '^\s*m_EditorVersion:\s*(\S+)\s*$') {
            return $Matches[1]
        }
    }
    throw "Could not read m_EditorVersion from $versionFile."
}

function Get-UnityHubInstallRoot {
    # Hub keeps a per-user override; an empty string means "use the default".
    $configured = Join-Path $env:APPDATA 'UnityHub\secondaryInstallPath.json'
    if (Test-Path -LiteralPath $configured) {
        $raw = (Get-Content -LiteralPath $configured -Raw).Trim().Trim('"')
        if ($raw) {
            return $raw
        }
    }
    return 'C:\Program Files\Unity\Hub\Editor'
}

function Resolve-UnityEditor {
    <#
    .SYNOPSIS
    Resolves the Unity executable, in order of decreasing explicitness.

    1. -Unity, when the caller passed one
    2. $env:UNITY_PATH — either the executable or the editor directory holding it
    3. the project's own editor version, under the Hub install root
    #>
    param(
        [string]$Unity,
        [Parameter(Mandatory)][string]$ProjectPath
    )

    if ($Unity) {
        if (-not (Test-Path -LiteralPath $Unity)) {
            throw "Unity executable not found at the path passed with -Unity: $Unity"
        }
        return (Resolve-Path -LiteralPath $Unity).Path
    }

    if ($env:UNITY_PATH) {
        $candidates = @(
            $env:UNITY_PATH,
            (Join-Path $env:UNITY_PATH 'Unity.exe'),
            (Join-Path $env:UNITY_PATH 'Editor\Unity.exe')
        )
        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
        throw "UNITY_PATH is set to '$($env:UNITY_PATH)' but no Unity.exe was found at it, or under Editor\ inside it."
    }

    $version = Get-UnityProjectVersion -ProjectPath $ProjectPath
    $root = Get-UnityHubInstallRoot
    $derived = Join-Path $root "$version\Editor\Unity.exe"
    if (Test-Path -LiteralPath $derived -PathType Leaf) {
        return (Resolve-Path -LiteralPath $derived).Path
    }

    # Fail with the version, because that is usually the mismatch: the editor was
    # upgraded, the project followed, and this version is simply not installed.
    throw @"
Could not locate Unity $version, which is what $ProjectPath asks for.
Looked under: $root
Install that version through Unity Hub, or point at it explicitly:
  -Unity 'C:\path\to\Unity.exe'          (this run only)
  `$env:UNITY_PATH = 'C:\path\to\editor'  (this shell)
"@
}
