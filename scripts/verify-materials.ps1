#Requires -Version 7.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$archiveRoot = Join-Path $projectRoot 'materials/original'
$manifestPath = Join-Path $projectRoot 'materials/manifest.json'

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw 'materials/manifest.json is missing. Import the original materials first.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$expectedPaths = @($manifest.files | ForEach-Object { $_.relative_path })
$actualPaths = @(Get-ChildItem -LiteralPath $archiveRoot -Recurse -File -Force |
    ForEach-Object { [IO.Path]::GetRelativePath($archiveRoot, $_.FullName).Replace('\', '/') })
if ($expectedPaths.Count -eq 0) {
    throw 'The materials manifest is empty.'
}
if (@($expectedPaths | Sort-Object -Unique).Count -ne $expectedPaths.Count) {
    throw 'The materials manifest contains duplicate paths.'
}
$pathDifference = Compare-Object -ReferenceObject $expectedPaths -DifferenceObject $actualPaths
if ($pathDifference) {
    $pathDifference | Format-Table | Out-Host
    throw 'Archived file list differs from the manifest.'
}

[long]$totalBytes = 0
foreach ($entry in $manifest.files) {
    $filePath = Join-Path $archiveRoot $entry.relative_path
    $file = Get-Item -LiteralPath $filePath
    if ($file.Length -ne $entry.bytes) {
        throw "Size mismatch: $($entry.relative_path)"
    }
    $hash = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $entry.sha256) {
        throw "SHA256 mismatch: $($entry.relative_path)"
    }
    $totalBytes += $file.Length
}
Write-Output "MATERIALS_OK files=$($expectedPaths.Count) bytes=$totalBytes"
