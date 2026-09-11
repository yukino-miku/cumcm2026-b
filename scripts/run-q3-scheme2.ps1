#Requires -Version 7.0
[CmdletBinding()]
param([string]$PythonExe, [switch]$PlotsOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $PythonExe) { $PythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe' }
Push-Location -LiteralPath $projectRoot
try {
    if ($PlotsOnly) {
        & $PythonExe -X utf8 scripts/build_q3_scheme2_assets.py --plots-only
    } else {
        & $PythonExe -X utf8 -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw '自动测试失败，停止方案二构建。' }
        & $PythonExe -X utf8 scripts/build_q3_scheme2_assets.py
    }
    if ($LASTEXITCODE -ne 0) { throw '方案二本地构建失败。' }
    & $PythonExe -X utf8 scripts/verify_q3_scheme2_results.py
    if ($LASTEXITCODE -ne 0) { throw '方案二验收失败。' }
    & $PythonExe -X utf8 scripts/verify_q3_results.py
    if ($LASTEXITCODE -ne 0) { throw '方案一原有成果核验失败。' }
    Write-Output '方案二构建与两方案验收完成；未调用官方模拟器。'
} finally { Pop-Location }
