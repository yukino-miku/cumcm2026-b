#Requires -Version 7.0
[CmdletBinding()]
param([string]$PythonExe, [switch]$PlotsOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $PythonExe) { $PythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe' }
Push-Location -LiteralPath $projectRoot
try {
    if ($PlotsOnly) {
        & $PythonExe -X utf8 scripts/build_q3_scheme1_assets.py --plots-only
    } else {
        & $PythonExe -X utf8 -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw '测试未通过，停止第三问构建。' }
        & $PythonExe -X utf8 scripts/build_q3_scheme1_assets.py
    }
    if ($LASTEXITCODE -ne 0) { throw '第三问本地构建失败。' }
    & $PythonExe -X utf8 scripts/verify_q3_results.py
    if ($LASTEXITCODE -ne 0) { throw '第三问结果验收失败。' }
    Write-Output '方案一本地构建完成：docs/第三问/阅读导航.md。未调用官方模拟器。'
} finally { Pop-Location }
