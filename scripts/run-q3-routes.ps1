#Requires -Version 7.0
[CmdletBinding()]
param([string]$PythonExe, [switch]$PlotsOnly)
$ErrorActionPreference = 'Stop'
$routeProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $PythonExe) { $PythonExe = Join-Path $routeProjectRoot '.venv/Scripts/python.exe' }
Push-Location -LiteralPath $routeProjectRoot
try {
    if ($PlotsOnly) {
        & $PythonExe -X utf8 scripts/build_q3_route_assets.py --plots-only
    } else {
        & $PythonExe -X utf8 -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw '自动测试失败，停止构建。' }
        & $PythonExe -X utf8 scripts/build_q3_route_assets.py
    }
    if ($LASTEXITCODE -ne 0) { throw '路线调度实验或绘图失败。' }
    & $PythonExe -X utf8 scripts/verify_q3_route_results.py
    if ($LASTEXITCODE -ne 0) { throw '路线调度归档核验失败。' }
    & $PythonExe -X utf8 scripts/verify_q3_results.py
    if ($LASTEXITCODE -ne 0) { throw '方案一原有成果核验失败。' }
    & $PythonExe -X utf8 scripts/verify_q3_scheme2_results.py
    if ($LASTEXITCODE -ne 0) { throw '方案二原有成果核验失败。' }
    Write-Output '路线调度构建与原有成果核验完成；未调用官方模拟器。'
} finally { Pop-Location }
