#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$PythonExe,
    [switch]$SkipTests,
    [switch]$PlotsOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $PythonExe) { $PythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw '未找到 Python 解释器，请先按照 environment/README.md 配置环境。'
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
Push-Location -LiteralPath $projectRoot
try {
    if (-not $SkipTests) {
        Write-Output '正在验证第一问基线与第二问全部测试……'
        & $PythonExe -X utf8 -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw '测试未通过，停止第二问构建。' }
    }
    Write-Output '正在构建第二问中文成果……'
    if ($PlotsOnly) {
        & $PythonExe -X utf8 scripts/build_q2_assets.py --plots-only
    } else {
        & $PythonExe -X utf8 scripts/build_q2_assets.py
    }
    if ($LASTEXITCODE -ne 0) { throw '第二问计算或图表生成失败。' }
    Write-Output '正在核验第二问结果、图表及原有成果……'
    & $PythonExe -X utf8 scripts/verify_q2_results.py
    if ($LASTEXITCODE -ne 0) { throw '第二问结果验收失败。' }
    Write-Output '第二问复现完成，入口：docs/第二问/阅读导航.md'
} finally {
    Pop-Location
}
