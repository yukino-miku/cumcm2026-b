#Requires -Version 7.0
[CmdletBinding()]
param(
    [string]$PythonExe,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $PythonExe) {
    $PythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw '未找到 Python 解释器，请先按 environment/README.md 配置虚拟环境，或通过 -PythonExe 指定解释器。'
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path

Push-Location -LiteralPath $projectRoot
try {
    if (-not $SkipTests) {
        Write-Output '正在验证第一问的几何算法……'
        & $PythonExe -m pytest tests/test_q1_geometry.py -q
        if ($LASTEXITCODE -ne 0) { throw '第一问测试未通过，已停止生成图表。' }
    }
    Write-Output '正在生成第一问的中文图表和构造算例记录……'
    & $PythonExe scripts/build_q1_assets.py
    if ($LASTEXITCODE -ne 0) { throw '第一问图表或实验记录生成失败。' }
    Write-Output '第一问复现完成，阅读入口：docs/第一问/阅读导航.md'
} finally {
    Pop-Location
}
