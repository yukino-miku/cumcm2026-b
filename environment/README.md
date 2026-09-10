# 环境

2026-09-10 初始化时已检测：Windows / PowerShell、Git 2.45.1、GitHub CLI 2.96.0，Python Launcher 列出 Python 3.13 和 3.8。`python` 命令目前指向 Windows 应用别名，后续优先使用 `py` 或虚拟环境中的解释器。

本轮没有创建虚拟环境、安装建模依赖或确定 Python 版本。正式开始后根据实际计算需要选择解释器，将依赖及版本记录到 `requirements.txt`，环境建在项目内 `.venv/`（不提交 Git）。

版本检查：

```powershell
py -0p
git --version
gh --version
```
