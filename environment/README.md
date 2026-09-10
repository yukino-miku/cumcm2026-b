# 环境

2026-09-10 初始化时已检测：Windows / PowerShell、Git 2.45.1、GitHub CLI 2.96.0，Python Launcher 列出 Python 3.13 和 3.8。`python` 命令目前指向 Windows 应用别名，后续优先使用 `py` 或虚拟环境中的解释器。

## 第一问环境

第一问已经建立项目专用 `.venv/`（不提交 Git），解释器为 Python 3.12.14，由本机 Codex 捆绑解释器创建。实际使用 NumPy 2.3.5、SciPy 1.18.1、Matplotlib 3.11.1、pytest 9.1.1；所有直接及传递依赖的已安装版本保存在 `requirements.txt`。

本机直接运行：

```powershell
pwsh -NoProfile -File .\scripts\run-q1.ps1
```

其他机器安装 Python 3.12 后，在项目根目录建立虚拟环境并安装锁定依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment/requirements.txt
pwsh -NoProfile -File .\scripts\run-q1.ps1
```

Linux/macOS 可用 `python3.12 -m venv .venv` 建立环境，依次执行 `.venv/bin/python -m pytest -q` 和 `.venv/bin/python scripts/build_q1_assets.py`。如果使用其他 Python 版本，需要重新通过测试，不能直接声称与记录环境完全一致。

中文图表使用本机微软雅黑字体；其他机器的字体可用性会影响绘图，应按图表脚本的提示安装中文字体。提交的 SVG 将文字保存为图形路径，查看时不依赖本机字体。

切换 B 题时已检测 Git LFS 3.5.1，在本仓库使用 `git lfs install --local` 安装过滤器与推送钩子，以保存模拟器压缩包和演示视频。新克隆需要 `git lfs pull` 下载实际对象，之后才能进行 SHA256 校验。

## 本机 Git 网络

初始化时 Git 直连 `github.com:443` 出现连接失败，Windows 当前系统代理为 `127.0.0.1:10808`。已验证该代理可访问远端，并仅在本仓库 `.git/config` 设置 `http.proxy=http://127.0.0.1:10808`；未修改 Git 全局配置或系统代理。该设置不随克隆传递。后续代理端口或运行状态改变时，应按本机实际配置调整，避免直接在其他机器照搬。

版本检查：

```powershell
py -0p
git --version
gh --version
```
