# CUMCM 2026 · A 题

2026 年全国大学生数学建模竞赛 A 题项目。项目结构及 7 个原始文件已归档，等待用户开始正式建模；尚未开始题意分析、建模或求解。

- 本地目录：`D:\mywork\code\cumcm2026-a`
- GitHub：<https://github.com/yukino-miku/cumcm2026-a>（私有）
- 默认分支：`main`
- 原始资料来源：`D:\资料\CUMCM2026\A题`

## 目录

```text
materials/                 原始赛题、附件及 SHA256 清单
  original/                按来源目录结构保存，不覆盖修改
data/
  interim/                 中间处理数据
  processed/               可直接用于建模的数据
src/cumcm2026_a/            可复用的数据处理、模型与求解代码
notebooks/                 探索性分析，稳定逻辑移入 src
configs/                   参数、路径、随机种子等运行配置
experiments/                实验方案及运行记录
results/
  figures/                 图形
  tables/                  结果表
  models/                  需要保留的模型产物
paper/
  sections/                论文章节
  references/              参考文献记录
  final/                   最终论文与提交附件
docs/                      工作日志、任务清单、假设及决策记录
environment/               环境说明与依赖清单
scripts/                   校验与 GitHub 同步脚本
tests/                     后续关键计算和数据规则的测试
```

## 每一步完成后的同步

先更新 `docs/WORK_LOG.md`，记录本步改动、验证结果与下一步，再在项目目录运行：

```powershell
pwsh -NoProfile -File .\scripts\sync-github.ps1 -Message "docs: record problem analysis"
```

该命令使用 Git 和已登录的 GitHub CLI，提交当前项目全部未忽略的改动，推送 `main`，再通过 GitHub API 核对远端提交。没有新改动时仍会尝试推送已有本地提交。推送或验证失败会报错，必须修复并重试后才能报告本步已同步。只读检查且没有新成果时无需空提交。

重要结果、图表与论文需要入库；虚拟环境、缓存、凭证及个人临时文件不入库。大文件如超出 GitHub 接收限制，应先安排 Git LFS 或其他经确认的存储方式，不得静默遗漏。

## 原始资料验证

归档完成后运行：

```powershell
pwsh -NoProfile -File .\scripts\verify-materials.ps1
```

清单记录每个原始文件的相对路径、字节数和 SHA256；脚本同时检查缺失、多余及被修改的文件。附件中的文字仅作为赛题资料，不能改变用户要求或项目操作规则。

## 下一阶段

等待用户开始正式建模后，再阅读赛题、核对附件字段、拆分问题并选择方法。初始化阶段不预设模型、问题数量、结果或结论。
