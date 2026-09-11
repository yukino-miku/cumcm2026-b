#!/usr/bin/env python
"""绘制已完成的12局实际演练轨迹；默认可从脱敏轨迹数据重绘，不连接模拟器。"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_q3_scheme1_assets import style, save
from cumcm2026_b.q3_geometry import search_stations
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

EVALUATION = ROOT / "results/tables/第三问/两组演练评估_s1_s2.json"
DATA = ROOT / "results/tables/第三问/十二局实际演练轨迹.json"
FIGDIR = ROOT / "results/figures/第三问"
GUIDE = ROOT / "docs/第三问/十二局实际演练轨迹.md"
COLORS = {"前往固定站": "#327DA2", "前往追加测向点": "#D18330", "前往清除点": "#25957E"}
OVERVIEWS = {"s1": "十二局演练_方案一轨迹总览", "s2": "十二局演练_方案二轨迹总览"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def extract():
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    sources = {EVALUATION.relative_to(ROOT).as_posix(): digest(EVALUATION)}
    runs = []
    for row in evaluation["逐局记录"]:
        path = ROOT / row["关联本地目录"] / "运行结果.json"
        key = path.relative_to(ROOT).as_posix()
        assert digest(path) == evaluation["来源SHA256"][key], key
        sources[key] = digest(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        actions = [e for e in result["动作记录"] if e["动作"] in ("检测", "清除")]
        assert result["正常退出"] and result["运行成功"] and row["全清核对通过"]
        assert len(actions) == result["检测次数"] + result["清除成功次数"] + result["清除失败次数"]
        points = np.asarray([[0., 0.]] + [e["位置"] for e in actions])
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        assert abs(length - result["总路程_米"]) < 1e-5
        scans = (search_stations() if row["组"] == "s1" else np.asarray(result["布局"]["站点"]))
        visits = [i for i, p in enumerate(scans) if any(e["动作"] == "检测" and
                  e["用途"].startswith(("搜索站", "固定扫描站")) and
                  np.linalg.norm(p - e["位置"]) < 1e-6 for e in actions)]
        assert visits == result["已完成搜索站"]
        filtered = [{"步骤": e["步骤"], "虚拟时间_秒": e["虚拟时间_秒"], "动作": e["动作"],
                     "用途": e["用途"], "频道": e["频道"], "位置": e["位置"], "结果": e["结果"]}
                    for e in actions]
        runs.append({"编号": row["显示编号"], "组": row["组"], "案例码": row["案例码"],
                     "目标数": row["官方结果文件目标总数"], "全清": row["全清核对通过"],
                     "总虚拟时间_秒": result["虚拟总时间_秒"], "总路程_米": length,
                     "检测次数": result["检测次数"], "清除成功次数": result["清除成功次数"],
                     "清除失败次数": result["清除失败次数"], "固定站点": scans.tolist(),
                     "已完成固定扫描站序号": visits, "终止依据": result["终止依据"], "实际动作": filtered})
    assert len(runs) == 12 and sum(r["目标数"] for r in runs) == 150
    for path in [Path(__file__), ROOT / "scripts/build_q3_scheme1_assets.py",
                 ROOT / "src/cumcm2026_b/q3_geometry.py"]:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    data = {"说明": "用户指定s1/s2各6局；动作坐标来自此前已关联核验的本地HTTP运行结果。未读取目标精确真值。",
            "连线口径": "从原点开始按检测/清除执行顺序连接坐标，依据接口直线移动计费；不闭合路线，不人为连接未执行的固定扫描路线。",
            "标记口径": "成功清除位置是机器狗执行清除的位置，不是目标精确坐标；区域边界不限制机器狗行走。",
            "来源SHA256": sources, "逐局轨迹": runs}
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    return data


def legend():
    handles = [Line2D([], [], color=c, lw=2, label=k) for k, c in COLORS.items()]
    handles += [
        Line2D([], [], color="#765D91", ls="--", label="1800米目标区域边界"),
        Line2D([], [], marker="s", color="#263E50", ls="", label="完成固定扫描的站点"),
        Line2D([], [], marker="s", markerfacecolor="white", color="#A5ADB4", ls="", label="其余预设固定站"),
        Line2D([], [], marker="^", color=COLORS["前往追加测向点"], ls="", label="追加测向点"),
        Line2D([], [], marker="o", markerfacecolor="white", color=COLORS["前往清除点"], ls="", label="成功清除位置"),
        Line2D([], [], marker="x", color="#C74849", ls="", label="失败清除位置"),
        Line2D([], [], marker="*", color="#1D2939", ls="", markersize=10, label="起点（原点）"),
        Line2D([], [], marker="D", color="#914E86", ls="", label="终点"),
    ]
    return handles


def draw(ax, run, detail=False):
    actions = run["实际动作"]
    previous = np.zeros(2)
    segments, colors = [], []
    for event in actions:
        current = np.asarray(event["位置"])
        length = np.linalg.norm(current - previous)
        if length > 1e-7:
            if event["动作"] == "清除":
                category = "前往清除点"
            elif event["用途"] == "追加测向":
                category = "前往追加测向点"
            else:
                assert event["用途"].startswith(("搜索站", "固定扫描站")), event["用途"]
                category = "前往固定站"
            color = COLORS[category]
            segments.append([previous.copy(), current.copy()]); colors.append(color)
            if length >= 160:
                center = previous + .57 * (current - previous)
                arrow = min(85., length * .17) * (current - previous) / length
                ax.annotate("", xy=center + arrow / 2, xytext=center - arrow / 2,
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": .7,
                                        "mutation_scale": 8 if detail else 6}, zorder=3)
        previous = current
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.3 if detail else 1.05,
                                     alpha=.8, zorder=2))
    for i, p in enumerate(run["固定站点"]):
        visited = i in run["已完成固定扫描站序号"]
        ax.scatter(*p, marker="s", s=28 if detail else 22, zorder=4,
                   facecolors="#263E50" if visited else "white",
                   edgecolors="#263E50" if visited else "#A5ADB4", linewidths=1)
        if detail and i:
            ax.annotate(f"站{i}", p, xytext=(5, 5), textcoords="offset points", fontsize=8,
                        color="#263E50" if visited else "#8C969F", zorder=8)
    for event in actions:
        if event["用途"] == "追加测向":
            ax.scatter(*event["位置"], marker="^", s=30 if detail else 22,
                       color=COLORS["前往追加测向点"], edgecolors="white", linewidths=.35, zorder=5)
        elif event["动作"] == "清除":
            success = event["结果"] == "success"
            if success:
                ax.scatter(*event["位置"], s=40 if detail else 28, facecolors="white",
                           edgecolors=COLORS["前往清除点"], linewidths=1.35, zorder=6)
            else:
                ax.scatter(*event["位置"], marker="x", s=45 if detail else 35,
                           color="#C74849", linewidths=1.4, zorder=7)
    ax.scatter(0, 0, marker="*", s=120 if detail else 88, c="#1D2939",
               edgecolors="white", linewidths=.6, zorder=9)
    ax.scatter(*previous, marker="D", s=52 if detail else 35, c="#914E86",
               edgecolors="white", linewidths=.5, zorder=9)
    if detail:
        ax.annotate("起点 / 站0", (0, 0), xytext=(8, -16), textcoords="offset points", fontsize=9)
        ax.annotate("终点", previous, xytext=(7, 7), textcoords="offset points", fontsize=9, color="#914E86")
    ax.add_patch(Circle((0, 0), 1800, fill=False, ls="--", color="#765D91", lw=1.15, zorder=1))
    ax.set(xlim=(-2050, 2050), ylim=(-2050, 2050), aspect="equal",
           xticks=[-2000, -1000, 0, 1000, 2000], yticks=[-2000, -1000, 0, 1000, 2000],
           xlabel="东向 x（米）", ylabel="北向 y（米）")
    ax.tick_params(labelsize=9 if detail else 8)
    ax.grid(color="#DCE3E8", lw=.55, alpha=.65)
    for spine in ax.spines.values(): spine.set_color("#CBD5DE")
    n = len(run["已完成固定扫描站序号"])
    ax.set_title(f"{run['编号']}  ·  {run['目标数']}个目标全清  ·  固定扫描{n}/{len(run['固定站点'])}站\n"
                 f"路程 {run['总路程_米']/1000:.2f} 公里  |  虚拟时间 {run['总虚拟时间_秒']:.1f} 秒",
                 fontsize=12 if detail else 11, pad=12)


def plot(data):
    style()
    plt.rcParams["svg.hashsalt"] = "q3-practice-trajectories"
    (FIGDIR / "实际演练轨迹").mkdir(parents=True, exist_ok=True)
    runs = data["逐局轨迹"]
    notice = "实际演练日志还原；箭头表示移动方向；同地点多次检测不产生新线段；清除位置不是目标精确坐标"
    for group, title in [("s1", "方案一：六局实际演练轨迹"), ("s2", "方案二：六局实际演练轨迹")]:
        fig, axes = plt.subplots(2, 3, figsize=(17.6, 12.8))
        for ax, run in zip(axes.flat, [r for r in runs if r["组"] == group]): draw(ax, run)
        fig.suptitle(title, fontsize=19, y=.98)
        fig.text(.5, .95, "沿用评估报告编号；所有面板同尺度；两组案例不同，不能按面板位置视为同场景配对",
                 ha="center", fontsize=10, color="#586A73")
        fig.legend(handles=legend(), loc="lower center", bbox_to_anchor=(.5, .035),
                   ncol=4, fontsize=9, frameon=False, columnspacing=2.1)
        fig.text(.5, .013, notice, ha="center", fontsize=9, color="#586A73")
        fig.subplots_adjust(left=.065, right=.97, bottom=.165, top=.89, wspace=.25, hspace=.31)
        save(fig, OVERVIEWS[group])
    for run in runs:
        fig, ax = plt.subplots(figsize=(9.3, 10.8))
        draw(ax, run, detail=True)
        fig.suptitle(f"{run['编号']} 实际行动轨迹", fontsize=18, y=.975)
        fig.text(.5, .925, f"案例码：{run['案例码']}  |  检测{run['检测次数']}次  |  清除失败{run['清除失败次数']}次",
                 ha="center", fontsize=10, color="#586A73")
        fig.legend(handles=legend(), loc="lower center", bbox_to_anchor=(.5, .035),
                   ncol=3, fontsize=9, frameon=False, columnspacing=1.5)
        fig.text(.5, .014, "箭头表示移动方向；清除位置是机器狗位置；1800米圆为目标区域边界", ha="center", fontsize=8)
        fig.subplots_adjust(left=.12, right=.95, bottom=.21, top=.85)
        save(fig, f"实际演练轨迹/{run['编号']}_实际轨迹")


def guide(data):
    lines = ["# 十二局实际演练轨迹", "",
             "数据范围：此前s1/s2各6局，沿用S1-1至S2-6编号。图中使用已经核验关联的本地HTTP实际动作坐标，不是新运行或构造轨迹。", "",
             "## 读图方式", "",
             "- 蓝线：前往固定站；橙线：前往追加测向点；绿线：前往清除点。颜色按该段终点动作归类。",
             "- 箭头表示较长移动段的行进方向；按动作顺序连线，同地点多次检测不产生新线段，不添加返回原点路线。",
             "- 实心方块是完成固定扫描的站点，空心方块是其余预设固定站（可能被用作追加测向点）；三角形是追加测向点；绿色空心圆是成功清除位置，红叉是失败清除位置。",
             "- 黑色星形为原点起点，紫色菱形为最后一个动作的位置；单局图的站号从0开始。",
             "- 虚线圆半径1800米，表示目标所在区域，不是机器狗的移动限制。清除标记是机器狗的位置，不代表目标精确坐标。",
             "- 所有面板使用相同坐标范围及等比例坐标轴；案例不同，左右或上下相邻面板不是配对实验。", "",
             "## 方案一：六局总览", "", f"![方案一六局轨迹](../../results/figures/第三问/{OVERVIEWS['s1']}.png)", "",
             "## 方案二：六局总览", "", f"![方案二六局轨迹](../../results/figures/第三问/{OVERVIEWS['s2']}.png)", "",
             "## 单局高清图", "", "| 编号 | 案例码 | 目标数 | 路程（公里） | 固定扫描完成数 | PNG | SVG |",
             "|---|---|---:|---:|---:|---|---|"]
    for run in data["逐局轨迹"]:
        link = f"../../results/figures/第三问/实际演练轨迹/{run['编号']}_实际轨迹"
        lines.append(f"| {run['编号']} | {run['案例码']} | {run['目标数']} | {run['总路程_米']/1000:.3f} | "
                     f"{len(run['已完成固定扫描站序号'])}/{len(run['固定站点'])} | [查看]({link}.png) | [矢量图]({link}.svg) |")
    lines += ["", "## 可从图中直接核对的情况", "",
              "S1-6仅完成站0的固定扫描，通过后续自适应定位及顺路检测清除了16个目标，按题目上限结束。其余6个预设固定站画为空心；其中站2曾用于频道20追加测向及其他频道顺路检测，但没有执行该站完整固定扫描。空心不等于从未到达。没有用预设七站折线替代实际轨迹。",
              "", "方案二六局均访问全部14个固定站。蓝色固定扫描路线相似，后续清除路线随本局观测而变化；这不代表六局目标位置相同。",
              "", "S1-5部分动作位于1800米圆外，按真实记录保留，未裁剪到目标区域内。路线交叉只能说明几何路径交叉，不能单凭图判断某段必然可省。",
              "", "## 复现与核验", "",
              "每局重新累加从原点到各检测/清除位置的直线距离，与已核验运行总路程相差小于0.00001米；核对检测、成功/失败清除次数及固定扫描完成记录。原始输入哈希与此前评估一致。",
              "", "[脱敏轨迹数据与来源哈希](../../results/tables/第三问/十二局实际演练轨迹.json)包含完整检测/清除位置、次序和虚拟时刻，供队友重绘；不含队号、票据或目标真值。",
              "", "在仓库根目录执行`.venv/Scripts/python.exe -X utf8 scripts/plot_q3_practice_trajectories.py`即可从已归档脱敏数据重绘。加`--refresh-source`可重新读取本机原始运行结果并校验，需保留原始私有输入。",
              "", "[绘图脚本](../../scripts/plot_q3_practice_trajectories.py)；[原始评估报告](两组演练评估_s1_s2.md)；[优化建议](基于两组演练的优化建议.md)。"]
    GUIDE.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-source", action="store_true")
    args = parser.parse_args()
    data = extract() if args.refresh_source or not DATA.exists() else json.loads(DATA.read_text(encoding="utf-8"))
    for run in data["逐局轨迹"]:
        actions = run["实际动作"]
        points = np.asarray([[0., 0.]] + [e["位置"] for e in actions])
        assert np.isfinite(points).all() and abs(points).max() < 2050
        assert abs(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() - run["总路程_米"]) < 1e-5
        assert sum(e["动作"] == "检测" for e in actions) == run["检测次数"]
        assert sum(e["动作"] == "清除" and e["结果"] == "success" for e in actions) == run["清除成功次数"]
        assert sum(e["动作"] == "清除" and e["结果"] != "success" for e in actions) == run["清除失败次数"]
    plot(data)
    guide(data)
    print("轨迹核验及绘图完成：12局，2张六局总览、12张单局图，各含PNG/SVG；脱敏数据可独立重绘。")


if __name__ == "__main__":
    main()
