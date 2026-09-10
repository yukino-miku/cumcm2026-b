#!/usr/bin/env python
r"""从第一问几何算法生成中文图表和可复现实验资料。

项目根目录执行：.\.venv\Scripts\python.exe scripts/build_q1_assets.py
本脚本只使用确定性构造算例，不调用官方模拟器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from cumcm2026_b.q1_geometry import solve_bearings


FIGURES = ROOT / "results" / "figures" / "第一问"
TABLES = ROOT / "results" / "tables" / "第一问"
EXPERIMENTS = ROOT / "experiments" / "第一问"
BLUE = "#276C9E"
TEAL = "#1B8278"
ORANGE = "#D48322"
RED = "#B84347"
INK = "#243647"
GRAY = "#637483"
SEED = 20260910
NOTICE = "构造算例，非官方模拟器测试"
KIND_CN = {
    "empty": "空集",
    "unbounded": "无界区域",
    "polygon": "凸多边形",
    "segment": "线段",
    "point": "单点",
}


def setup_style():
    """优先使用项目所在 Windows 环境中的微软雅黑。"""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((p for p in candidates if p.is_file()), None)
    if font_path is None:
        raise RuntimeError("没有找到中文字体；请安装微软雅黑、黑体或 Noto Sans CJK 后重试。")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "cumcm2026-b-q1",
        "pdf.fonttype": 42,
    })
    return str(font_path)


def save_figure(fig, name):
    fig.savefig(FIGURES / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    # Matplotlib 的路径字符串带行尾空格；统一文本格式，图形内容不变。
    svg_path = FIGURES / (name + ".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def compass_to(stations, target):
    """输出以横轴正向为零、逆时针为正的几何角。"""
    delta = np.asarray(target) - np.asarray(stations)
    return np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))


def polygon_area(vertices):
    v = np.asarray(vertices)
    if len(v) < 3:
        return 0.0
    return float(abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1))) / 2)


def result_record(name, stations, bearings, half_width, result):
    """将算法输出映射为适合队友查阅的中文键名。"""
    def optional(value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.floating, np.integer)):
            return float(value) if math.isfinite(float(value)) else "无穷"
        return np.asarray(value).tolist()

    return {
        "算例名称": name,
        "来源说明": NOTICE,
        "角度约定": "横轴正向为零度，逆时针为正；这是几何坐标角，使用其他示向度定义时须转换",
        "检测点坐标_米": np.asarray(stations).tolist(),
        "测得几何方向角_度": np.asarray(bearings).tolist(),
        "误差半宽_度": float(half_width),
        "区域类型": KIND_CN[result.kind],
        "区域顶点_米": optional(result.vertices),
        "区域面积_平方米": polygon_area(result.vertices) if result.kind != "unbounded" else "不适用",
        "区域直径_米": optional(result.diameter),
        "直径点对_米": optional(result.diameter_pair),
        "候选圆心_米": optional(result.circle_center),
        "候选覆盖半径_米": optional(result.cover_radius),
        "最远顶点到候选圆心距离_米": optional(result.max_center_distance),
        "最大超出量_米": optional(result.coverage_excess),
        "同直径圆是否覆盖": optional(result.covered),
    }


def axis_geometry(ax):
    ax.set_xlabel("横坐标（米）")
    ax.set_ylabel("纵坐标（米）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#DCE3E8", alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_region(ax, result, color=BLUE, label="定位区域", alpha=0.25):
    v = np.asarray(result.vertices)
    if len(v) >= 3:
        ax.add_patch(Polygon(v, closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.7, label=label))
        closed = np.vstack((v, v[0]))
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.5)
    elif len(v) == 2:
        ax.plot(v[:, 0], v[:, 1], color=color, label=label, lw=2)
    elif len(v) == 1:
        ax.scatter(v[:, 0], v[:, 1], color=color, label=label)


def draw_diameter(ax, result):
    pair = np.asarray(result.diameter_pair)
    ax.plot(pair[:, 0], pair[:, 1], color=RED, ls="--", lw=1.7, label=f"直径点对连线（{result.diameter:.2f} 米）")
    ax.scatter(pair[:, 0], pair[:, 1], color=RED, s=35, zorder=5)


def figure_flowchart():
    fig, ax = plt.subplots(figsize=(14, 10.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("第一问：定位区域直径与同直径圆覆盖判定流程", fontsize=19, fontweight="bold", y=0.98, color=INK)

    def box(x, y, w, h, text, color=BLUE, fill="#EEF5F9", fontsize=12):
        patch = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.09", linewidth=1.2, edgecolor=color, facecolor=fill)
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, linespacing=1.5)

    def arrow(x1, y1, x2, y2, text="", label_offset=(0.08, 0)):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.15, color=GRAY))
        if text:
            ax.text((x1+x2)/2+label_offset[0], (y1+y2)/2+label_offset[1], text, fontsize=10, color=GRAY, ha="left", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    box(7, 9.25, 7.3, 0.66, "输入检测点、测得方向角与误差半宽（题设为 1°）")
    box(7, 8.13, 7.3, 0.7, "方向角转换与坐标归一化\n每个向前角域转为两个闭半平面约束")
    box(7, 6.9, 5.4, 0.7, "求半平面交：先判可行性与有界性")
    arrow(7, 8.92, 7, 8.48)
    arrow(7, 7.78, 7, 7.25)

    box(1.8, 5.68, 3.1, 0.84, "空集\n检测约束彼此不相容", RED, "#FCEFF0", 11)
    box(7, 5.68, 4.2, 0.84, "非空且有界\n保留真实边界与退化情况", TEAL, "#EAF6F3", 11)
    box(12.1, 5.68, 3.1, 0.84, "无界区域\n直径为无穷，无有限覆盖圆", ORANGE, "#FFF5E6", 10.5)
    arrow(4.3, 6.9, 1.8, 6.1, "不可行", (0, 0.2))
    arrow(7, 6.55, 7, 6.1)
    arrow(9.7, 6.9, 12.1, 6.1, "存在无界方向", (0, 0.2))

    box(3.0, 4.3, 4.7, 0.8, "单点或线段\n单点直径为零；线段直径为两端点距离", TEAL, "#EAF6F3", 10.5)
    box(9.8, 4.3, 5.5, 0.8, "凸多边形\n求顶点，枚举顶点对，得到直径及端点", TEAL, "#EAF6F3", 11)
    arrow(6, 5.26, 3, 4.7, "退化", (0, 0.08))
    arrow(8, 5.26, 9.8, 4.7, "具有面积", (0, 0.05))

    box(7, 2.97, 8.3, 0.78, "候选圆：圆心取直径点对中点，半径取区域直径的一半\n检查所有顶点到圆心的距离是否均不超过该半径", BLUE, "#EEF5F9", 11.5)
    arrow(3, 3.9, 5.5, 3.36)
    arrow(9.8, 3.9, 8.5, 3.36)
    box(3.4, 1.63, 5.5, 0.75, "全部满足：存在同直径覆盖圆\n由凸性，覆盖顶点即覆盖整个区域", TEAL, "#EAF6F3", 11)
    box(10.6, 1.63, 5.5, 0.75, "存在超出顶点：同直径圆无法覆盖\n圆心已被直径端点唯一确定", RED, "#FCEFF0", 11)
    arrow(5.5, 2.58, 3.4, 2.01, "是", (0, 0.02))
    arrow(8.5, 2.58, 10.6, 2.01, "否", (0, 0.02))
    ax.text(7, 0.48, "计算范围：附录角域交会模型。若再与圆形先验区域相交，应另行处理圆弧边界。\n数值实现使用容差，并保留最大超出量，便于检查临界判定。", ha="center", va="center", fontsize=11, color=GRAY, linespacing=1.6)
    fig.subplots_adjust(top=0.94, bottom=0.025, left=0.045, right=0.955)
    save_figure(fig, "01_算法流程图")


def figure_intersection(stations, bearings, result):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("两站测向：由 ±1° 角域交会得到定位多边形", fontsize=18, fontweight="bold", color=INK)
    for i, (station, bearing) in enumerate(zip(stations, bearings)):
        color = [BLUE, ORANGE][i]
        angles = np.radians([bearing-1, bearing+1])
        end = station + 1800 * np.column_stack((np.cos(angles), np.sin(angles)))
        axes[0].add_patch(Polygon(np.vstack((station, end)), facecolor=color, edgecolor="none", alpha=0.17, label=f"检测点{i+1}的可行角域"))
        for point in end:
            axes[0].plot([station[0], point[0]], [station[1], point[1]], color=color, linewidth=0.8)
        axes[0].scatter(*station, marker="^", s=75, c=color, zorder=4)
        axes[0].annotate(f"检测点{i+1}", station, xytext=(0, -21), textcoords="offset points", ha="center", color=color)
    draw_region(axes[0], result, TEAL, "交会定位区域", 0.6)
    axes[0].set_xlim(-690, 690)
    axes[0].set_ylim(-150, 1270)
    axes[0].set_title("角域交会全景", pad=13)
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.96)

    draw_region(axes[1], result, TEAL, "定位多边形")
    draw_diameter(axes[1], result)
    axes[1].scatter(result.vertices[:, 0], result.vertices[:, 1], color=TEAL, s=22, zorder=3)
    axes[1].scatter([0], [1000], marker="+", color=INK, s=90, linewidth=1.7, label="构造目标位置")
    axes[1].add_patch(Circle(result.circle_center, result.cover_radius, facecolor="none", edgecolor=RED, linestyle=":", linewidth=1.6, label="以直径点对为直径的圆"))
    pad = result.diameter * 0.17
    center = np.mean(result.vertices, axis=0)
    radius = max(result.cover_radius, np.max(np.linalg.norm(result.vertices-center, axis=1))) + pad
    axes[1].set_xlim(center[0]-radius, center[0]+radius)
    axes[1].set_ylim(center[1]-radius, center[1]+radius)
    axes[1].set_title("局部放大：顶点与直径", pad=13)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9.5, frameon=False, ncol=1)
    for ax in axes:
        axis_geometry(ax)
    fig.text(0.02, 0.015, NOTICE + "；方向角以横轴正向为零、逆时针为正。", fontsize=10, color=GRAY)
    fig.subplots_adjust(top=0.83, bottom=0.22, wspace=0.22)
    save_figure(fig, "02_双站角域与定位区域")


def figure_counterexample(stations, bearings, result):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.5), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("反例：区域直径相同，并不能保证圆覆盖区域", fontsize=18, fontweight="bold", color=INK)
    ax = axes[0]
    draw_region(ax, result, BLUE, "三站角域交会得到的等边三角形", 0.15)
    draw_diameter(ax, result)
    ax.add_patch(Circle(result.circle_center, result.cover_radius, facecolor=RED, edgecolor=RED, alpha=0.08))
    ax.add_patch(Circle(result.circle_center, result.cover_radius, fill=False, edgecolor=RED, lw=1.8, ls="--", label="同直径圆：半径为直径的一半"))
    circumcenter = np.mean(result.vertices, axis=0)
    circumradius = result.diameter / np.sqrt(3)
    ax.add_patch(Circle(circumcenter, circumradius, fill=False, edgecolor=TEAL, lw=1.8, label="最小覆盖圆：半径为直径除以根号三"))
    ax.scatter(*result.circle_center, marker="x", color=RED, s=70, zorder=5, label="直径点对中点")
    ax.scatter(*circumcenter, marker="+", color=TEAL, s=90, zorder=5, label="等边三角形外心")
    distances = np.linalg.norm(result.vertices-result.circle_center, axis=1)
    farthest = result.vertices[int(np.argmax(distances))]
    ax.scatter(*farthest, s=60, c=RED, zorder=6)
    ax.annotate("此顶点位于同直径圆外", farthest, xytext=(10, 18), textcoords="offset points", color=RED, fontsize=10, arrowprops={"arrowstyle": "->", "color": RED})
    ax.set_xlim(-6, 28.4)
    ax.set_ylim(-11.4, 25.2)
    axis_geometry(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5, frameon=False, ncol=2)

    ax = axes[1]
    labels = ["同直径圆半径", "最小覆盖圆半径", "超出顶点到\n候选圆心距离"]
    values = [result.cover_radius, circumradius, result.max_center_distance]
    colors = [RED, TEAL, ORANGE]
    bars = ax.barh(labels, values, color=colors, height=0.46)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.2)
    ax.set_xlabel("距离（米）")
    ax.set_title("半径与覆盖需求的直接比较", fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value+0.22, bar.get_y()+bar.get_height()/2, f"{value:.3f}", va="center", fontsize=11, color=INK)
    fig.text(0.61, 0.19, f"边长 = 区域直径 = {result.diameter:.3f} 米\n最大超出量 = {result.coverage_excess:.3f} 米\n结论：换圆心也不能实现同直径覆盖。", va="top", fontsize=10.5, color=INK, linespacing=1.6)
    fig.text(0.02, 0.01, NOTICE + "；三个检测角域的半宽均为 1°，完整站点和角度见实验数据。", fontsize=10, color=GRAY)
    fig.subplots_adjust(top=0.83, bottom=0.32, wspace=0.56)
    save_figure(fig, "03_同直径覆盖圆反例")


def figure_nested(two, three):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("新增相容检测约束：可行定位区域嵌套缩小", fontsize=18, fontweight="bold", color=INK)
    draw_region(axes[0], two, BLUE, "原两站定位区域", 0.13)
    draw_region(axes[0], three, TEAL, "新增第三站后的定位区域", 0.4)
    axes[0].scatter([0], [1000], color=INK, marker="+", s=95, label="构造目标位置")
    axes[0].set_title("同一组数据，保留原有约束", fontsize=14)
    axis_geometry(axes[0])
    axes[0].autoscale_view()
    axes[0].margins(0.2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=False)

    ratios = [three.diameter/two.diameter, polygon_area(three.vertices)/polygon_area(two.vertices)]
    x = np.arange(2)
    axes[1].bar(x-0.16, [1, 1], 0.3, color=BLUE, alpha=0.45, label="原两站区域")
    bars = axes[1].bar(x+0.16, ratios, 0.3, color=TEAL, label="新增第三站后")
    axes[1].set_xticks(x, ["区域直径", "区域面积"])
    axes[1].set_ylim(0, 1.22)
    axes[1].set_ylabel("相对于原两站区域的比例")
    axes[1].set_title("集合包含关系对应的数值核查", fontsize=14)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    for bar, value in zip(bars, ratios):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+0.025, f"{value:.3f}", ha="center", fontsize=11)
    axes[1].legend(frameon=False, loc="upper right", fontsize=10)
    fig.text(0.02, 0.01, NOTICE + "。这里只验证交集与直径的性质，不据此提出第二问的选点策略。", fontsize=10, color=GRAY)
    fig.subplots_adjust(top=0.81, bottom=0.29, wspace=0.4)
    save_figure(fig, "04_新增检测约束的区域嵌套")


def angle_experiments():
    rows = []
    for angle in range(5, 176):
        stations = np.array([[-1000, 0], [-1000*np.cos(np.radians(angle)), -1000*np.sin(np.radians(angle))]])
        bearings = compass_to(stations, [0, 0])
        result = solve_bearings(stations, bearings, half_width_deg=1)
        record = result_record(f"交会角{angle}度", stations, bearings, 1, result)
        record["中心射线交会角_度"] = angle
        rows.append(record)
    return rows


def width_experiments():
    stations = np.array([[-1000, 0], [-500, -500*np.sqrt(3)]])
    bearings = compass_to(stations, [0, 0])
    rows = []
    for half_width in np.linspace(0.2, 3.0, 29):
        result = solve_bearings(stations, bearings, half_width_deg=float(half_width))
        rows.append(result_record(f"误差半宽{half_width:.1f}度", stations, bearings, half_width, result))
    return rows


def figure_angle_sweep(rows):
    angles = np.array([r["中心射线交会角_度"] for r in rows])
    diameters = np.array([r["区域直径_米"] for r in rows])
    covered = np.array([r["同直径圆是否覆盖"] for r in rows], dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.18})
    fig.suptitle("固定检测距离下的交会角扫描：核查直径与覆盖判定", fontsize=16, fontweight="bold", color=INK)
    axes[0].plot(angles, diameters, color=BLUE, linewidth=2)
    axes[0].set_ylabel("定位区域直径（米）")
    axes[0].grid(alpha=0.22)
    for angle in [10, 45, 90, 135, 170]:
        k = int(np.where(angles == angle)[0][0])
        axes[0].scatter(angle, diameters[k], color=BLUE, s=26)
        axes[0].annotate(f"{diameters[k]:.1f}", (angle, diameters[k]), xytext=(4, 8), textcoords="offset points", fontsize=9, color=BLUE)
    axes[0].text(0.5, 0.9, "两站到构造目标的距离均为 1000 米；方向误差半宽固定为 1°", transform=axes[0].transAxes, ha="center", fontsize=10, color=GRAY)
    for mask, color, label, y in [(covered, TEAL, "能够覆盖", 1), (~covered, RED, "不能覆盖", 0)]:
        if np.any(mask):
            axes[1].scatter(angles[mask], np.full(np.sum(mask), y), color=color, s=12, marker="s", label=label)
    for index in np.where(~covered)[0]:
        axes[1].annotate(f"{angles[index]}°：超出 {rows[index]['最大超出量_米']:.3f} 米", (angles[index], 0), xytext=(15, -2), textcoords="offset points", fontsize=9, color=RED, va="center")
    axes[1].set_yticks([0, 1], ["不能覆盖", "能够覆盖"])
    axes[1].set_ylim(-0.4, 1.4)
    axes[1].set_ylabel("同直径圆")
    axes[1].set_xlabel("中心射线交会角（度）")
    axes[1].set_xticks(np.arange(0, 181, 15))
    axes[1].set_xlim(3, 177)
    axes[1].grid(axis="x", alpha=0.22)
    fig.text(0.02, 0.015, NOTICE + "。扫描仅用于第一问算法验证；不作为第二问选点策略或模拟器成绩。", fontsize=9.5, color=GRAY)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.13, right=0.97)
    save_figure(fig, "05_构造交会角扫描")


def figure_width_sweep(rows):
    widths = np.array([r["误差半宽_度"] for r in rows])
    diameters = np.array([r["区域直径_米"] for r in rows])
    fig, ax = plt.subplots(figsize=(10.0, 5.7))
    ax.plot(widths, diameters, marker="o", markersize=3.4, linewidth=2, color=BLUE, label="几何算法计算的区域直径")
    index = int(np.argmin(abs(widths-1)))
    ax.axvline(1, ls="--", color=RED, lw=1.2)
    ax.scatter([1], [diameters[index]], color=RED, s=65, zorder=5)
    ax.annotate(f"题设误差半宽：1°\n区域直径：{diameters[index]:.3f} 米", (1, diameters[index]), xytext=(26, -6), textcoords="offset points", fontsize=11, color=RED, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#F0D8D9"})
    ax.set_title("误差半宽增大时，可行定位区域及其直径不减", fontsize=17, pad=18)
    ax.set_xlabel("构造方向误差半宽（度）")
    ax.set_ylabel("定位区域直径（米）")
    ax.set_xlim(0.1, 3.1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.text(0.03, 0.91, "检测距离固定为 1000 米，中心射线交会角固定为 60°", transform=ax.transAxes, fontsize=10, color=GRAY)
    fig.text(0.025, 0.017, NOTICE + "。除 1° 外的参数仅用于敏感性核查，不表示题设误差发生变化。", fontsize=9.5, color=GRAY)
    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.1, right=0.97)
    save_figure(fig, "06_构造误差半宽敏感性")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_reports(cases, angle_rows, width_rows, font_path):
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_head = "无法读取；以脚本哈希核对代码版本"
    import scipy
    metadata = {
        "实验名称": "第一问确定性几何构造验证",
        "数据来源": NOTICE,
        "随机种子": SEED,
        "随机性说明": "保留固定种子；本轮全部为确定性构造，无随机抽样",
        "计算范围": "只求示向误差角域交集，未与半径1800米的圆形先验区域求交",
        "生成命令": ".\\.venv\\Scripts\\python.exe scripts/build_q1_assets.py",
        "角度约定": "横轴正向为零度、逆时针为正",
        "几何距离单位": "米",
        "运行环境": {
            "Python版本": platform.python_version(),
            "NumPy版本": np.__version__,
            "SciPy版本": scipy.__version__,
            "Matplotlib版本": matplotlib.__version__,
            "操作系统": platform.platform(),
            "绘图字体": font_path,
        },
        "生成时已有Git提交": git_head,
        "版本说明": "生成时提交可早于本轮结果提交；下列源文件SHA256精确标识实际使用的代码",
        "代码SHA256": {
            "scripts/build_q1_assets.py": sha256(Path(__file__)),
            "src/cumcm2026_b/q1_geometry.py": sha256(ROOT / "src" / "cumcm2026_b" / "q1_geometry.py"),
        },
        "交会角扫描配置": {"检测距离_米": 1000, "误差半宽_度": 1, "起始角_度": 5, "终止角_度": 175, "步长_度": 1},
        "误差半宽扫描配置": {"检测距离_米": 1000, "交会角_度": 60, "起始半宽_度": 0.2, "终止半宽_度": 3.0, "步长_度": 0.1},
    }
    write_json(EXPERIMENTS / "实验配置与环境.json", metadata)
    write_json(TABLES / "构造算例输入输出.json", cases)
    write_json(TABLES / "交会角扫描输入输出.json", angle_rows)
    write_json(TABLES / "误差半宽扫描输入输出.json", width_rows)

    def row(record):
        covered = "是" if record["同直径圆是否覆盖"] else "否"
        return f"| {record['算例名称']} | {record['区域类型']} | {len(record['区域顶点_米'])} | {record['区域直径_米']:.6f} | {record['候选覆盖半径_米']:.6f} | {record['最大超出量_米']:.6f} | {covered} |"

    summary = [
        "# 第一问构造算例结果汇总", "",
        "> 数据均由第一问几何算法实际计算；属于构造算例，非官方模拟器测试。", "",
        "所有坐标和距离单位为米；误差半宽均为 1°。角度以横轴正向为零、逆时针为正。", "",
        "| 算例 | 区域类型 | 顶点数 | 直径 | 同直径圆半径 | 最大超出量 | 同直径圆覆盖 |",
        "|---|---|---:|---:|---:|---:|---|",
        *(row(r) for r in cases), "",
        "最大超出量 = 最远顶点到候选圆心的距离 − 区域直径的一半。理论上非负；程序将浮点舍入引起的微小负值记录为零。", "",
        "完整输入、输出顶点、直径点对和候选圆心见同目录 JSON 文件。",
    ]
    (TABLES / "构造算例汇总.md").write_text("\n".join(summary)+"\n", encoding="utf-8")

    two, triangle, three = cases
    angle_samples = [r for r in angle_rows if r["中心射线交会角_度"] in [5, 10, 30, 45, 60, 90, 91, 120, 135, 150, 170, 175]]
    angle_lines = ["| 交会角（度） | 区域直径（米） | 最大超出量（米） | 同直径圆覆盖 |", "|---:|---:|---:|---|"]
    for r in angle_samples:
        angle_lines.append(f"| {r['中心射线交会角_度']} | {r['区域直径_米']:.6f} | {r['最大超出量_米']:.6f} | {'是' if r['同直径圆是否覆盖'] else '否'} |")
    width_samples = [r for r in width_rows if any(np.isclose(r["误差半宽_度"], val) for val in [0.2, 0.5, 1, 1.5, 2, 2.5, 3])]
    width_lines = ["| 误差半宽（度） | 区域直径（米） |", "|---:|---:|"]
    width_lines += [f"| {r['误差半宽_度']:.1f} | {r['区域直径_米']:.6f} |" for r in width_samples]
    included = sum(bool(r["同直径圆是否覆盖"]) for r in angle_rows)
    report = f"""# 第一问构造实验说明与复现记录

## 1. 目的与边界

本目录用于帮助队友理解和复核第一问的几何算法，并保留可供论文使用的图表。全部结果来自确定性构造输入，**不是官方模拟器测试结果，也不是实际任务成绩**。脚本未调用模拟器。

本轮只研究示向误差角域的交集。坐标采用平面直角坐标系，几何方向角从横轴正向逆时针计量，与赛题附录 2 的示向度定义一致。若导入其他方位角定义的数据，必须先进行坐标角转换。除专门的敏感性扫描外，方向误差半宽固定为题设的 1°。

## 2. 复现方法

在项目根目录执行：

```powershell
.\\.venv\\Scripts\\python.exe scripts/build_q1_assets.py
```

所需 Python 依赖为 NumPy、SciPy、Matplotlib，版本和实际源代码 SHA256 见 [实验配置与环境](实验配置与环境.json)。脚本用微软雅黑输出中文，PNG 分辨率为 220 点/英寸，SVG 将字体转换为路径便于跨机器使用。当前构造无随机抽样，保留固定种子 {SEED}。

## 3. 双站角域交会

检测点分别为（−500，0）和（500，0），测得方向取它们指向构造目标（0，1000）的方向。每个角域宽度为 2°，由两个闭半平面描述。

这是图表采用的双站示意构造，区别于配置文件中用于解析单元验证的正交双站构造；各组输入独立保存在对应配置或结果数据中。

算法得到 {len(two['区域顶点_米'])} 个顶点，定位直径为 {two['区域直径_米']:.6f} 米，面积为 {two['区域面积_平方米']:.6f} 平方米。同直径圆覆盖判定为“{'是' if two['同直径圆是否覆盖'] else '否'}”。图 02 同时保留检测几何全景与区域局部，避免把小定位区域误当成无面积点。

## 4. 同直径圆不能覆盖的三站反例

先取边长为 20 米的逆时针等边三角形，顶点为（0，0）、（20，0）、（10，10√3）。对于每条逆时针边，将其起点沿边的反方向延伸 25 倍边长（即 500 米）设置检测点；测得方向角取该边方向加 1°，误差半宽取 1°。三站分别为（−500，0）、（270，−250√3）、（260，260√3），测得方向角为 1°、121°、241°（角度相差 360° 表示同一方向）。

这样，角域的一条边界与三角形一条支撑边重合，另一条边界包含三角形且不截去任何顶点。三个角域的交集恰好是原等边三角形。完整站点、角度和算法顶点均保存在输入输出 JSON 中，可独立复核。

计算得到区域直径 {triangle['区域直径_米']:.6f} 米，同直径圆半径 {triangle['候选覆盖半径_米']:.6f} 米。以直径端点中点为圆心时，第三个顶点到圆心距离为 {triangle['最远顶点到候选圆心距离_米']:.6f} 米，超出 {triangle['最大超出量_米']:.6f} 米；该圆不能覆盖。等边三角形的最小覆盖圆半径为 {triangle['区域直径_米']/math.sqrt(3):.6f} 米。

这组检测点与论文反例、项目配置中的构造一致。若把全向干扰源置于三角形重心，各检测距离均约为 510 米，大于 5 米且小于 1000 米，满足题设可检测距离条件。等边三角形有三组并列直径点对，图中的候选圆可以选择与正文不同的一条边，覆盖失败的结论及超出量相同。该构造只用于第一问角域几何性质的证明，不指定后续问题的航路。

## 5. 新增相容约束的集合嵌套核查

在双站算例基础上新增检测点（1000，1000），测得方向为 180°，仍指向同一构造目标。新增区域是原区域与新角域的交集，因此必定包含于原区域。

区域直径由 {two['区域直径_米']:.6f} 米变为 {three['区域直径_米']:.6f} 米，比例为 {three['区域直径_米']/two['区域直径_米']:.6f}；面积由 {two['区域面积_平方米']:.6f} 平方米变为 {three['区域面积_平方米']:.6f} 平方米，比例为 {three['区域面积_平方米']/two['区域面积_平方米']:.6f}。这是交集单调性的验证；本轮不将其扩展为第二问选点策略。

## 6. 交会角扫描

构造目标位于原点，两站到目标的距离均为 1000 米。第一站为（−1000，0），第二站为（−1000 cos γ，−1000 sin γ），各自测得方向均指向原点。误差半宽固定为 1°，γ 从 5° 到 175°，步长为 1°。共 {len(angle_rows)} 组输入，均独立调用同一几何算法。

以下为完整扫描的部分行，完整输入输出保存在 [交会角扫描数据](../../results/tables/第一问/交会角扫描输入输出.json)。同直径圆能够覆盖的构造样本为 {included} 组，不能覆盖的构造样本为 {len(angle_rows)-included} 组。此比例仅描述这一确定性构造族，不是概率估计，也不能外推到任意检测位置。

{chr(10).join(angle_lines)}

## 7. 误差半宽敏感性核查

固定两站检测距离均为 1000 米、中心射线交会角为 60°。将误差半宽从 0.2° 扫描至 3.0°，步长为 0.1°，共 {len(width_rows)} 组。扩大误差角域只会放宽约束，因此可行区域及其直径不会减小。只有半宽 1° 对应本题设定，其余参数用于算法核查。

{chr(10).join(width_lines)}

## 8. 图表目录与论文使用建议

| 图号 | 中文图名 | 用途 |
|---|---|---|
| 01 | 算法流程图 | 展示输入、空集/无界/退化分支、直径计算与覆盖判定 |
| 02 | 双站角域与定位区域 | 解释角域交会与顶点直径计算 |
| 03 | 同直径覆盖圆反例 | 支撑“同直径圆不一定覆盖”的结论 |
| 04 | 新增检测约束的区域嵌套 | 复核交集与直径单调性 |
| 05 | 构造交会角扫描 | 展示同一算法在一组连续变化输入下的输出 |
| 06 | 构造误差半宽敏感性 | 验证误差角域扩张导致直径不减 |

每张图同时保存 PNG 与 SVG，位于 [中文图表目录](../../results/figures/第一问/)。第一问论文优先使用图 01—03，图 04—06 可作为验证资料或附录，避免为丰富版面重复堆图。图注应明确注明“构造算例，非官方模拟器测试”；不要改写为实测数据或比赛模拟器成绩。
"""
    report = report.replace("示向度 convention", "示向度定义")
    (EXPERIMENTS / "构造实验报告.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="生成第一问中文图表及构造实验资料")
    parser.parse_args()
    for directory in (FIGURES, TABLES, EXPERIMENTS):
        directory.mkdir(parents=True, exist_ok=True)
    font_path = setup_style()
    np.random.seed(SEED)

    stations_two = np.array([[-500., 0.], [500., 0.]])
    bearings_two = compass_to(stations_two, [0., 1000.])
    two = solve_bearings(stations_two, bearings_two)
    vertices_expected = np.array([[0., 0.], [20., 0.], [10., 10.*np.sqrt(3)]])
    edges = np.roll(vertices_expected, -1, axis=0)-vertices_expected
    station_triangle = vertices_expected - 25 * edges
    bearings_triangle = (np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))+1) % 360
    triangle = solve_bearings(station_triangle, bearings_triangle)
    stations_three = np.vstack((stations_two, [1000., 1000.]))
    bearings_three = compass_to(stations_three, [0., 1000.])
    three = solve_bearings(stations_three, bearings_three)

    # 这些断言针对数学不变量和解析反例，避免输出失真的“漂亮图片”。
    assert all(r.kind == "polygon" for r in (two, triangle, three)), "基本构造必须产生有面积的有界多边形"
    assert len(triangle.vertices) == 3 and np.isclose(triangle.diameter, 20, atol=1e-6), "反例交集应为边长20米的等边三角形"
    assert not triangle.covered and np.isclose(triangle.coverage_excess, 10*np.sqrt(3)-10, atol=1e-6), "等边三角形的解析覆盖反例未通过"
    assert np.max(np.linalg.norm(vertices_expected[None, :, :]-station_triangle[:, None, :], axis=2)) < 1500, "反例的每个顶点均应位于各站1500米检测距离以内"
    assert three.diameter <= two.diameter+1e-7 and polygon_area(three.vertices) <= polygon_area(two.vertices)+1e-6, "增加约束后区域不应扩大"
    angle_rows = angle_experiments()
    width_rows = width_experiments()
    assert all(row["区域类型"] == "凸多边形" for row in angle_rows+width_rows), "扫描区间应全部有界且非退化"
    assert np.min(np.diff([r["区域直径_米"] for r in width_rows])) >= -1e-7, "误差半宽增大时直径不应减少"

    cases = [
        result_record("双站基础算例", stations_two, bearings_two, 1, two),
        result_record("三站等边三角形反例", station_triangle, bearings_triangle, 1, triangle),
        result_record("新增第三站算例", stations_three, bearings_three, 1, three),
    ]
    figure_flowchart()
    figure_intersection(stations_two, bearings_two, two)
    figure_counterexample(station_triangle, bearings_triangle, triangle)
    figure_nested(two, three)
    figure_angle_sweep(angle_rows)
    figure_width_sweep(width_rows)
    write_reports(cases, angle_rows, width_rows, font_path)
    print("已生成 6 张中文图（PNG + SVG）、3 份完整输入输出 JSON、1 份结果汇总、实验配置和中文报告。")
    print(f"基础直径：{two.diameter:.6f} 米；反例直径：{triangle.diameter:.6f} 米；反例最大超出量：{triangle.coverage_excess:.6f} 米。")
    print("全部为构造算例，非官方模拟器测试。")


if __name__ == "__main__":
    main()
