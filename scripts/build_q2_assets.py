#!/usr/bin/env python
"""生成第二问中文图表、实验记录和队友用表格；--plots-only 可重画归档数据。"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cumcm2026_b.q2_physical_geometry import analytic_candidates, sample_region
from q2_experiments import make_region, run_experiments, to_local, write_json

FIGURES = ROOT / "results/figures/第二问"
TABLES = ROOT / "results/tables/第二问"
EXPERIMENTS = ROOT / "experiments/第二问"
NOTICE = "构造算例，非官方模拟器测试"
BLUE, TEAL, ORANGE, RED = "#276C9E", "#1B8278", "#D48322", "#B84347"
INK, GRAY = "#243647", "#637483"


def setup_style():
    """沿用第一问中文字体和配色；SVG 字形转路径便于论文移植。"""
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        raise RuntimeError("缺少中文字体，请安装微软雅黑、黑体或 Noto Sans CJK。")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "path",
        "svg.hashsalt": "cumcm2026-b-q2", "pdf.fonttype": 42})
    return str(font_path)


def save_figure(fig, name):
    # 先在内存中编码，再以只写模式落盘，避免部分 Windows 环境拒绝 w+b 打开图片。
    buffer=BytesIO()
    fig.savefig(buffer,format="png",dpi=220,bbox_inches="tight")
    (FIGURES / (name + ".png")).write_bytes(buffer.getvalue())
    path = FIGURES / (name + ".svg")
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def close_curve(points):
    array = np.asarray(points)
    return np.vstack([array, array[0]])


def physical_outline(region, local=False):
    """按凸边界顺序排序绘图样本；曲线本身仍由精确参数保存在 JSON。"""
    points = sample_region(region, boundary_count=384, radial_levels=0)
    center = region.representative_point
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    return to_local(points, region) if local else points


def label_axes(ax, local=True):
    ax.set_xlabel("沿首次示向方向坐标（米）" if local else "东向坐标 x（米）")
    ax.set_ylabel("垂直首次示向方向坐标（米）" if local else "北向坐标 y（米）")
    ax.grid(alpha=0.18)


def footer(fig, extra=""):
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
    elif "\n" in extra:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.17))
    fig.text(0.5, 0.01, NOTICE + ("；" + extra if extra else ""), ha="center", color=GRAY, fontsize=9)


def finite_value(value):
    return value is not None and np.isfinite(value)


def number(value, digits=3):
    return f"{value:.{digits}f}" if finite_value(value) else "不提供有限值"


def load_archived():
    """仅重画时加载完整结果，不重复执行搜索。"""
    def load(name):
        path = TABLES / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少归档结果 {path}，请先运行不带 --plots-only 的完整构建。")
        return json.loads(path.read_text(encoding="utf-8"))
    archive = load("鲁棒搜索输入输出.json")
    configuration = archive["configuration"]
    runs = []
    for item in archive["cases"]:
        runs.append({"case": item["input"], "region": make_region(item["input"], configuration["physical"]),
            "search": item["search"], "reliable_boundary": np.asarray(item["reliable_boundary"]),
            "targets": np.asarray(item["target_scenarios"]), "reliable_area_m2": item["reliable_area_m2"]})
    sensitivity = load("物理范围敏感性.json")
    return configuration, {"runs": runs, "baselines": load("策略对比输入输出.json")["strategies"],
        "convergence": load("采样收敛分析.json"), "angle_sensitivity": load("角度误差敏感性.json")["rows"],
        "radii": sensitivity["reception_radius"], "distances": sensitivity["distance_interval"],
        "angle_geometry": load("交会角与直径数据.json")["rows"]}


def figure_flow():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    blocks = [
        (1, 8.5, "首次检测信息", "第一站坐标、示向度、题设物理范围"),
        (6, 8.5, "构造目标物理可行域", "窄角域 ∩ 1500 米接收圆 ∩ 1800 米目标圆"),
        (6, 6.5, "连续几何保证接收与可测向", "最远距离 ≤ 1000 米；最近距离 > 5 米"),
        (1, 6.5, "中心线解析候选", "距离区间 → 两圆透镜 → 两侧候选初值"),
        (1, 4.5, "鲁棒定位直径评价", "连续可见方向区间＋误差；复用第一问几何"),
        (6, 4.5, "粗格网与局部细化", "两侧分别搜索；最坏观测由第一问复核"),
        (6, 2.5, "生成近最优候选区域", "直径 ≤ (1＋阈值) × 当前数值最优值"),
        (1, 2.5, "保存结果，供第三问调用", "候选坐标、定位指标、移动距离与几何参数"),
    ]
    for x, y, title, body in blocks:
        ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.65), 3.7, 1.2, boxstyle="round,pad=0.16", facecolor="#EDF4F8", edgecolor=BLUE, linewidth=1.3))
        ax.text(x + 1.25, y + 0.1, title, ha="center", color=INK, fontsize=12, weight="bold")
        ax.text(x + 1.25, y - 0.25, body, ha="center", color=GRAY, fontsize=8.6)
    for p, q in [((4.3,8.5),(5.2,8.5)),((7.3,7.7),(7.3,7.25)),((5.2,6.5),(4.3,6.5)),
                 ((2.3,5.7),(2.3,5.25)),((4.3,4.5),(5.2,4.5)),((7.3,3.7),(7.3,3.25)),((5.2,2.5),(4.3,2.5))]:
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=16,color=TEAL,linewidth=1.6))
    ax.text(5, 1.2, "三层验证：解析几何测试 → 采样与格网收敛 → 第一问最坏场景复核", ha="center", color=TEAL, fontsize=12)
    ax.set_title("第二问总体算法：从首次测向信息到可靠的近最优候选区", pad=14)
    footer(fig, "连续物理约束精确求值；鲁棒搜索仍为可验证的数值近似")
    save_figure(fig, "01_第二问总体算法流程图")


def figure_physical(data):
    standard, edge = data["runs"][0], data["runs"][-1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    ax = axes[0]; region = standard["region"]
    ax.add_patch(Circle(region.target_center, 1800, fill=False, edgecolor=GRAY, linestyle="--", label="1800 米目标圆"))
    ax.add_patch(Circle(region.station, 1500, fill=False, edgecolor=ORANGE, label="1500 米最大接收圆"))
    outline = physical_outline(region)
    ax.fill(*outline.T, color=TEAL, alpha=.6, label="首次物理可行域")
    ax.plot([0, 1600], [0, 0], color=BLUE, linestyle="--", label="首次中心示向线")
    ax.scatter(*region.station, color=INK, s=40, zorder=4)
    ax.annotate("第一检测点／目标圆心", region.station, xytext=(-1700, -350), fontsize=9, arrowprops={"arrowstyle":"-", "color":GRAY})
    ax.set_xlim(-1900, 1900); ax.set_ylim(-1900, 1900); ax.set_aspect("equal")
    ax.set_title("标准构造：三类约束叠加"); label_axes(ax, False); ax.legend(fontsize=8,loc="lower left")
    ax = axes[1]; outline = physical_outline(region, True)
    ax.fill(*outline.T, color=TEAL, alpha=.25, label="物理可行域")
    ax.plot(*close_curve(outline).T, color=TEAL)
    ax.plot([0,1500],[0,0],color=BLUE,linestyle="--",label="中心示向线")
    for sign in [-1,1]:
        ax.plot([0,1500*np.cos(np.radians(1))],[0,sign*1500*np.sin(np.radians(1))], color=ORANGE)
    ax.annotate("角域上、下边界：±1°",(1000,17.45),xytext=(200,42),arrowprops={"arrowstyle":"->","color":GRAY},fontsize=9)
    ax.set_ylim(-55,55); ax.set_xlim(-40,1550); ax.set_title("窄角域放大（纵横比例不同）"); label_axes(ax); ax.legend(fontsize=8,loc="lower left")
    ax=axes[2]; region=edge["region"]; outline=physical_outline(region,True)
    ax.fill(*outline.T,color=TEAL,alpha=.25,label="被目标圆截断的可行域")
    ax.plot(*close_curve(outline).T,color=TEAL)
    ax.plot([0,210],[0,0],linestyle="--",color=BLUE,label="首次中心示向线")
    ax.scatter(0,0,color=INK,s=30)
    ax.set_title("近圆边缘构造：两侧截断不同")
    ax.set_ylim(-4.5,4.5); ax.set_xlim(-5,210); label_axes(ax)
    ax.legend(fontsize=8,loc="lower left")
    fig.suptitle("首次测向物理可行域：角度信息与两个物理圆共同约束目标",fontsize=16)
    footer(fig,"首次五米内小区域保留为保守凸外包；右两图为窄角域放大")
    save_figure(fig,"02_首次测向物理可行域")


def lens_boundary(count=600):
    x=np.linspace(500,1000,count)
    y=np.minimum(np.sqrt(np.maximum(0,1000**2-x**2)),np.sqrt(np.maximum(0,1000**2-(x-1500)**2)))
    return np.vstack([np.column_stack([x,y]),np.column_stack([x[::-1],-y[::-1]])])


def figure_lens(data):
    fig,ax=plt.subplots(figsize=(11,7))
    for center, color, label in [([0,0],BLUE,"以距离区间近端为圆心，半径 1000 米"),([1500,0],ORANGE,"以距离区间远端为圆心，半径 1000 米")]:
        ax.add_patch(Circle(center,1000,fill=False,color=color,linewidth=1.3,label=label))
    lens=lens_boundary(); ax.fill(*lens.T,color=TEAL,alpha=.18,label="中心线近似下的可靠透镜")
    ax.plot([0,1500],[0,0],color=INK,linewidth=3,label="中心线距离区间：0～1500 米")
    candidates=analytic_candidates(data["runs"][0]["region"])
    ax.scatter(*candidates.T,marker="*",s=200,color=RED,zorder=5,label="两侧解析候选")
    ax.plot([750,750],[0,candidates[0,1]],color=GRAY,linestyle="--")
    ax.text(790,310,"661.438 米",color=GRAY); ax.text(320,35,"750 米",color=GRAY)
    ax.annotate("(750, 661.438)", candidates[0],xytext=(900,730),fontsize=11,arrowprops={"arrowstyle":"->","color":RED})
    ax.annotate("(750, −661.438)", candidates[1],xytext=(870,-820),fontsize=11,arrowprops={"arrowstyle":"->","color":RED})
    ax.scatter([0,1500],[0,0],s=45,color=INK,zorder=4)
    ax.set_xlim(-150,1650); ax.set_ylim(-1050,1050); ax.set_aspect("equal")
    label_axes(ax); ax.set_title("中心线解析近似：用端点两圆交集平衡侧向偏移和保证接收")
    ax.legend(loc="upper left",bbox_to_anchor=(1.02,1),fontsize=9)
    footer(fig,"完整 ±1° 模型还需修正；图中解析候选不是精确全局最优解")
    save_figure(fig,"03_中心线近似与透镜可靠域")


def figure_reliable(data):
    run=data["runs"][0]; region=run["region"]
    # 标准完整扇形的 K 恰好是原点和两弧端点对应三圆交。
    # 用解析上边界加入尖点，避免仅256条射线漏掉尖点而使放大图失真。
    radius=run["search"]["configuration"]["reception_lower_m"]-run["search"]["configuration"]["safety_margin_m"]
    delta=math.radians(region.half_width_deg); far_x=region.reception_upper*math.cos(delta); far_y=region.reception_upper*math.sin(delta)
    left=far_x-math.sqrt(radius**2-far_y**2)
    height=math.sqrt(radius**2-(region.reception_upper/2)**2)
    corner_x=far_x/2+height*math.sin(delta)
    x=np.sort(np.r_[np.linspace(left,radius,1000),corner_x])
    y=np.minimum(np.sqrt(np.maximum(0,radius**2-x*x)),-far_y+np.sqrt(np.maximum(0,radius**2-(x-far_x)**2)))
    curve=np.vstack([np.column_stack([x,y]),np.column_stack([x[::-1],-y[::-1]])]); lens=lens_boundary()
    candidates=to_local(analytic_candidates(region),region)
    fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
    for ax in axes:
        ax.fill(*lens.T,color=ORANGE,alpha=.18,label="中心线近似透镜")
        ax.plot(*close_curve(lens).T,color=ORANGE,linewidth=1.3)
        ax.fill(*curve.T,color=TEAL,alpha=.2,label="完整可行域得到的可靠接收域 K")
        ax.plot(*close_curve(curve).T,color=TEAL,linewidth=2)
        ax.scatter(*candidates.T,color=RED,marker="x",s=70,label="未经修正的解析候选",zorder=5)
        ax.axhline(0,color=GRAY,linestyle="--",linewidth=.8)
        label_axes(ax); ax.grid(alpha=.15)
    axes[0].set_xlim(470,1030); axes[0].set_ylim(-720,720); axes[0].set_aspect("equal")
    axes[0].set_title("完整角域使可靠域进一步收缩"); axes[0].legend(fontsize=8,loc="lower right")
    axes[1].set_xlim(730,790); axes[1].set_ylim(620,680); axes[1].set_aspect("equal")
    axes[1].set_title("上侧解析候选附近放大")
    maximum=region.max_distance(analytic_candidates(region)[0])
    axes[1].annotate(f"解析点最远距离\n{maximum:.3f} 米 > 1000 米",candidates[0],xytext=(758,671),fontsize=9,
        arrowprops={"arrowstyle":"->","color":RED},ha="center",color=RED)
    fig.suptitle("可靠二次检测域由整个连续物理可行域确定",fontsize=16)
    footer(fig,"标准 K 用三圆交解析边界加密；K 是统一 1000 米准则的充分保证域\nK 外不等于利用首次成功信息后无法保证接收；本文测向选点还须远离目标域超过 5 米")
    save_figure(fig,"04_可靠二次检测域")


def search_overlays(ax,run,show_analytic=True,show_reference_line=True):
    region=run["region"]; result=run["search"]
    boundary=to_local(run["reliable_boundary"],region)
    ax.plot(*close_curve(boundary).T,color=INK,lw=1.2,label="可靠接收域边界")
    if show_reference_line:
        ax.axhline(0,color=GRAY,ls="--",lw=.8)
    if show_analytic:
        analytic=to_local(result["analytic_candidates"],region)
        ax.scatter(*analytic.T,marker="x",color=RED,s=45,label="中心线解析候选",zorder=6)
    optima=to_local([row["station"] for row in result["side_optima"]],region)
    ax.scatter(*optima.T,marker="*",color=ORANGE,edgecolor=INK,linewidth=.5,s=120,label="两侧数值最优",zorder=7)
    ax.set_xlim(boundary[:,0].min()-55,boundary[:,0].max()+55)
    ax.set_ylim(boundary[:,1].min()-55,boundary[:,1].max()+55)
    label_axes(ax); ax.set_aspect("equal")


def figure_heatmaps(data):
    """热力图同时呈现首次观测和三类边界；圆心须转换到各案例局部坐标。"""
    fig,axes=plt.subplots(2,2,figsize=(17,14),constrained_layout=True)
    for ax,run in zip(axes.flat,data["runs"]):
        region=run["region"]
        grid=run["search"]["grid"]
        diameters=np.asarray(grid["diameter_m"],dtype=float)
        finite=diameters[np.isfinite(diameters)]
        best=run["search"]["best"]["diameter_m"]
        ceiling=max(best*1.1,float(np.percentile(finite,90)))
        mesh=ax.pcolormesh(grid["x_local_m"],grid["y_local_m"],np.ma.masked_invalid(diameters),
            shading="auto",cmap="YlGnBu_r",norm=LogNorm(vmin=best,vmax=ceiling),rasterized=True)
        colorbar=fig.colorbar(mesh,ax=ax,pad=.02,shrink=.8,extend="max")
        colorbar.set_label("最坏定位直径（米，对数色标）")
        search_overlays(ax,run,show_reference_line=False)

        # 目标圆限制未知目标，不限制检测站；三类边界用不同颜色区分。
        target_center=to_local(region.target_center,region)
        ax.add_patch(Circle(target_center,region.target_radius,fill=False,
            edgecolor="#97528D",linestyle=(0,(6,3)),linewidth=1.5,zorder=4,
            label="1800 米目标区域边界"))
        outline=physical_outline(region,local=True)
        ax.fill(*outline.T,color=TEAL,alpha=.12,zorder=3)
        ax.plot(*close_curve(outline).T,color=TEAL,linewidth=1.4,zorder=5,
            label="首次目标可行域边界（含 ±1°）")

        boundary=to_local(run["reliable_boundary"],region)
        # 同时包含第一站、完整物理域、可靠域和目标圆的前向交点；
        # 保留等比例局部视图，不把整圆强行塞入画面而压缩热力图。
        right=max(float(boundary[:,0].max()),float(outline[:,0].max()),0.)
        if abs(target_center[1])<=region.target_radius:
            forward_hit=target_center[0]+math.sqrt(max(0.,region.target_radius**2-target_center[1]**2))
            right=max(right,float(forward_hit))
        left=min(float(boundary[:,0].min()),float(outline[:,0].min()),0.)
        x_margin=max(90.,(right-left)*.07)
        lower=min(float(boundary[:,1].min()),float(outline[:,1].min()),0.)
        upper=max(float(boundary[:,1].max()),float(outline[:,1].max()),0.)
        ax.set_xlim(left-x_margin,right+x_margin)
        ax.set_ylim(lower-100.,upper+100.)

        ray_end=right+.55*x_margin
        ax.plot([0,ray_end],[0,0],color=BLUE,linestyle="--",linewidth=1.3,zorder=6,
            label="首次观测方向射线")
        ax.annotate("",xy=(ray_end,0),xytext=(ray_end-.12*(right-left),0),
            arrowprops={"arrowstyle":"-|>","color":BLUE,"lw":1.3},zorder=6)
        ax.scatter([0],[0],s=75,color=INK,edgecolor="white",linewidth=1.1,zorder=9,
            label="第一观测点 $S_1$（局部原点）")
        ax.annotate("$S_1$",(0,0),xytext=(-12,12),textcoords="offset points",
            color=INK,fontsize=11,fontweight="bold",zorder=10,
            bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1.5})
        station=region.station
        ax.set_title(f"{run['case']['name_zh']}｜当前数值最优 {best:.3f} 米\n"
            f"第一站全局坐标 ({station[0]:g}, {station[1]:g}) 米；示向度 {region.bearing_deg:g}°",
            fontsize=11)
        ax.set_facecolor("#F3F4F5")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.063),ncol=4,
        fontsize=9,frameon=True,columnspacing=1.5,handlelength=2.6)
    fig.suptitle("最坏定位直径热力图：目标边界、首次观测与第二站候选位置",fontsize=18)
    footer(fig,"各图采用以第一站为原点、首次示向为横轴的等比例局部坐标；目标圆仅约束目标位置\n"
        "灰白区无有限可靠测向指标；每幅独立对数色标，上端超过 90% 样本分位的值用顶端颜色显示")
    # rect 的末两项为宽、高；顶部不超过1，避免总标题落进子图。
    fig.get_layout_engine().set(rect=(0,.15,1,.85))
    save_figure(fig,"05_最坏定位直径热力图")


def figure_candidates(data):
    fig,axes=plt.subplots(1,2,figsize=(13,8),constrained_layout=True)
    for ax,run in zip(axes,[data["runs"][0],data["runs"][-1]]):
        result=run["search"]
        for key,color,size in [("0.10","#DDEBE7",14),("0.05",TEAL,12),("0.02",BLUE,9)]:
            info=result["candidate_regions"][key]
            points=[result["evaluations"][index]["station"] for index in info["point_indices"]]
            if points:
                local=to_local(points,run["region"])
                ax.scatter(*local.T,s=size,color=color,marker="s",alpha=.75,label=f"恶化不超过 {100*info['eta']:.0f}% 的已评估点")
        search_overlays(ax,run,False)
        selected=[result["evaluations"][index]["station"] for index in result["candidate_regions"]["0.10"]["point_indices"]]
        if selected:
            points=to_local(selected,run["region"])
            margin=np.maximum(np.ptp(points,axis=0)*.15,[15,20])
            ax.set_xlim(points[:,0].min()-margin[0],points[:,0].max()+margin[0])
            ax.set_ylim(points[:,1].min()-margin[1],points[:,1].max()+margin[1])
            ax.set_aspect("auto")
        count=result["candidate_regions"]["0.05"]["point_count"]
        ax.set_title(f"{run['case']['name_zh']}\n5% 阈值保留 {count} 个已评估候选点")
        ax.legend(fontsize=8,loc="center right")
    fig.suptitle("近最优候选区域：保留两个侧向选择，供后续路径决策使用",fontsize=16)
    footer(fig,"候选点局部放大，纵横比例不同；散点不是连续边界，面积另用统一规则格网估计")
    save_figure(fig,"06_近最优候选区域")


def figure_angles(data):
    fig,ax=plt.subplots(figsize=(10,6))
    for distance,color in [(500,TEAL),(1000,BLUE)]:
        rows=sorted([row for row in data["angle_geometry"] if row["distance_m"]==distance],key=lambda r:r["angle_deg"])
        ax.plot([r["angle_deg"] for r in rows],[r["diameter_m"] for r in rows],color=color,lw=2,label=f"两站到目标距离均为 {distance} 米")
    ax.axvline(90,color=ORANGE,ls="--",label="90° 交会")
    ax.set_yscale("log"); ax.set_xticks([5,30,60,90,120,150,175])
    ax.set_xlabel("两条中心示向射线的交会角（度）"); ax.set_ylabel("定位区域直径（米，对数刻度）")
    ax.set_title("角误差固定为 1° 时：交会角和站源距离共同决定定位精度")
    ax.grid(alpha=.2); ax.legend()
    footer(fig,"1000 米曲线复用第一问归档结果；500 米曲线重新调用第一问算法")
    save_figure(fig,"07_交会角与定位直径关系")


def figure_sensitivity(data):
    fig,axes=plt.subplots(1,3,figsize=(16,5.4),constrained_layout=True)
    rows=data["angle_sensitivity"]; x=[r["half_width_deg"] for r in rows]
    axes[0].plot(x,[r["fixed_strategy"]["diameter_m"] for r in rows],"o--",color=GRAY,label="固定题设 1° 最优位置")
    axes[0].plot(x,[r["optimized_strategy"]["diameter_m"] for r in rows],"o-",color=TEAL,label="按扰动条件重新优化")
    for row in rows:
        if not row["fixed_strategy"]["reliable_bearing"]:
            axes[0].scatter(row["half_width_deg"],row["fixed_strategy"]["diameter_m"],marker="x",s=85,color=RED,zorder=4)
    axes[0].axvline(1,color=ORANGE,ls=":",label="题设半宽 1°")
    axes[0].set_xlabel("角误差半宽（度）"); axes[0].set_ylabel("最坏定位直径（米）")
    axes[0].set_title("角度误差敏感性"); axes[0].legend(fontsize=8)
    unreliable=any(not row["fixed_strategy"]["reliable_bearing"] for row in rows)
    axes[0].text(.03,.96,"红叉：固定位置未过本文测向准则" if unreliable else "该范围内固定位置均通过本文测向准则",transform=axes[0].transAxes,va="top",fontsize=8,color=RED if unreliable else TEAL)
    rows=data["distances"]
    axes[1].plot([r["centerline_interval_m"][1] for r in rows],[r["best"]["diameter_m"] for r in rows],"o-",color=BLUE)
    axes[1].set_xlabel("首次中心线可能距离上限（米）"); axes[1].set_ylabel("重新优化后的最坏直径（米）")
    axes[1].set_title("目标圆裁剪与距离先验")
    rows=data["radii"]
    axes[2].plot([r["guaranteed_radius_m"] for r in rows],[r["best"]["diameter_m"] for r in rows],"o-",color=TEAL)
    axes[2].axvline(1000,color=ORANGE,ls=":",label="题设下界 1000 米")
    axes[2].set_xlabel("保证接收半径下界（米）"); axes[2].set_ylabel("重新优化后的最坏直径（米）")
    axes[2].set_title("可靠接收半径的理论扰动"); axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle("误差和物理先验变化时，重新检验可行性并优化检测位置",fontsize=16)
    footer(fig,"非题设参数仅用于理论敏感性分析；正式值为 1°、1000 米")
    save_figure(fig,"08_角度误差敏感性")


def figure_baselines(data):
    rows=data["baselines"]; fig,ax=plt.subplots(figsize=(12,7))
    finite=[r["diameter_m"] for r in rows if finite_value(r["diameter_m"])]
    maximum=max(finite)*1.3
    for index,row in enumerate(rows):
        value=row["diameter_m"]
        color=TEAL if row["reliable_bearing"] else "#D6DBDF"
        if finite_value(value):
            ax.barh(index,value,color=color,height=.62,hatch=None if row["reliable_bearing"] else "//",edgecolor=GRAY,linewidth=.3)
            ax.text(value+maximum*.012,index,f"{value:.3f} 米"+("（未过本文准则）" if not row["reliable_detection"] else ""),va="center",fontsize=9,color=INK)
        else:
            ax.scatter(maximum*.03,index,marker="x",color=RED,s=55)
            ax.text(maximum*.075,index,"存在无法获得示向度的目标情形；不作为有限可靠方案",va="center",fontsize=9,color=RED)
    ax.set_yticks(range(len(rows)),[r["strategy_zh"] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0,maximum); ax.set_xlabel("最坏定位区域直径（米）")
    ax.set_title("策略对照：在统一 1000 米充分保证准则下比较最坏定位直径")
    ax.grid(axis="x",alpha=.18)
    footer(fig,"青色通过本文可靠测向准则；灰色斜纹未通过保守的 1000 米准则\n未通过本文准则不等于无法利用首次成功信息，在更大条件可靠域内获得保证")
    save_figure(fig,"09_解析策略与数值优化对比")


def figure_convergence(data):
    convergence=data["convergence"]
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),constrained_layout=True)
    rows=convergence["angle_grid"]
    axes[0].plot([r["angle_count"] for r in rows],[r["evaluation"]["diameter_m"] for r in rows],"o-",color=BLUE)
    axes[0].set_xscale("log",base=2); axes[0].set_xlabel("连续观测角区间的离散点数")
    axes[0].set_ylabel("最坏定位直径（米）"); axes[0].set_title("内层观测角加密")
    rows=convergence["target_error_grid"]
    axes[1].plot([r["scenario_count"] for r in rows],[r["evaluation"]["diameter_m"] for r in rows],"o-",color=TEAL)
    axes[1].set_xscale("log"); axes[1].set_xlabel("目标位置 × 误差的场景数")
    axes[1].set_ylabel("场景最大定位直径（米）"); axes[1].set_title("独立二维目标与误差核查")
    rows=convergence["station_grid"]
    x=[r["grid_step_m"] for r in rows]
    axes[2].plot(x,[r["best"]["diameter_m"] for r in rows],"o-",color=BLUE,label="数值最优直径")
    values=[r["best"]["diameter_m"] for r in rows]
    margin=max(1.0,0.02*max(values))
    axes[2].set_ylim(min(values)-margin,max(values)+margin)
    twin=axes[2].twinx(); twin.plot(x,[r["candidate_regions"]["0.05"]["estimated_area_m2"] for r in rows],"s--",color=ORANGE,label="5% 候选面积估计")
    twin.set_ylabel("5% 候选面积估计（平方米）",color=ORANGE)
    axes[2].invert_xaxis(); axes[2].set_xlabel("外层格网步长（米，由粗到细）")
    axes[2].set_ylabel("最坏定位直径（米）",color=BLUE); axes[2].set_title("最优值与候选面积分别核查")
    axes[2].legend(fontsize=8,loc="upper left"); twin.legend(fontsize=8,loc="lower left")
    for ax in axes: ax.grid(alpha=.18); ax.ticklabel_format(axis="y",style="plain",useOffset=False)
    fig.suptitle("采样收敛分析：稳定的最优值不等于已经精确的候选区域面积",fontsize=16)
    footer(fig,"有限网格与局部优化结果，不构成连续全局最优证明")
    save_figure(fig,"10_采样与搜索收敛分析")


def write_reports(configuration,data,font_path,plots_only):
    summary=["# 第二检测候选点汇总", "", NOTICE + "。坐标均为题面全局坐标；所有长度单位为米。", "",
        "| 构造案例 | 数值最优点 | 最坏定位直径 | 移动距离 | 可靠域面积近似（平方米） | 5%候选面积估计（平方米） |",
        "|---|---|---:|---:|---:|---:|"]
    details=[]
    for run in data["runs"]:
        result=run["search"]; best=result["best"]; point=best["station"]
        summary.append(f"| {run['case']['name_zh']} | ({point[0]:.3f}, {point[1]:.3f}) | {best['diameter_m']:.6f} | {best['movement_distance_m']:.3f} | {run['reliable_area_m2']:.1f} | {result['candidate_regions']['0.05']['estimated_area_m2']:.1f} |")
        details += ["",f"## {run['case']['name_zh']}","", "| 侧向数值解 | 坐标 | 最坏直径 | 连续最远距离 | 连续最近距离 |", "|---|---|---:|---:|---:|"]
        for index,row in enumerate(result["side_optima"]):
            s=row["station"]
            details.append(f"| 第{index+1}侧 | ({s[0]:.6f}, {s[1]:.6f}) | {row['diameter_m']:.6f} | {row['max_distance_m']:.6f} | {row['min_distance_m']:.6f} |")
        details += ["", "| 近优阈值 | 已评估候选点数 | 规则格网面积估计 | 候选移动距离范围 |", "|---|---:|---:|---|"]
        for info in result["candidate_regions"].values():
            movement=info["movement_range_m"]
            text="无" if movement is None else f"{movement[0]:.3f}～{movement[1]:.3f}"
            details.append(f"| {info['eta']:.0%} | {info['point_count']} | {info['estimated_area_m2']:.1f} | {text} |")
    summary += ["", "可靠域面积来自 256 点内接边界多边形；候选域面积来自统一格网中心计数，均为近似值。局部细化点参与候选点集和移动距离范围，不参与格网面积计数。", *details,
        "", "第三问可直接读取 `鲁棒搜索输入输出.json` 中的 `search.evaluations`，再按 `candidate_regions` 的 `point_indices` 提取候选坐标、鲁棒直径和移动距离。", ""]
    (TABLES/"第二检测候选点汇总.md").write_text("\n".join(summary),encoding="utf-8",newline="\n")
    strategy=["# 第二问策略对比", "", NOTICE+"。本文采用统一1000米充分保证准则；未通过该准则的策略保留几何指标作对照。", "",
        "| 策略 | 第二检测点 | 最坏直径（米） | 通过1000米准则 | 通过本文测向准则 | 连续最远距离（米） | 移动距离（米） |",
        "|---|---|---:|---|---|---:|---:|"]
    for row in data["baselines"]:
        point=row["station"]
        strategy.append(f"| {row['strategy_zh']} | ({point[0]:.3f}, {point[1]:.3f}) | {number(row['diameter_m'],6)} | {'是' if row['reliable_detection'] else '否'} | {'是' if row['reliable_bearing'] else '否'} | {row['max_distance_m']:.3f} | {row['movement_distance_m']:.3f} |")
    strategy += ["", "首次检测成功还意味着实际接收半径至少为第一站到真实目标的距离。因此，本文统一1000米域是充分保证域，未通过本表准则不能直接推断在利用首次成功信息后的更大条件可靠域内仍无法保证接收。", "", "沿首次中心线布站虽然可能保证接收，却会遇到目标过近而没有示向度的情形；另有合法同向双站场景使第一问角域交集无界。", "", "本表采用确定性最坏值，无目标均匀分布或误差正态分布假设。", ""]
    (TABLES/"策略对比汇总.md").write_text("\n".join(strategy),encoding="utf-8",newline="\n")
    best=data["runs"][0]["search"]["best"]
    plot_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    compute_manifest_path=EXPERIMENTS/"计算配置与来源.json"
    compute_manifest=json.loads(compute_manifest_path.read_text(encoding="utf-8")) if compute_manifest_path.exists() else None
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    metadata={"来源":NOTICE,"执行模式":"仅重画已归档计算结果" if plots_only else "完整计算与图表构建",
        "Python":platform.python_version(),"依赖版本":{"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
        "中文字体":font_path,"绘图代码SHA256":{"scripts/build_q2_assets.py":plot_hash},"执行时Git_HEAD":head,
        "计算来源记录":compute_manifest,"版本说明":"计算来源单独保存于计算配置与来源.json；仅重画不替换其计算源码版本。绘图SHA256标识本次绘图代码。Git_HEAD仅记录当时已提交版本。",
        "configuration":configuration,"主构造案例数量":len(data["runs"]),"图表数量":10,"图表格式":["PNG 220dpi","SVG 字形转路径"],
        "各案例实际规模":[{"id":run["case"]["id"],"target_count":len(run["targets"]),"evaluation_count":len(run["search"]["evaluations"]),
                         "search_configuration":run["search"]["configuration"]} for run in data["runs"]],
        "归档搜索结果SHA256":hashlib.sha256((TABLES/"鲁棒搜索输入输出.json").read_bytes()).hexdigest(),
        "验证说明":"自动测试数量以 docs/第二问/验证与复现记录.md 的实际统一验收输出为准。"}
    path=EXPERIMENTS/"实验配置与环境.json"
    write_json(path,metadata)
    report=["# 第二问构造实验报告","",NOTICE+"。本轮脚本不调用官方模拟器，也不产生正式比赛成绩。","",
        "## 1. 实验模型与解释边界","",
        "真实目标场景采用首次窄角域与1500米接收圆、1800米目标圆的连续交集。可靠接收约束要求第二站到整个物理可行域的最远距离不超过1000米，并向内保留数值安全裕度。由于题面五米内没有示向度，正式选点再要求最近距离至少为五米加正安全裕度，因此严格超过五米。","",
        "首次成功还蕴含实际接收半径至少为第一站到真实目标的距离。因此，本文统一1000米准则采用的是保守充分域，未通过这一准则不等于在利用首次成功信息的更大条件可靠域内无法保证接收。主优化结果仅在本文明确选择的充分域内比较。","",
        "定位质量沿用第一问纯角域交集的直径。利用凸物理域的连续可见方向区间消去目标位置维度，在第二示向度区间上加密求最坏值；外围用规则格网、两侧局部格网和受约束连续微调搜索。所有数值最优结果均经过第一问原始求解器复核最坏场景。","",
        "## 2. 主构造结果","",*summary[4:6+len(data['runs'])],"",
        f"标准构造在本文充分域内的当前数值最优直径为 **{best['diameter_m']:.6f} 米**。中心线解析候选为 (750, ±{analytic_candidates(data['runs'][0]['region'])[0,1]:.6f})，其连续最远目标距离为{data['runs'][0]['region'].max_distance(analytic_candidates(data['runs'][0]['region'])[0]):.6f}米，未通过本文统一1000米准则；这不否定它在更大条件可靠域中的接收保证。策略对照表同时保存解析回缩和最终数值优化结果。","",
        "## 3. 收敛与敏感性","",
        "| 边界预算 | 目标点数 | 误差数 | 场景总数 | 场景最坏直径 | 可靠域内接面积 |","|---:|---:|---:|---:|---:|---:|"]
    for row in data["convergence"]["target_error_grid"]:
        report.append(f"| {row['boundary_budget']} | {row['target_count']} | {row['error_count']} | {row['scenario_count']} | {row['evaluation']['diameter_m']:.9f} | {row['K_polygon_area_m2']:.3f} |")
    report += ["", "观测角加密、目标×误差采样和外层格网加密均归档于 `../../results/tables/第二问/采样收敛分析.json`。近优面积是独立的数值量，应观察其随格网变化的幅度，不能因为最优直径稳定便宣称面积精确。","",
        "灵敏度涵盖误差半宽0.5°、1°、1.5°、2°，保证接收下界950、1000、1050米，以及中心线距离上限600、900、1200、1500米。非题设参数只作理论扰动。" + ("固定原方案在部分扰动下未通过本文可靠测向准则，图中用红叉标出。" if any(not row["fixed_strategy"]["reliable_bearing"] for row in data["angle_sensitivity"]) else "本次四种误差半宽下，固定题设最优位置均通过本文可靠测向准则；固定与重新优化曲线非常接近。"),"",
        "## 4. 文件与复现入口","",
        "完整验收运行 `pwsh -NoProfile -File scripts/run-q2.ps1`；仅计算及生成资料可运行 `.venv/Scripts/python.exe -X utf8 scripts/build_q2_assets.py`；仅从归档JSON重画可附加 `--plots-only`。","",
        "- `../../results/figures/第二问/`：10张中文图，每张提供PNG和SVG。",
        "- `../../results/tables/第二问/`：全部主搜索输入输出、候选点索引、策略表、收敛和灵敏度原始数据。",
        "- `../../docs/第二问/`：完整推导、阅读导航、实际验收记录。",
        "- `../../paper/sections/第二问_论文备用稿.md`：可供论文改写的模型与结果。","",
        "可靠域图中的边界连线与候选散点是连续集合的数值呈现；候选点按连续物理约束逐点检验，候选区域本身不被伪装为经过严格全局认证的连续集合。","",
        "执行环境及源代码哈希见同目录 `实验配置与环境.json`。"," "]
    (EXPERIMENTS/"构造实验报告.md").write_text("\n".join(report).rstrip()+"\n",encoding="utf-8",newline="\n")


def main():
    parser=argparse.ArgumentParser(description="第二问中文构造实验、图表和论文取材资料构建。")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/q2_cases.json",help="构造案例配置JSON")
    parser.add_argument("--plots-only",action="store_true",help="仅加载已经归档的计算结果重画图表")
    args=parser.parse_args()
    for directory in [FIGURES,TABLES,EXPERIMENTS]: directory.mkdir(parents=True,exist_ok=True)
    font_path=setup_style()
    if args.plots_only:
        configuration,data=load_archived()
    else:
        configuration=json.loads(args.config.read_text(encoding="utf-8-sig"))
        data=run_experiments(configuration)
    figure_flow(); figure_physical(data); figure_lens(data); figure_reliable(data)
    figure_heatmaps(data); figure_candidates(data); figure_angles(data)
    figure_sensitivity(data); figure_baselines(data); figure_convergence(data)
    write_reports(configuration,data,font_path,args.plots_only)
    print(f"第二问10组中文图表及实验报告已生成：{FIGURES}",flush=True)


if __name__ == "__main__":
    main()
