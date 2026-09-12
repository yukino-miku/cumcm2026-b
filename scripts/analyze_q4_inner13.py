"""核算第四问圆内13站的覆盖反例和最短遍历路线；不运行机器人策略。"""
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/tables/第四问/初步覆盖构造核算.json'
OUTPUT = ROOT / 'results/tables/第四问/圆内十三站_覆盖与最短路线.json'
FIG = ROOT / 'results/figures/第四问/圆内十三站_最短路线与覆盖反例'


def shortest_length(points):
    """独立用子集动态规划核对：第一个点出发，终点自由，不返回。"""
    n = len(points) - 1
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    costs = np.full((1 << n, n), np.inf)
    for j in range(n):
        costs[1 << j, j] = distances[0, j+1]
    for mask in range(1, 1 << n):
        for j in range(n):
            if not mask & (1 << j):
                continue
            previous = mask ^ (1 << j)
            if previous:
                indices = [k for k in range(n) if previous & (1 << k)]
                costs[mask, j] = min(costs[previous, k] + distances[k+1, j+1] for k in indices)
    return float(costs[-1].min())


def calculate():
    source = json.loads(INPUT.read_text(encoding='utf-8'))
    points = np.array(source['站点坐标_米'])
    inside = points[np.linalg.norm(points, axis=1) <= 1800 + 1e-8]
    a = float(source['三角网间距_米'])
    h = a * math.sqrt(3) / 2
    path = np.array([(0, 0), (a, 0), (1.5*a, h), (.5*a, h), (0, 2*h),
                     (-.5*a, h), (-1.5*a, h), (-a, 0), (-1.5*a, -h),
                     (-.5*a, -h), (0, -2*h), (.5*a, -h), (1.5*a, -h)])
    assert len(inside) == len(path) == 13
    # 双向检查路径和来源中的圆内站集合一致，且没有遗漏或重复。
    distances_to_input = np.linalg.norm(path[:, None, :] - inside[None, :, :], axis=2)
    assert (distances_to_input.min(axis=0) < 1e-7).all()
    assert (distances_to_input.min(axis=1) < 1e-7).all()
    pairwise = np.linalg.norm(path[:, None, :] - path[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    lower_bound = 12 * float(pairwise.min())
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    optimum = shortest_length(path)
    assert np.allclose(segments, 950)
    assert abs(segments.sum() - lower_bound) < 1e-7
    assert abs(optimum - lower_bound) < 1e-7

    north = np.array([0., 1800.])
    north_distances = np.linalg.norm(path - north, axis=1)
    east = np.array([1700., 0.])  # 严格位于目标圆盘内部，排除仅靠边界退化的解释。
    east_distances = np.linalg.norm(path - east, axis=1)
    projections = (path - east) @ np.array([1., 0.])
    assert (north_distances <= 1000).sum() == 1
    assert (projections < 0).all()
    assert (east_distances <= 1000).sum() == 3
    north_order = np.argsort(north_distances)

    # 附图红线按最北端至最南端顺次理解：10条950米边和2条sqrt(3)*950米边。
    # 附图未指定中心出发后的接入方式，下列另列一种直接接到端点的解释，不声称唯一。
    red = path[[4, 6, 5, 3, 2, 1, 0, 7, 8, 9, 11, 12, 10]]
    red_length = float(np.linalg.norm(np.diff(red, axis=0), axis=1).sum())
    return {
        '性质': '第四问待审核几何分析；无策略实现、模拟器运行或实际耗时成绩',
        '来源_SHA256': {'results/tables/第四问/初步覆盖构造核算.json': sha256(INPUT.read_bytes()).hexdigest(),
                        'scripts/analyze_q4_inner13.py': sha256(Path(__file__).read_bytes()).hexdigest()},
        '路线条件': '从圆心出发，自由终点，不要求返回；只访问固定站，尚未插入补测、清除或外围补漏',
        '路线站点坐标_按1至13顺序': path.tolist(),
        '每段长度_米': segments.tolist(),
        '任意两站最小间距_米': float(pairwise.min()),
        '长度下界_米': lower_bound,
        '构造路线长度_米': float(segments.sum()),
        '子集动态规划最短长度_米': optimum,
        '纯移动虚拟时间_秒': optimum / 5,
        '原图红线解释': {'红线本身长度_米': red_length,
                          '中心直接接到最北端后完整走红线_米': float(np.linalg.norm(red[0]) + red_length),
                          '备注': '只比较一种接入方式；用户附图未指定起终点，无需强制按此执行'},
        '全向双覆盖反例': {'源位置': north.tolist(), '接收半径_米': 1000,
                           '1000米内站数': 1,
                           '最近三站坐标': path[north_order[:3]].tolist(),
                           '最近三站距离_米': north_distances[north_order[:3]].tolist()},
        '定向发现反例': {'源位置': east.tolist(), '发射轴单位向量': [1, 0],
                         '1000米内站数': int((east_distances <= 1000).sum()),
                         '所有站在发射轴上的相对投影_米': projections.tolist(),
                         '任何允许半径下的实际接收站数': 0},
    }


def plot(data):
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 11,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'q4-inner13-analysis'})
    path = np.array(data['路线站点坐标_按1至13顺序'])
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.7))
    for ax in axes:
        ax.set(aspect='equal', xlim=(-2050, 2850), ylim=(-2100, 2450),
               xlabel='东向 x（米）', ylabel='北向 y（米）')
        ax.add_patch(Circle((0, 0), 1800, fill=False, color='#85639d', ls='--', lw=1.7))
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    ax = axes[0]
    ax.set_xlim(-2050, 2050)
    ax.plot(path[:, 0], path[:, 1], color='#318a83', lw=1.8, zorder=2)
    ax.scatter(path[:, 0], path[:, 1], color='#315363', s=42, zorder=4)
    for i, point in enumerate(path):
        ax.annotate(str(i+1), point, xytext=(7, 6), textcoords='offset points',
                    fontsize=11, fontweight='bold', zorder=6)
    for a, b in zip(path[:-1], path[1:]):
        v = b-a
        ax.annotate('', a+.61*v, a+.44*v,
                    arrowprops={'arrowstyle': '-|>', 'color': '#318a83', 'lw': 1.6})
    ax.scatter(0, 0, marker='*', s=150, color='#d28632', zorder=5)
    ax.scatter(*path[-1], marker='D', s=60, facecolors='none', edgecolors='#b64b40', lw=1.6, zorder=5)
    ax.set_title('从1号圆心依次走到13号：最短11.4公里', pad=16)

    ax = axes[1]
    ax.scatter(path[:, 0], path[:, 1], color='#315363', s=32, zorder=4)
    north = np.array(data['全向双覆盖反例']['源位置'])
    east = np.array(data['定向发现反例']['源位置'])
    ax.add_patch(Wedge(east, 1000, -90, 90, facecolor='#f1d3b1', edgecolor='#c0803d', alpha=.8,
                       label='向东发射的1000米半圆接收区'))
    ax.scatter(*north, marker='*', s=105, color='#b94b48', zorder=5)
    ax.scatter(*east, marker='*', s=105, color='#b94b48', zorder=5)
    for q in np.array(data['全向双覆盖反例']['最近三站坐标']):
        ax.plot([north[0], q[0]], [north[1], q[1]], color='#b94b48', ls=':', lw=1.5)
    ax.annotate('北端：最近154.6米\n次近1086.6米，不能保证双覆盖', north,
                xytext=(-1850, 2130), fontsize=10,
                arrowprops={'arrowstyle': '->', 'color': '#b94b48'})
    ax.annotate('源(1700,0)朝东发射\n13站全部在背侧', east,
                xytext=(-450, -1400), fontsize=10,
                arrowprops={'arrowstyle': '->', 'color': '#b94b48'})
    ax.set_title('13站存在双覆盖缺口，也可能漏掉定向源', pad=16)
    ax.legend(loc='lower left', fontsize=9)
    fig.suptitle('第四问圆内13站：路线可以最短，发现保障仍须补充', y=.985, fontsize=16)
    fig.text(.5, .045, '几何核算；不返回圆心，不含补测、清除及外围补漏；最短遍历固定站不等于全任务最短耗时。',
             ha='center', fontsize=10, color='#56636d')
    fig.tight_layout(rect=(0, .08, 1, .95))
    for suffix in ['.png', '.svg']:
        fig.savefig(FIG.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    data = calculate()
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plot(data)
    print(json.dumps({k: data[k] for k in ['构造路线长度_米', '子集动态规划最短长度_米',
        '全向双覆盖反例', '定向发现反例']}, ensure_ascii=False, indent=2))
