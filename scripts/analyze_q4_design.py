"""第四问待审核方案的几何核算与中文示意图；不执行搜索策略或模拟器。"""
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/tables/第四问/初步覆盖构造核算.json'
FIG = ROOT / 'results/figures/第四问/定向源发现与三角网备选'
SPACING = 950.0


def triangle_distance(triangle, point):
    """点到闭三角形距离，供选出与目标圆盘相交的网格单元。"""
    p = np.asarray(triangle, float)
    q = np.asarray(point, float)
    edges = np.roll(p, -1, axis=0) - p
    offsets = q - p
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    if np.all(cross >= -1e-8) or np.all(cross <= 1e-8):
        return 0.0
    t = np.clip((offsets * edges).sum(axis=1) / (edges * edges).sum(axis=1), 0, 1)
    return float(np.linalg.norm(q - (p + t[:, None] * edges), axis=1).min())


def lattice_point(index):
    i, j = index
    return SPACING * np.array([i + j / 2, math.sqrt(3) * j / 2])


def calculate():
    triangles = []
    indices = set()
    # 无限网格中遗漏单元的顶点至少有一个整数坐标绝对值>=6；
    # 其径向下界950*sqrt(3)*6/2，减去三角形直径950仍大于1800。
    for i in range(-6, 6):
        for j in range(-6, 6):
            cells = [((i, j), (i+1, j), (i, j+1)),
                     ((i+1, j), (i+1, j+1), (i, j+1))]
            for cell in cells:
                triangle = np.array([lattice_point(k) for k in cell])
                if triangle_distance(triangle, (0, 0)) <= 1800 + 1e-7:
                    triangles.append(triangle)
                    indices.update(cell)
    indices = sorted(indices)
    stations = np.array([lattice_point(k) for k in indices])
    triangles = np.array(triangles)
    side_lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    assert np.allclose(side_lengths, SPACING)
    assert SPACING * math.sqrt(3) * 6 / 2 - SPACING > 1800
    assert len(stations) == 31 and len(triangles) == 42

    # 独立的采样交叉核查：999.5米以内各站相对源位置的最大圆周角间隙。
    # 这只是数值诊断；连续保证来自三角形的凸组合和直径界。
    grid = np.arange(-1800, 1801, 30)
    xx, yy = np.meshgrid(grid, grid)
    samples = np.column_stack((xx.ravel(), yy.ravel()))
    samples = samples[np.linalg.norm(samples, axis=1) <= 1800 + 1e-8]
    angles = np.arange(720) * 2 * math.pi / 720
    samples = np.vstack((samples, 1800 * np.column_stack((np.cos(angles), np.sin(angles)))))
    worst_gap = 0.0
    for g in samples:
        vectors = stations - g
        distances = np.linalg.norm(vectors, axis=1)
        vectors = vectors[(distances <= 999.5) & (distances > 1e-8)]
        bearings = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
        assert len(bearings) >= 2
        gap = float(np.diff(np.r_[bearings, bearings[0] + 2 * math.pi]).max())
        assert gap <= math.pi + 1e-8
        worst_gap = max(worst_gap, gap)

    old_angles = np.arange(6) * math.pi / 3
    old_stations = np.vstack((np.zeros((1, 2)),
        1200 * np.column_stack((np.cos(old_angles), np.sin(old_angles)))))
    source = np.array([1800.0, 0.0])
    outward_axis = np.array([1.0, 0.0])
    signed = (old_stations - source) @ outward_axis
    assert (signed < 0).all()
    inputs = ['materials/original/problem/B题.pdf',
              'materials/original/problem/附件/附件1.docx',
              'materials/original/problem/附件/附件2.docx',
              'scripts/analyze_q4_design.py']
    return {
        '性质': '第四问待审核思路的几何构造与数值复核，不是策略运行或官方测试',
        '来源_SHA256': {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs},
        '三角网间距_米': SPACING,
        '三角形数量': len(triangles),
        '站点数量': len(stations),
        '区域外站点数量': int((np.linalg.norm(stations, axis=1) > 1800 + 1e-8).sum()),
        '最远站点离圆心_米': float(np.linalg.norm(stations, axis=1).max()),
        '站点整数索引': indices,
        '站点坐标_米': stations.tolist(),
        '相交三角形顶点_米': triangles.tolist(),
        '采样复核': {'点数_含重复边界': len(samples), '保守检测距离_米': 999.5,
                     '最大圆周角间隙_度': math.degrees(worst_gap),
                     '超过180度的失败数': 0,
                     '说明': '采样不代替连续证明；重合站不作为有方向的覆盖证据'},
        '七站反例': {'源位置': source.tolist(), '发射轴': outward_axis.tolist(),
                    '原七站': old_stations.tolist(), '有向投影': signed.tolist()},
        '全部31站扫描20频道费用_秒': 31 * (20 * 5 + 19),
        '费用条件': '每站先测当前频道再依次测其余19频道；不含移动、补测和清除；未实施此固定扫描流程',
        '单次示向保守半宽_米': 1500 * math.sin(math.radians(1.005)),
        '清除兜底': {'包围盒_米': [1500, 60], '方格边长_米': 20,
                     '清除候选上界': 225, '单格外接圆半径_米': 10 * math.sqrt(2)},
    }


def plot(data):
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 11, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'q4-design'})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 8.0))
    for ax in axes:
        ax.add_patch(Circle((0, 0), 1800, fill=False, ls='--', color='#80609a', lw=1.8,
                            label='干扰源所在区域边界：1800米'))
        ax.set(aspect='equal', xlim=(-2900, 2900), ylim=(-2900, 2900),
               xlabel='东向 x（米）', ylabel='北向 y（米）')
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    ax = axes[0]
    old = np.array(data['七站反例']['原七站'])
    ax.add_patch(Wedge((1800, 0), 1000, -90, 90, facecolor='#f6d0a7', edgecolor='#be7834',
                       alpha=.75, label='反例：向东发射，半径1000米'))
    ax.scatter(old[:, 0], old[:, 1], s=45, marker='s', c='#26485a', label='第三问原七站')
    ax.scatter(1800, 0, s=75, c='#b64336', marker='*', zorder=5)
    ax.annotate('边界源 G=(1800, 0)', (1800, 0), xytext=(150, -650),
                arrowprops={'arrowstyle': '->', 'color': '#943b32'}, fontsize=10)
    ax.annotate('', (2660, 0), (1900, 0), arrowprops={'arrowstyle': '->', 'lw': 2, 'color': '#b36e25'})
    ax.text(-200, 1900, '七站均在发射半平面背侧', ha='center', color='#943b32', fontsize=12)
    ax.set_title('为什么第四问不能沿用七站发现保证', pad=14)
    ax.legend(loc='lower left', fontsize=9, framealpha=.95)

    ax = axes[1]
    for triangle in data['相交三角形顶点_米']:
        ax.add_patch(Polygon(triangle, fill=False, edgecolor='#bac5cd', lw=.8))
    stations = np.array(data['站点坐标_米'])
    inside = np.linalg.norm(stations, axis=1) <= 1800 + 1e-8
    ax.scatter(stations[inside, 0], stations[inside, 1], s=36, color='#28566a', zorder=4, label='区域内候选站：13个')
    ax.scatter(stations[~inside, 0], stations[~inside, 1], s=45, marker='D', color='#d38b37', zorder=4, label='区域外候选站：18个')
    sample_triangle = np.array([lattice_point(k) for k in [(0, 0), (1, 0), (0, 1)]])
    ax.add_patch(Polygon(sample_triangle, facecolor='#55ad92', alpha=.3, edgecolor='#28836b', lw=2))
    ax.set_title('有连续覆盖证明的31站备选构造', pad=14)
    ax.legend(loc='upper left', fontsize=9, framealpha=.95)
    fig.suptitle('第四问：未知朝向的半平面发射与发现保障', fontsize=16, y=.98)
    fig.text(.27, .10, '朝外发射的边界源，可能使所有原七站均无信号。\n题目允许机器狗到1800米圆盘外检测。',
             ha='center', va='center', fontsize=10)
    fig.text(.75, .10, '三角形内任意位置距三个顶点均不超过950米。\n任意发射朝向至少可被其中一个顶点站接收。',
             ha='center', va='center', fontsize=10)
    fig.text(.5, .035, '几何设计示意；三角网是可复用和删减的备选站点池，尚未确定执行路线；未运行模拟器。',
             ha='center', fontsize=10, color='#53616d')
    fig.tight_layout(rect=(0, .16, 1, .95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        fig.savefig(FIG.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    result = calculate()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plot(result)
    print(json.dumps({k: result[k] for k in ['三角形数量', '站点数量', '区域外站点数量',
        '最远站点离圆心_米', '采样复核', '全部31站扫描20频道费用_秒']}, ensure_ascii=False, indent=2))
