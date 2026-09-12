"""第四问方案二：中文最短路线、流程、8组配对轨迹、费用图及图文来源清单。"""
import argparse
from hashlib import sha256
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch
import numpy as np
from PIL import Image

from evaluate_q4_scheme2 import ROOT, TABLE, ROUTE, RUNS, CASES, read, dump, path_for
from cumcm2026_b.q4_scheme2.route import shortest_route

FIG = ROOT / 'results/figures/第四问/方案二十九站'
MANIFEST = ROOT / 'results/tables/第四问/方案二十九站_图文来源.json'
BLUE, ORANGE, GREEN, PURPLE = '#4b91ae', '#d88732', '#159b84', '#94609f'


def setup():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 10, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'q4-scheme2-19'})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        path = FIG / (name + suffix)
        fig.savefig(path, dpi=160, metadata={'Date': None} if suffix == '.svg' else None)
        if suffix == '.svg':
            # 路径节点用换行分隔，去掉Matplotlib附带的行末空格，不改变图形。
            path.write_bytes(b'\n'.join(line.rstrip() for line in path.read_bytes().splitlines()) + b'\n')
    plt.close(fig)


def axes_base(ax, limit=2200):
    ax.add_patch(Circle((0, 0), 1800, fill=False, color=PURPLE, ls='--', lw=1.5))
    ax.set(aspect='equal', xlim=(-limit, limit), ylim=(-limit, limit),
           xlabel='东向 x（米）', ylabel='北向 y（米）')
    ax.grid(alpha=.15)


def route_figure():
    points, cert = shortest_route()
    fig, ax = plt.subplots(figsize=(9, 9.3))
    axes_base(ax)
    ax.plot(*points.T, color=BLUE, lw=1.7, zorder=2)
    for a, b in zip(points[:-1], points[1:]):
        ax.annotate('', a+.6*(b-a), a+.4*(b-a),
                    arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': 1.5})
    for p, row in zip(points, cert['访问表']):
        new = row['类型'] == '新增外围站'
        ax.scatter(*p, c=ORANGE if new else '#294c5e', marker='D' if new else 's', s=65, zorder=3)
        ax.annotate(str(row['访问序号']), p, xytext=(6, 7), textcoords='offset points',
                    fontsize=11, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': .7})
    ax.scatter(0, 0, c='#294c5e', marker='*', s=190, edgecolor='white', zorder=4)
    ax.set_title('第四问方案二 · 19站全局最短固定路线\n圆心出发，终点自由：18段 × 950米 = 17.1公里', fontsize=15, pad=15)
    handles = [Line2D([], [], marker='s', ls='', color='#294c5e', label='原有13站'),
               Line2D([], [], marker='D', ls='', color=ORANGE, label='新增6站（半径1900米）'),
               Line2D([], [], ls='--', color=PURPLE, label='源区域边界（半径1800米）')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, .045), fontsize=9)
    fig.text(.5, .022, '数字表示到达顺序；图中仅绘固定路线，实际运行仍会插入顺路补测和清除。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.13, right=.95, bottom=.15, top=.89)
    save(fig, '01_十九站最短固定路线')


def flow_figure():
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis('off')
    items = [
        ('从圆心进入任务', '预先求得19站最短顺序；不读取目标真值或总数'),
        ('到达下一固定站并检测', '沿用频道免测规则；已清除频道不再检测'),
        ('更新已发现目标的连续位置区域', '定向无信号不能直接作距离排除；保留方向反馈评分'),
        ('评估通往剩余固定站的路线', '顺路清除、100米优先补测间距、800米绕行门槛均沿用'),
        ('执行合适的补测或清除，再继续固定路线', '实际停点继续顺路扫描；没有合适任务就前往下一固定站'),
        ('19站结束后处理尚未清除的已发现源', '仍使用原有定位与有限清除；不再追加未知源补漏站'),
        ('正常退出并保存日志', '若中途清除16源，可依数量上界提前退出；流程完成不等于全清证明'),
    ]
    for i, (title, detail) in enumerate(items):
        y = 8.7 - i * 1.25
        ax.add_patch(FancyBboxPatch((.85, y-.43), 8.1, .88, boxstyle='round,pad=.06',
                                   fc='#edf5f8' if i != 5 else '#fff4e8', ec=BLUE, lw=1.2))
        ax.text(4.9, y+.12, title, ha='center', va='center', fontsize=12)
        ax.text(4.9, y-.21, detail, ha='center', va='center', fontsize=9)
        if i < len(items)-1:
            ax.annotate('', (4.9, y-.75), (4.9, y-.5), arrowprops={'arrowstyle': '->', 'color': BLUE})
    ax.annotate('', (.5, 7.45), (.5, 3.7), arrowprops={'arrowstyle': '->', 'color': ORANGE, 'lw': 1.5})
    ax.plot([.5, .78], [3.7, 3.7], c=ORANGE)
    ax.plot([.5, .78], [7.45, 7.45], c=ORANGE)
    ax.text(.25, 5.6, '还有固定站', rotation=90, ha='center', va='center', color=ORANGE)
    fig.suptitle('第四问方案二执行流程', fontsize=18, y=.96)
    fig.text(.5, .035, '改变的是固定站集合和访问顺序；动态检测、补测、定位及清除规则直接复用方案一。', ha='center', fontsize=10)
    save(fig, '02_十九站方案执行流程')


def trajectory(ax, result, scheme, limit):
    axes_base(ax, limit)
    points = np.asarray(result['固定站坐标'])
    completed = set(result['已完成搜索站'])
    current = np.zeros(2)
    for event in result['动作记录']:
        if event['动作'] not in {'检测', '清除'}:
            continue
        p = np.asarray(event['位置'])
        color = (PURPLE if event['阶段'] == '固定扫描后处理已发现源' else
                 GREEN if event['动作'] == '清除' else
                 ORANGE if event['用途'] == '追加测向' else BLUE)
        if np.linalg.norm(p-current) > 1e-7:
            ax.plot(*np.array([current, p]).T, c=color, lw=1.1, alpha=.8)
            if np.linalg.norm(p-current) > 180:
                ax.annotate('', current+.62*(p-current), current+.42*(p-current),
                            arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': .8})
        if event.get('用途') == '追加测向':
            ax.scatter(*p, c=ORANGE, marker='^', s=22, edgecolor='white', linewidth=.3, zorder=4)
        if event['动作'] == '清除' and event['结果'] == 'success':
            ax.scatter(*p, facecolors='none', edgecolors=GREEN, s=57, lw=1.5, zorder=5)
        current = p
    for i, p in enumerate(points):
        new = np.linalg.norm(p) > 1850
        color = ORANGE if new else '#294c5e'
        ax.scatter(*p, marker='D' if new else 's', s=32, edgecolor=color,
                   facecolor=color if i in completed else 'none', alpha=1 if i in completed else .4, zorder=3)
    truth = result['离线真值核验']
    for source in truth['目标']:
        p = np.asarray(source['位置'])
        missed = source['频道'] in truth['遗漏频道']
        ax.scatter(*p, c='#c73e43' if missed else '#71777b', marker='x', s=39 if missed else 15, zorder=6)
        if missed:
            ax.annotate(f'频{source["频道"]}', p, xytext=(4, 4), textcoords='offset points', fontsize=8, color='#c73e43')
        if source['发射轴_度'] is not None:
            theta = math.radians(source['发射轴_度'])
            ax.annotate('', p + 135*np.array([math.cos(theta), math.sin(theta)]), p,
                        arrowprops={'arrowstyle': '->', 'color': '#999999', 'lw': .8}, zorder=2)
    ax.scatter(0, 0, marker='*', s=110, c='#294c5e', edgecolor='white', zorder=8)
    ax.scatter(*current, marker='P', s=62, c=PURPLE, edgecolor='white', zorder=8)
    ax.set_title(f'方案{scheme} · 清除{result["清除数"]}/{truth["目标数"]} · 固定站{len(completed)}/{len(points)}\n'
                 f'{result["总路程_米"]/1000:.2f}公里｜{result["虚拟总时间_秒"]:.1f}秒', fontsize=12)


def paired_figures():
    legend = [Line2D([], [], c=BLUE, label='固定站行进'), Line2D([], [], c=ORANGE, marker='^', label='主动补测'),
              Line2D([], [], c=GREEN, marker='o', mfc='none', label='清除行进及成功位置'),
              Line2D([], [], c=PURPLE, label='固定扫描后收尾'),
              Line2D([], [], ls='', c='#c73e43', marker='x', label='遗漏源（事后真值）'),
              Line2D([], [], ls='', c='#71777b', marker='x', label='已清除源（事后真值）')]
    for i, case in enumerate(CASES):
        pair = [read(path_for(i, s)) for s in [1, 2]]
        limit = max(2150, 1.07 * max(abs(v) for r in pair for e in r['动作记录']
                                  if e['动作'] in {'检测', '清除'} for v in e['位置']))
        fig, axs = plt.subplots(1, 2, figsize=(13, 7.8))
        for scheme, (ax, result) in enumerate(zip(axs, pair), 1):
            trajectory(ax, result, scheme, limit)
        fig.suptitle(f'同图本地构造 {i+1:02d} · {case[0]}', fontsize=16, y=.97)
        fig.legend(handles=legend, loc='lower center', ncol=3, bbox_to_anchor=(.5, .055), fontsize=9)
        fig.text(.5, .025, '虚线圈：1800米源边界；橙色菱形：新增固定站；灰箭头：源发射朝向；空心固定站尚未执行。非官方成绩。', ha='center', fontsize=9)
        fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.20, wspace=.20)
        save(fig, f'{i+3:02d}_场景{i+1:02d}_同图轨迹')


def comparison_figure():
    rows = read(TABLE)['逐场']
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    x = np.arange(8)
    for scheme, color, shift in [(1, BLUE, -.19), (2, ORANGE, .19)]:
        group = [r for r in rows if r['方案'] == scheme]
        bars = axs[0].bar(x+shift, [r['虚拟时间_秒'] for r in group], .36, color=color, label=f'方案{scheme}')
        for bar, r in zip(bars, group):
            if not r['实际全清']:
                bar.set_hatch('///')
            axs[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                        f'{r["清除数"]}/{r["目标数"]}', ha='center', fontsize=8)
        axs[1].bar(x+shift, [r['路程_米']/1000 for r in group], .36, color=color)
    axs[0].set(ylabel='虚拟时间（秒）', ylim=(0, max(r['虚拟时间_秒'] for r in rows)*1.15))
    axs[1].set(ylabel='实际路程（公里）', xticks=x,
               xticklabels=[f'{i+1:02d}\n{c[0].replace("边界全部朝外", "边界朝外")}' for i, c in enumerate(CASES)])
    for ax in axs:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('19站与13站的本地对照：清除情况、时间与路程', fontsize=16)
    fig.legend(handles=[Patch(color=BLUE, label='方案一：13站'), Patch(color=ORANGE, label='方案二：19站'),
                        Patch(facecolor='white', edgecolor='black', hatch='///', label='该次仍有遗漏')],
               loc='lower center', ncol=3, bbox_to_anchor=(.5, .045))
    fig.text(.5, .02, '柱顶为清除数/实际目标数。漏源退出较快不能视为效率胜出；8场本地构造不代表总体或官方表现。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.18, hspace=.13)
    save(fig, '11_同图清除时间与路程对照')


def manifest():
    files = [TABLE, ROUTE, Path(__file__).resolve(),
             ROOT / 'docs/第四问/方案二_十九站最短路线与运行.md',
             ROOT / 'paper/sections/第四问_方案二十九站_论文备用稿.md'] + sorted(RUNS.glob('*.json.gz')) + sorted(FIG.glob('*'))
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        setup()
        route_figure()
        flow_figure()
        paired_figures()
        comparison_figure()
        dump(MANIFEST, manifest())
    assert len(list(FIG.glob('*.png'))) == len(list(FIG.glob('*.svg'))) == 11
    for p in FIG.glob('*.png'):
        with Image.open(p) as img:
            assert img.width >= 1200 and img.height >= 1200
            img.verify()
    for p in FIG.glob('*.svg'):
        assert ET.parse(p).getroot().tag.endswith('svg')
        assert b'\r\n' not in p.read_bytes()
    assert read(MANIFEST) == manifest()
    print('11张中文图PNG/SVG及来源哈希核验通过。')


if __name__ == '__main__':
    main()
