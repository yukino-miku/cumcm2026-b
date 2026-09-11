"""计算并绘制扇形清空设计；只做解析几何核算，不运行策略或模拟器。"""
import argparse
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
FIG = ROOT / 'results/figures/第三问/扇形清空_边界与补充扫描'
DATA = ROOT / 'results/tables/第三问/扇形清空_几何核算.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def point(radius, degrees):
    return np.array([radius*math.cos(math.radians(degrees)), radius*math.sin(math.radians(degrees))])


def calculate():
    current, following, middle, example = point(1200, 0), point(1200, 60), point(1200, 30), point(1800, 50)
    # 对900至1800米的径向区间，距离平方为凸函数，最大值在径向及角度边界。
    edge = [math.sqrt(r*r+1200**2-2*r*1200*math.cos(math.pi/6)) for r in [900, 1800]]
    bound = max(900., *edge)
    assert bound < 999.5
    assert np.linalg.norm(example-current) > 1000 and np.linalg.norm(example-middle) < 1000
    rho = np.linspace(0, 1800, 1801)[:, None]
    theta = np.linspace(0, math.pi/3, 601)[None, :]
    squared = rho*rho+1200**2-2*rho*1200*np.cos(theta-math.pi/6)
    sampled = float(np.minimum(rho, np.sqrt(np.maximum(squared, 0))).max())
    assert abs(sampled-bound) < 1e-8  # 数值复核，连续覆盖的依据是上面的解析边界。
    return {
        '性质': '待实施扇形方案的解析几何构造；不是算法运行或实测收益',
        '扇形角度_度': [0, 60], '扇形半径_米': 1800,
        '当前站': current.tolist(), '下一站': following.tolist(), '补充扫描站': middle.tolist(),
        '中心与补充站联合覆盖最大最近距离_米': bound,
        '连续覆盖证明径向端点距离_米': edge, '网格复核最大最近距离_米': sampled,
        '仅加入补充站的折线路程增量_米': float(np.linalg.norm(current-middle)+np.linalg.norm(middle-following)-np.linalg.norm(current-following)),
        '盲区反例': {'位置': example.tolist(), '到中心_米': 1800.,
                   '到当前站_米': float(np.linalg.norm(example-current)),
                   '到补充站_米': float(np.linalg.norm(example-middle))},
    }


def plot():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'svg.hashsalt': 'q3-sector-design'})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    colors = ['#DCECF6', '#DAEFE6', '#F8EAD3', '#EFDFEC', '#E3E0F2', '#F4DEDC']
    for i in range(6):
        axes[0].add_patch(Wedge((0, 0), 1800, i*60, (i+1)*60, facecolor=colors[i], edgecolor='white'))
        q = point(1420, i*60+30)
        axes[0].text(*q, f'第{i+1}扇形', ha='center', va='center')
        q = point(1800, i*60)
        axes[0].plot([0, q[0]], [0, q[1]], color='#71818B', lw=.8)
        q = point(1200, i*60)
        axes[0].scatter(*q, marker='s', color='#263E50', s=45, zorder=5)
        axes[0].annotate(f'站{i+1}', q, xytext=(6, 6), textcoords='offset points', fontsize=10)
    axes[0].scatter(0, 0, marker='*', s=100, color='#263E50', zorder=6)
    axes[0].set_title('按固定站径向连线划分六个60°扇形')
    ax = axes[1]
    ax.add_patch(Wedge((0, 0), 1800, 0, 60, facecolor='#DCECF6', edgecolor='#577F9A', alpha=.7))
    m = point(1200, 30)
    for center, color, label in [(np.zeros(2), '#777777', '中心1000米接收范围'), (m, '#2B9679', '补充站1000米接收范围')]:
        ax.add_patch(Circle(center, 1000, fill=False, color=color, linestyle='--', lw=1.4, label=label))
    route = np.array([point(1200, 0), m, point(1200, 60)])
    ax.plot(route[:, 0], route[:, 1], color='#C27E28', lw=2, label='仅连接扫描站的示意路线')
    for q, label, offset in [(route[0], '当前站：0°', (5, -20)), (m, '补充站：30°', (10, -22)), (route[2], '下一站：60°', (-94, 14))]:
        ax.scatter(*q, s=55, marker='D' if label.startswith('补充') else 's', color='#263E50', zorder=5)
        ax.annotate(label, q, xytext=offset, textcoords='offset points', fontsize=10)
    x = point(1800, 50)
    ax.scatter(*x, marker='x', s=65, color='#B54D4D', zorder=6)
    ax.annotate('当前站可能漏检的构造点', x, xytext=(-60, 22), textcoords='offset points', color='#A14141', fontsize=10)
    ax.scatter(0, 0, marker='*', s=100, color='#263E50', zorder=6)
    ax.annotate('圆心', (0, 0), xytext=(-30, -18), textcoords='offset points')
    ax.set_title('中心与角平分线补充站联合覆盖本扇形')
    ax.legend(loc='lower right', fontsize=8, framealpha=.95)
    for a in axes:
        a.set_aspect('equal'); a.set_xlabel('东向 x（米）'); a.set_ylabel('北向 y（米）')
        a.grid(alpha=.17); a.set_axisbelow(True)
    axes[0].set_xlim(-2050, 2050); axes[0].set_ylim(-2050, 2050)
    ax.set_xlim(-1120, 2180); ax.set_ylim(-1120, 1950)
    fig.suptitle('扇形内全部清除后再推进：明确边界，并补足未知频道的发现覆盖', fontsize=16, y=.97)
    fig.text(.5, .025, '几何设计示意，非实际轨迹；扫描覆盖不等于完成定位，仍须取得反馈后补测与安全清除。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.07, right=.985, bottom=.14, top=.87, wspace=.23)
    for suffix in ['.png', '.svg']:
        fig.savefig(FIG.with_suffix(suffix), dpi=180, metadata={'Date': None} if suffix == '.svg' else None)
        if suffix == '.svg':
            path = FIG.with_suffix(suffix)
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',
                            encoding='utf-8', newline='\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='重算几何并核对已保存来源及图表哈希')
    args = parser.parse_args()
    result = calculate()
    if not args.check:
        plot()
    result['来源SHA256'] = {Path(__file__).relative_to(ROOT).as_posix(): digest(Path(__file__))}
    result['图表SHA256'] = {FIG.with_suffix(s).relative_to(ROOT).as_posix(): digest(FIG.with_suffix(s)) for s in ['.png', '.svg']}
    if args.check:
        assert result == json.loads(DATA.read_text(encoding='utf-8'))
    else:
        DATA.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
