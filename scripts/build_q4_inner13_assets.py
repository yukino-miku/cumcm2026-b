"""运行并核验12个第四问本地构造，输出中文轨迹、费用图和报告；无官方接口调用。"""
from q4_archive_sources import matches_source
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q4_strategy import Q4Config, SchemeFour
from run_q4 import provenance

RUNS = ROOT/'results/models/第四问/十三站初版'
TABLE = ROOT/'results/tables/第四问/十三站初版_验证汇总.json'
FIG = ROOT/'results/figures/第四问/十三站初版'
REPORT = ROOT/'docs/第四问/十三站试验版_验证与图集.md'
CASE_SPECS = [
    ('均匀全向', 10, 'uniform', 'min', 0., 'random'),
    ('均匀混合', 13, 'uniform', 'mixed', .5, 'random'),
    ('均匀定向', 16, 'uniform', 'min', 1., 'random'),
    ('边界混合', 10, 'boundary', 'min', .5, 'random'),
    ('边界朝内混合', 13, 'boundary', 'mixed', .5, 'inward'),
    ('边界切向定向', 16, 'boundary', 'min', 1., 'tangent'),
    ('聚集混合', 10, 'cluster', 'min', .5, 'random'),
    ('聚集定向', 13, 'cluster', 'max', 1., 'random'),
    ('中心混合', 16, 'center', 'min', .5, 'random'),
    ('中心定向', 13, 'center', 'mixed', 1., 'random'),
    ('边界全部朝外_10源', 10, 'boundary', 'min', 1., 'outward'),
    ('边界全部朝外_16源', 16, 'boundary', 'min', 1., 'outward'),
]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def run_cases():
    parameters = json.loads((ROOT/'configs/q4_inner13.json').read_text(encoding='utf-8'))
    config = Q4Config(**parameters)
    source = provenance()
    for i, (label, count, layout, radius, fraction, orientation) in enumerate(CASE_SPECS, 1):
        settings = dict(seed=20260911+i, count=count, layout=layout, radius_mode=radius,
                        error_mode='hash', directional_fraction=fraction, orientation=orientation)
        env = make_case(**settings)
        client = RobotClient(env, 'local-robot')
        try:
            result = SchemeFour(client, config).run()
            result.update(构造编号=i, 构造名称=label, 构造参数=settings, 运行来源=source,
                          运行类型='本地代表布局构造，非官方演练或成绩')
            result['离线真值核验'] = env.truth_for_evaluation()
            dump(RUNS/f'场景{i:02d}.json', result)
            assert result['流程正常完成'], (i, result['异常'])
            print(f'场景{i:02d} {label}：{result["清除数"]}/{count}，{result["虚拟总时间_秒"]:.3f}秒', flush=True)
        finally:
            client.close()


def verify_run(result):
    assert result['流程正常完成'] and result['正常退出'] and result['异常'] is None
    truth = result['离线真值核验']
    sources = {s['频道']: s for s in truth['目标']}
    cleared = set()
    current = np.zeros(2)
    channel = 1
    virtual = distance_sum = 0.
    counts = dict(measure=0, switch=0, success=0, fail=0, envelope=0)
    for event in result['动作记录']:
        if '外包顶点' in event:
            assert min_distance(event['外包顶点'], sources[event['频道']]['位置']) < 1e-5
            counts['envelope'] += 1
        if event['动作'] not in {'检测', '清除'}:
            continue
        point = np.asarray(event['位置'])
        travel = float(np.linalg.norm(point-current))
        if event.get('用途') == '顺路检测':
            assert travel < 1e-6  # 顺路扫描只能复用当前真实停点。
        current = point
        distance_sum += travel
        virtual += travel/5
        c = event['频道']
        target = sources.get(c) if c not in cleared else None
        distance = math.dist(point, target['位置']) if target else math.inf
        if event['动作'] == '检测':
            assert event['阶段'] != '末尾增站评估'
            assert event['用途'].startswith('固定搜索站') or event['用途'] in {'顺路检测', '追加测向'}
            counts['measure'] += 1
            switch = int(c != channel)
            counts['switch'] += switch
            virtual += 5+switch
            channel = c
            reception = target is not None and distance <= target['接收半径']
            if reception and target['发射轴_度'] is not None:
                angle = math.radians(target['发射轴_度'])
                dx, dy = point-np.asarray(target['位置'])
                reception = dx*math.cos(angle)+dy*math.sin(angle) >= -1e-9
            expected = 'no_signal' if not reception else 'near' if distance <= 5 else 'direction'
            assert expected == event['结果']
            if expected == 'direction':
                delta = np.asarray(target['位置'])-point
                bearing = math.degrees(math.atan2(delta[1], delta[0]))
                error = (event['示向度']-bearing+180) % 360-180
                assert abs(error) <= 1.005+1e-8
        else:
            success = distance <= 20
            assert event['结果'] == ('success' if success else 'no_target_in_range')
            if event['具有覆盖证书']:
                assert success and event['最远可能目标距离_米'] < 20
            virtual += 5 if success else 3
            counts['success' if success else 'fail'] += 1
            if success:
                cleared.add(c)
        assert abs(virtual-event['虚拟时间_秒']) < .001
    assert cleared == set(result['已清除频道'])
    assert sources.keys()-cleared == set(truth['遗漏频道'])
    assert bool(len(sources) == len(cleared)) == truth['全部清除']
    assert abs(distance_sum-result['总路程_米']) < 1e-5
    assert abs(virtual-result['虚拟总时间_秒']) < .001
    assert abs(sum(result['时间分项_秒'].values())-virtual) < .001
    assert [counts[k] for k in ['measure', 'switch', 'success', 'fail']] == [
        result[k] for k in ['检测次数', '频道切换次数', '清除成功次数', '清除失败次数']]
    assert result['已完成搜索站'] == list(range(len(result['已完成搜索站'])))
    assert len(result['已完成搜索站']) == 13 or result['清除数'] == 16
    assert not result['已证明不存在频道'] and not result['已发现未清除频道']
    assert result['末尾增站评估']['自动新增未知源搜索站数'] == 0
    assert result['全部完成证据'] == (len(cleared) == 16)
    for record in result['频道记录']:
        assert all(e['来源'] == '清除失败' and e['半径'] == 20 for e in record['排除约束'])
        assert record['不存在证书'] is None
    return counts


def load_and_verify():
    records, rows = [], []
    aggregate = dict(measure=0, switch=0, success=0, fail=0, envelope=0)
    for i, spec in enumerate(CASE_SPECS, 1):
        path = RUNS/f'场景{i:02d}.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['构造编号'] == i and record['构造名称'] == spec[0]
        for rel, expected in record['运行来源']['源码_SHA256'].items():
            assert matches_source(ROOT, rel, expected), rel
        counts = verify_run(record)
        for key in aggregate:
            aggregate[key] += counts[key]
        records.append(record)
        truth = record['离线真值核验']
        rows.append({'编号': i, '名称': spec[0], '目标数': truth['目标数'], '清除数': record['清除数'],
                     '实际全清': truth['全部清除'], '策略具有全清证据': record['全部完成证据'],
                     '遗漏频道': truth['遗漏频道'], '总时间_秒': record['虚拟总时间_秒'],
                     '路程_米': record['总路程_米'], '检测次数': record['检测次数'],
                     '追加测向无信号次数': record['路线调度统计']['补测无信号次数'],
                     '顺路证书清除次数': record['路线调度统计']['顺路证书清除次数'],
                     '现实程序耗时_秒': record['程序运行时间_秒'], '时间分项_秒': record['时间分项_秒'],
                     '记录路径': path.relative_to(ROOT).as_posix(), '记录_SHA256': sha256(path.read_bytes()).hexdigest()})
    return records, {'性质': '12个代表布局的本地构造验证；非官方演练，不是独立随机样本的总体性能估计',
                     '策略来源提交': sorted({r['运行来源']['Git提交'] for r in records}),
                     '逐场景': rows, '逐动作核验计数': aggregate,
                     '实际全清场数': sum(r['实际全清'] for r in rows),
                     '总目标数': sum(r['目标数'] for r in rows), '总清除数': sum(r['清除数'] for r in rows)}


def setup_font():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'svg.hashsalt': 'q4-inner13-results'})


def save_figure(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        path = FIG/(name+suffix)
        fig.savefig(path, dpi=170, metadata={'Date': None} if suffix == '.svg' else None)
        if suffix == '.svg':
            # 与仓库*.svg eol=lf一致，避免Windows绘图换行导致队友克隆后哈希不符。
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    plt.close(fig)


def trajectory(ax, result, limit, overview=False):
    moves = [e for e in result['动作记录'] if e['动作'] in {'检测', '清除'}]
    points = np.array([[0., 0.]]+[e['位置'] for e in moves])
    fixed = np.array(result['固定站坐标'])
    ax.add_patch(Circle((0, 0), 1800, fill=False, color='#8968a1', ls='--', lw=1.4))
    ax.plot(points[:, 0], points[:, 1], color='#4d96b5', lw=1.1, alpha=.8, zorder=2)
    for a, b in zip(points[:-1], points[1:]):
        if np.linalg.norm(b-a) > 160:
            ax.annotate('', a+.58*(b-a), a+.43*(b-a),
                        arrowprops={'arrowstyle': '-|>', 'lw': .8, 'color': '#4d96b5'})
    ax.scatter(fixed[:, 0], fixed[:, 1], marker='s', s=16 if overview else 26, color='#304d5a', label='固定检测站')
    if not overview:
        for i, p in enumerate(fixed, 1):
            ax.annotate(str(i), p, xytext=(5, 5), textcoords='offset points', fontsize=8)
    truths = result['离线真值核验']
    missed = set(truths['遗漏频道'])
    labels = set()
    for source in truths['目标']:
        p = np.array(source['位置'])
        axis = source['发射轴_度']
        label = '遗漏源（事后真值）' if source['频道'] in missed else '已清除源（事后真值）'
        color = '#c74640' if source['频道'] in missed else '#53545b'
        ax.scatter(*p, color=color, marker='x', s=24 if overview else 40, zorder=6,
                   label=label if label not in labels else None)
        labels.add(label)
        if axis is not None:
            v = 135*np.array([math.cos(math.radians(axis)), math.sin(math.radians(axis))])
            ax.annotate('', p+v, p, arrowprops={'arrowstyle': '->', 'color': '#c0832f', 'lw': 1.1}, zorder=5)
    for event in moves:
        if event['动作'] != '清除':
            continue
        label = '顺路清除位置' if event['用途'] == '顺路证书清除' else '其他清除位置'
        if event['结果'] != 'success':
            label = '清除未命中'
        if event['结果'] == 'success':
            ax.scatter(*event['位置'], marker='o', s=21 if overview else 38,
                       facecolors='#249b80' if event['用途'] == '顺路证书清除' else 'none',
                       edgecolors='#249b80', alpha=.85, zorder=4,
                       label=label if label not in labels else None)
        else:
            ax.scatter(*event['位置'], marker='+', s=21 if overview else 38, color='#c0832f', alpha=.75, zorder=4,
                       label=label if label not in labels else None)
        labels.add(label)
    ax.scatter(0, 0, marker='*', s=90, c='#293d51', zorder=7, label='起点')
    ax.scatter(*points[-1], marker='D', s=28, c='#85497f', zorder=7, label='终点')
    ax.set(aspect='equal', xlim=(-limit, limit), ylim=(-limit, limit), xlabel='东向 x（米）', ylabel='北向 y（米）')
    ax.grid(alpha=.12)
    ax.set_title(f'{result["构造编号"]:02d} · {result["构造名称"]} · 清除{result["清除数"]}/{truths["目标数"]}\n'
                 f'路程{result["总路程_米"]/1000:.2f}公里｜虚拟时间{result["虚拟总时间_秒"]:.1f}秒',
                 fontsize=9 if overview else 13)
    if not overview:
        ax.legend(loc='upper left', fontsize=8, framealpha=.9)


def plot_all(records):
    setup_font()
    extent = max(abs(v) for r in records for e in r['动作记录'] if e['动作'] in {'检测', '清除'} for v in e['位置'])
    limit = max(2100, math.ceil((extent+120)/100)*100)
    for r in records:
        fig, ax = plt.subplots(figsize=(8.6, 8.6))
        trajectory(ax, r, limit)
        fig.text(.5, .025, '本地构造；灰/红叉为事后真值，橙箭头为发射方向；策略未读取真值。', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, 1))
        save_figure(fig, f'轨迹{r["构造编号"]:02d}')
    fig, axes = plt.subplots(4, 3, figsize=(16, 21))
    for ax, r in zip(axes.flat, records):
        trajectory(ax, r, limit, overview=True)
    fig.suptitle('第四问13站试验版：12个本地构造轨迹', fontsize=18)
    fig.text(.5, .017, '蓝线为实际路径；方块为固定站；红叉为遗漏源；橙箭头为定向轴。事后真值不参与决策，均非官方演练。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .975))
    save_figure(fig, '十二场景轨迹总览')
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.2), gridspec_kw={'height_ratios': [2, 1]})
    x = np.arange(12)
    bottom = np.zeros(12)
    for key, color in [('移动', '#69a3bd'), ('检测', '#75b79c'), ('切换', '#dfbc71'),
                       ('清除成功', '#9b87b2'), ('清除失败', '#ce8176')]:
        values = np.array([r['时间分项_秒'][key] for r in records])
        axes[0].bar(x, values, bottom=bottom, label=key, color=color)
        bottom += values
    axes[0].set(ylabel='虚拟时间（秒）', title='完整费用构成：同时记录移动、检测、切换及清除')
    axes[0].legend(ncol=5)
    totals = [r['离线真值核验']['目标数'] for r in records]
    done = [r['清除数'] for r in records]
    axes[1].bar(x, totals, color='#eadbd8', label='实际目标数')
    axes[1].bar(x, done, color='#5b9e8a', label='已清除数')
    for i, (a, b) in enumerate(zip(done, totals)):
        axes[1].text(i, b+.25, f'{a}/{b}', ha='center', fontsize=10)
    axes[1].set(ylabel='目标数量', ylim=(0, 19))
    axes[1].legend(ncol=2)
    for ax in axes:
        ax.set_xticks(x, [f'{r["构造编号"]:02d}' for r in records])
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[1].set_xlabel('构造场景编号；11—12为边界全部朝外的不利场景')
    fig.suptitle('第四问13站试验版：耗时与清除结果（本地构造）', fontsize=16)
    fig.text(.5, .025, '耗时较短可能源于漏检，必须结合清除结果判断；这些构造不用于估计官方全清概率。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .965))
    save_figure(fig, '费用与清除结果')


def write_report(summary):
    rows = summary['逐场景']
    lines = ['# 第四问13站试验版：本地验证与图集', '',
        '2026-09-12。按用户要求先运行圆内13站，保留顺路定位清除，末尾评估而不自动外围搜索。以下均为本地构造，未调用官方模拟器，不是正式成绩。', '',
        f'固定策略来源提交：`{summary["策略来源提交"][0]}`。运行记录保存实际配置和源码哈希；新增评估文件可使工作区显示非空，但逐个源码哈希已重新核对。', '',
        f'12个构造流程均正常结束，实际全清{summary["实际全清场数"]}/12场，合计清除{summary["总清除数"]}/{summary["总目标数"]}个目标。这些布局经过明确选择，不能把场数比例当成官方场景的成功概率。', '',
        '## 逐场结果', '',
        '| 编号 | 构造布局 | 清除/目标 | 路程（公里） | 总时间（秒） | 补测无信号次数 |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["编号"]:02d} | {r["名称"]} | {r["清除数"]}/{r["目标数"]} | {r["路程_米"]/1000:.3f} | {r["总时间_秒"]:.3f} | {r["追加测向无信号次数"]} |')
    lines += ['', '第11—12场把全部源放在圆盘边界附近并朝外发射，用于检查固定13站不足时的真实行为。程序不会为追求全清而擅自增加未知源搜索站；漏检及未发现频道保留在结果里。', '',
        '实际全清是退出后的真值核验；运行中全清证据只在清除16个不同频道时成立。本版少于16源的场景即使事后全清，仍保留未知频道待核查状态。', '',
        '## 动作核验', '',
        f'逐条重算{summary["逐动作核验计数"]["measure"]}次检测、{summary["逐动作核验计数"]["success"]}次成功清除和{summary["逐动作核验计数"]["fail"]}次失败清除；检查{summary["逐动作核验计数"]["envelope"]}份位置外包包含真实源。', '',
        '对每个动作独立按距离、固定发射轴、近距及光学半径重算反馈；核对示向误差、移动、切换、累计虚拟时间、固定站次序、同站扫描和清除证书。检查无信号没有添加1000米排除圆，不存在虚假的频道不存在证书。', '',
        '本阶段沿用已通过的26项第四问回归及226项全库回归。构造检查没有调节默认策略参数，也没有据结果自动启用外侧补站。', '',
        '## 中文图表', '',
        '![费用与清除结果](../../results/figures/第四问/十三站初版/费用与清除结果.png)', '',
        '![十二场景轨迹总览](../../results/figures/第四问/十三站初版/十二场景轨迹总览.png)', '',
        '固定站、真实动作、起终点、顺路与其他清除均作标记。绿色实心圆为顺路证书清除，空心圆为其他成功清除，橙色加号为未命中尝试。真值位置和发射方向只用于事后绘图，不提供给策略；清除标记表示机器狗位置，可能与源真值略有不同。', '',
        '| 场景 | 高清轨迹 | 可编辑矢量图 |', '|---|---|---|']
    for r in rows:
        i = r['编号']
        lines.append(f'| {i:02d} {r["名称"]} | [PNG](../../results/figures/第四问/十三站初版/轨迹{i:02d}.png) | [SVG](../../results/figures/第四问/十三站初版/轨迹{i:02d}.svg) |')
    lines += ['', '## 复现', '', '在项目根目录执行：', '', '```powershell',
        r'.\.venv\Scripts\python.exe -X utf8 .\scripts\build_q4_inner13_assets.py', '```', '',
        '以上重新运行12个构造并更新图表。只重读归档、核验和重绘：', '', '```powershell',
        r'.\.venv\Scripts\python.exe -X utf8 .\scripts\build_q4_inner13_assets.py --plots-only', '```', '',
        '只核验已有数据与图表来源：', '', '```powershell',
        r'.\.venv\Scripts\python.exe -X utf8 .\scripts\build_q4_inner13_assets.py --verify', '```', '',
        '[统计与来源哈希](../../results/tables/第四问/十三站初版_验证汇总.json)；[完整运行记录目录](../../results/models/第四问/十三站初版)；[使用说明](十三站试验版_实现与运行.md)。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plots-only', action='store_true')
    group.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.plots_only and not args.verify:
        run_cases()
    records, summary = load_and_verify()
    if args.verify:
        saved = json.loads(TABLE.read_text(encoding='utf-8'))
        assert matches_source(ROOT, Path(__file__).relative_to(ROOT).as_posix(), saved['生成脚本_SHA256'])
        for key, value in summary.items():
            assert saved[key] == value, key
        for rel, expected in saved['图表_SHA256'].items():
            assert matches_source(ROOT, rel, expected), rel
    else:
        plot_all(records)
        summary['图表_SHA256'] = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(FIG.glob('*')) if p.suffix in {'.png', '.svg'}}
        summary['生成脚本_SHA256'] = sha256(Path(__file__).read_bytes()).hexdigest()
        dump(TABLE, summary)
        write_report(summary)
    print(json.dumps({k: summary[k] for k in ['实际全清场数', '总目标数', '总清除数', '逐动作核验计数']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
