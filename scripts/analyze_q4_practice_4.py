#!/usr/bin/env python
"""分析目录4的既有演练，关联HTTP记录，归档脱敏动作并绘制中文轨迹；不调用模拟器。"""
from __future__ import annotations

from q4_archive_sources import matches_source
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from analyze_q3_practice_log import public_header, audit_actions
from evaluate_q3_routed_practice import audit_trace
from cumcm2026_b.q4_strategy import DirectionalRegion, search_stations
from cumcm2026_b.q3_geometry import max_distance, minimum_circle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

SOURCE = ROOT / 'materials/original/simulator/CUMCM2026B/Jammers-simulator-win64/Jammers-simulator/JammersSimulatorData/behavior-logs/4'
MODELS = ROOT / 'results/models/第四问/实测4目录'
FIGURES = ROOT / 'results/figures/第四问/实测4目录'
TABLE = ROOT / 'results/tables/第四问/实测4目录_评估.json'
REPORT = ROOT / 'docs/第四问/实测4目录_结果分析与轨迹.md'
COLORS = {'固定站移动': '#438CB0', '追加测向移动': '#D88932', '扫描中清除移动': '#229E86', '扫描后收尾': '#9B5B9D'}
RESULT_KEYS = ['方案', '配置', '运行成功', '流程正常完成', '全部完成证据', '正常退出', '异常',
               '清除数', '已清除频道', '已发现未清除频道', '未发现频道', '已证明不存在频道',
               '虚拟总时间_秒', '总路程_米', '检测次数', '频道切换次数', '清除成功次数',
               '清除失败次数', '时间分项_秒', '已完成搜索站', '频道记录', '动作记录',
               '路线调度统计', '末尾增站评估', '运行来源']


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def relative(path):
    return path.relative_to(ROOT).as_posix()


def is_fixed(event):
    return event.get('用途', '').startswith(('固定搜索站', '搜索站'))


def matching_logs():
    index = []
    for number, name in [(3, '第三问'), (4, '第四问')]:
        for folder in (ROOT / 'local-only' / name).glob('*http*'):
            log, result = folder / '请求响应日志.jsonl', folder / '运行结果.json'
            if not log.exists() or not result.exists():
                continue
            items = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
            exits = {x['payload']['request_id']: x['payload'].get('robot_id') for x in items
                     if x.get('类型') == '请求' and x['path'] == '/exit'}
            for x in items:
                if x.get('类型') == '响应' and x['request_id'] in exits and x['response'].get('accepted') is True:
                    index.append((number, exits[x['request_id']], x['response']['real_timestamp_ms'], folder, items))
    return index


def collect():
    index = matching_logs()
    runs = []
    counters = Counter()
    files = sorted(SOURCE.glob('*.result.json'), key=lambda p: read(p)['ended_at_utc'])
    assert len(files) == 15, '输入目录场次发生变化，需重新审阅分析范围'
    for file in files:
        official = read(file)
        stem = file.name.removesuffix('.result.json')
        jlog, psum = SOURCE / (stem + '.jlog'), SOURCE / (stem + '.psum')
        header, summary = public_header(jlog)
        raw = psum.read_bytes()
        assert raw[:8] == b'JMBPSUM1' and int.from_bytes(raw[8:10], 'big') == 1
        length = int.from_bytes(raw[10:14], 'big')
        assert 0 < length <= len(raw)-14
        ph = json.loads(raw[14:14+length])
        assert header['package_type'] == 'practice_behavior_log' and ph['package_type'] == 'practice_summary'
        assert header['formal_index'] is None and ph['formal_index'] is None
        assert header['team_no'] == ph['team_no']
        for key in ['problem_no', 'practice_run_no', 'case_code']:
            assert official[key] == header[key] == ph[key], key
        assert official['package_sha256'] == digest(jlog)
        assert official['jammer_count'] == official['omnidirectional_jammer_count'] + official['directional_jammer_count']
        created = datetime.fromisoformat(header['created_at_utc'].replace('Z', '+00:00')).timestamp()*1000
        matches = [x for x in index if x[0] == official['problem_no'] and x[1] == header['team_no'] and abs(created-x[2]) <= 2000]
        assert len(matches) == 1, f'{stem}不能唯一关联HTTP记录'
        _, _, exited, folder, items = matches[0]
        original = read(folder / '运行结果.json')
        audit = audit_actions(items, original)
        actions = audit_trace(items, original)
        # 额外核对测向值；请求ID只用于内存关联，不写入归档。
        responses = {x['request_id']: x['response'] for x in items if x.get('类型') == '响应'}
        requests = {x['payload']['request_id']: x for x in items if x.get('类型') == '请求' and x['path'] in ('/measure', '/clear')}
        for (rid, req), event in zip(requests.items(), actions):
            if req['path'] == '/measure':
                assert event.get('示向度') == responses[rid].get('svd_deg')
        number = official['problem_no']
        counters[number] += 1
        label = f'P{number}-{counters[number]}'
        fixed = {}
        for event in actions:
            if is_fixed(event):
                station = int(re.search(r'\d+$', event['用途']).group())
                point = event['位置']
                assert station not in fixed or math.dist(point, fixed[station]) < 1e-8
                fixed[station] = point
        record = {'编号': label, '题号': number, '官方结果': official, '公开头摘要': summary,
                  '关联': {'本地目录': relative(folder), '同队号': True, '生成减退出响应_毫秒': created-exited,
                           '依据': '同队号、唯一退出时间匹配；未解密正文、未验证包签名'},
                  '协议核验': audit, '固定站坐标': [fixed[k] for k in sorted(fixed)],
                  '结果': {k: original[k] for k in RESULT_KEYS if k in original},
                  '来源SHA256': {relative(p): digest(p) for p in [file, jlog, psum, folder / '运行结果.json', folder / '请求响应日志.jsonl']}}
        if number == 4:
            meta = original['运行来源']
            assert meta['Git提交'] == 'eec554a91e4df12bb88de1d751b43396e707e41d' and meta['Git工作区干净']
            for name, value in meta['源码_SHA256'].items():
                assert matches_source(ROOT, name, value), name
            assert np.allclose(record['固定站坐标'], search_stations(), rtol=0, atol=1e-8)
        record['诊断'] = analyze(record)
        write(MODELS / (label + '.json'), record)
        runs.append(record)
    assert counters == {3: 9, 4: 6}
    return runs


def analyze(run):
    """从脱敏动作独立复算费用、扫描后尾程与频道过程；未知真值不参与。"""
    r = run['结果']
    events = r['动作记录']
    actions = [e for e in events if e['动作'] in ('检测', '清除')]
    fixed_end = max(e['步骤'] for e in actions if is_fixed(e))
    markers = [e for e in events if e['动作'] == '固定扫描完成']
    if markers:
        fixed_end = markers[0]['步骤']
    position, channel = [0., 0.], 1
    distance = virtual = 0.
    parts = Counter({'移动': 0., '检测': 0., '切换': 0., '清除成功': 0., '清除失败': 0.})
    cleared = set()
    measurements = switches = failures = certificates = 0
    groups = defaultdict(lambda: Counter({'路程_米': 0., '时间_秒': 0., '清除数': 0}))
    scan_counts, feedback = Counter(), Counter()
    channels = defaultdict(lambda: {'首次发现': None, '追加测向次数': 0, '追加无信号次数': 0, '成功清除': None})
    regions = {c: DirectionalRegion() for c in range(1, 21)} if run['题号'] == 4 else {}
    legs = []
    for e in actions:
        c, p = e['频道'], e['位置']
        travel = math.dist(position, p)
        oldtime = virtual
        distance += travel
        virtual += travel / 5
        parts['移动'] += travel / 5
        tail = e['步骤'] > fixed_end
        kind = '扫描后收尾' if tail else '固定站移动' if is_fixed(e) else '扫描中清除移动' if e['动作'] == '清除' else '追加测向移动'
        groups[kind]['路程_米'] += travel
        if e['动作'] == '检测':
            assert c not in cleared
            measurements += 1
            sw = int(c != channel)
            switches += sw
            channel = c
            virtual += 5 + sw
            parts['检测'] += 5
            parts['切换'] += sw
            feedback[e['结果']] += 1
            purpose = '固定站检测' if is_fixed(e) else e['用途']
            scan_counts[purpose] += 1
            if e['用途'] == '顺路检测':
                assert travel < 1e-6
            if e['结果'] in ('direction', 'near') and channels[c]['首次发现'] is None:
                channels[c]['首次发现'] = {k: e[k] for k in ['步骤', '用途', '位置', '虚拟时间_秒']}
            if e['用途'] == '追加测向':
                channels[c]['追加测向次数'] += 1
                channels[c]['追加无信号次数'] += int(e['结果'] == 'no_signal')
            if regions:
                regions[c].observe(p, e['结果'], e.get('示向度'))
                if '外包顶点' in e:
                    assert np.allclose(regions[c].vertices, e['外包顶点'], rtol=0, atol=1e-7)
        else:
            assert c not in cleared
            success = e['结果'] == 'success'
            virtual += 5 if success else 3
            parts['清除成功' if success else '清除失败'] += 5 if success else 3
            if regions and e['具有覆盖证书']:
                farthest = max_distance(regions[c].vertices, p)
                assert abs(farthest - e['最远可能目标距离_米']) < 1e-5 and farthest < 20
                certificates += 1
            if success:
                cleared.add(c)
                groups[kind]['清除数'] += 1
                channels[c]['成功清除'] = {'步骤': e['步骤'], '位置': p, '虚拟时间_秒': e['虚拟时间_秒'], '扫描后': tail}
            else:
                failures += 1
                if regions:
                    regions[c].failed_clear(p)
        groups[kind]['时间_秒'] += virtual - oldtime
        assert abs(virtual-e['虚拟时间_秒']) < .001
        if travel > 1e-6:
            legs.append({'步骤': e['步骤'], '起点': position, '终点': p, '路程_米': travel, '类别': kind, '频道': c, '用途': e['用途'], '结果': e['结果']})
        position = p
    assert cleared == set(r['已清除频道']) and len(cleared) == r['清除数']
    assert abs(distance-r['总路程_米']) < 1e-5 and abs(virtual-r['虚拟总时间_秒']) < .001
    assert (measurements, switches, failures) == (r['检测次数'], r['频道切换次数'], r['清除失败次数'])
    for name, value in parts.items():
        assert abs(value - r['时间分项_秒'][name]) < 1e-5
    assert r['正常退出'] and r['异常'] is None
    if regions:
        assert r['流程正常完成'] and r['已完成搜索站'] == list(range(13))
        assert r['末尾增站评估']['自动新增未知源搜索站数'] == 0
        assert r['全部完成证据'] == (len(cleared) == 16)
        assert not r['已发现未清除频道'] and not r['已证明不存在频道']
        for state in r['频道记录']:
            assert state['不存在证书'] is None
            assert all(x['来源'] == '清除失败' and x['半径'] == 20 for x in state['排除约束'])
    target = run['官方结果']['jammer_count']
    assert len(cleared) <= target
    n_tail = groups['扫描后收尾']['清除数']
    tail_channels = [{'频道': c, **x} for c, x in sorted(channels.items()) if x['成功清除'] and x['成功清除']['扫描后']]
    latest = {}
    for event in events:
        if event['步骤'] <= fixed_end and '外包顶点' in event:
            latest[event['频道']] = event['外包顶点']
    for state in tail_channels:
        # 仅对本次没有清除失败的记录使用末次正观测外包。
        assert failures == 0
        state['扫描结束时外包覆盖圆半径_米'] = minimum_circle(latest[state['频道']])[1]
        state['扫描结束时已可安全清除'] = state['扫描结束时外包覆盖圆半径_米'] <= 19.5
    return {'目标数': target, '清除数': len(cleared), '实际全清': len(cleared) == target,
            '路程_米': distance, '时间_秒': virtual, '时间分项_秒': dict(parts), '固定扫描终点步骤': fixed_end,
            '阶段及移动用途': {k: dict(v) for k, v in groups.items()}, '检测用途次数': dict(scan_counts),
            '测量反馈': dict(feedback), '扫描中清除数': len(cleared)-n_tail, '扫描后清除数': n_tail,
            '补测无信号次数': sum(x['追加无信号次数'] for x in channels.values()),
            '首次发现用途': dict(Counter('固定站检测' if x['首次发现']['用途'].startswith(('固定搜索站', '搜索站')) else x['首次发现']['用途'] for x in channels.values() if x['首次发现'])),
            '顺路首次发现频道': [c for c, x in channels.items() if x['首次发现'] and x['首次发现']['用途'] == '顺路检测'],
            '重算通过的清除证书数': certificates, '尾程逐频道': tail_channels,
            '扫描结束时已定位待清除数': sum(x['扫描结束时已可安全清除'] for x in tail_channels),
            '逐频道': [{'频道': c, **x} for c, x in sorted(channels.items())], '移动线段': legs}


def handles():
    items = [Line2D([], [], color=color, lw=2, label=kind) for kind, color in COLORS.items()]
    items += [Line2D([], [], marker=marker, color='none', markerfacecolor=color, markeredgecolor='white', markersize=size, label=label)
              for marker, color, size, label in [('s', '#283E50', 7, '固定检测站'), ('^', '#D88932', 7, '追加测向点'), ('o', '#229E86', 7, '成功清除位置'), ('*', '#172F42', 12, '起点'), ('D', '#9B5B9D', 7, '终点')]]
    items.append(Line2D([], [], color='#836096', ls='--', label='1800米区域边界'))
    return items


def draw(ax, run, detailed=False):
    d = run['诊断']
    ax.add_patch(Circle((0, 0), 1800, fill=False, ls='--', color='#836096', lw=1.3))
    for leg in d['移动线段']:
        a, b = np.array(leg['起点']), np.array(leg['终点'])
        color = COLORS[leg['类别']]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.2, alpha=.85)
        if leg['路程_米'] > 65:
            ax.annotate('', xy=a + .61*(b-a), xytext=a + .44*(b-a), arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': .7, 'mutation_scale': 8})
    actions = [e for e in run['结果']['动作记录'] if e['动作'] in ('检测', '清除')]
    for e in actions:
        p = e['位置']
        if e['动作'] == '清除':
            ax.scatter(*p, marker='o' if e['结果'] == 'success' else 'x', s=38, facecolors='white' if e['步骤'] > d['固定扫描终点步骤'] else '#229E86', edgecolors='#229E86', linewidths=1.3, zorder=6)
            if detailed:
                ax.annotate(str(e['频道']), p, xytext=(5, 6), textcoords='offset points', fontsize=9, color='#206F5F')
        elif e['用途'] == '追加测向':
            ax.scatter(*p, marker='^', s=26, c='#D88932', edgecolors='white', linewidths=.4, zorder=5)
    for i, p in enumerate(run['固定站坐标'], 1):
        ax.scatter(*p, marker='s', s=23, c='#283E50', zorder=4)
        if detailed:
            ax.annotate(f'站{i}', p, xytext=(-6, -14), textcoords='offset points', fontsize=8, color='#283E50')
    ax.scatter(0, 0, marker='*', s=120, c='#172F42', edgecolors='white', linewidths=.6, zorder=8)
    ax.scatter(*actions[-1]['位置'], marker='D', s=46, c='#9B5B9D', edgecolors='white', linewidths=.5, zorder=8)
    bound = max(2000, max(abs(v) for e in actions for v in e['位置'])+180)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect='equal', xlabel='东向 x（米）', ylabel='北向 y（米）', xticks=[-2000, -1000, 0, 1000, 2000], yticks=[-2000, -1000, 0, 1000, 2000])
    ax.grid(color='#DEE5EB', lw=.6)
    ax.set_title(f"{run['编号']} · 清除 {d['清除数']}/{d['目标数']} · 扫描后清除 {d['扫描后清除数']} 个\n路程 {d['路程_米']/1000:.2f} 公里 | 虚拟时间 {d['时间_秒']:.1f} 秒", fontsize=11 if not detailed else 14, pad=10)


def save(fig, name):
    paths = []
    for ext in ['png', 'svg']:
        path = FIGURES / (name + '.' + ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, metadata={'Date': None} if ext == 'svg' else None)
        if ext == 'svg':
            path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        paths.append(path)
    plt.close(fig)
    return paths


def plot(runs):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'svg.hashsalt': 'practice-directory-4', 'font.size': 10})
    paths = []
    for number in [4, 3]:
        group = [r for r in runs if r['题号'] == number]
        rows = 2 if number == 4 else 3
        fig, axes = plt.subplots(rows, 3, figsize=(16, 5.5*rows+1))
        for ax, run in zip(axes.flat, group):
            draw(ax, run)
        title = '第四问：六局实际演练轨迹' if number == 4 else '目录4中的第三问旧日志：九局实际轨迹（单独统计）'
        fig.suptitle(title, fontsize=21, y=.975)
        fig.legend(handles=handles(), loc='lower center', bbox_to_anchor=(.5, .025), ncol=5, fontsize=10, frameon=False)
        fig.text(.5, .012, '箭头为真实移动方向；空心清除点表示扫描后收尾；清除点为机器狗位置，非干扰源精确真值', ha='center', fontsize=10)
        fig.subplots_adjust(left=.065, right=.97, bottom=.13 if number == 4 else .10, top=.90 if number == 4 else .93, wspace=.30, hspace=.36)
        paths += save(fig, '第四问_六局轨迹总览' if number == 4 else '第三问旧日志_九局轨迹总览')
        for run in group:
            fig, ax = plt.subplots(figsize=(10, 11))
            draw(ax, run, True)
            fig.suptitle(f"{run['编号']} 实际轨迹 · {run['官方结果']['case_code']}", fontsize=15, y=.975)
            fig.legend(handles=handles(), loc='lower center', bbox_to_anchor=(.5, .04), ncol=3, fontsize=9, frameon=False)
            fig.text(.5, .015, '站号为访问顺序，圆点旁数字为清除频道；路线依实际动作连接，没有人为闭合', ha='center', fontsize=9)
            fig.subplots_adjust(left=.10, right=.95, bottom=.23, top=.88)
            paths += save(fig, ('第四问' if number == 4 else '第三问旧日志') + '/' + run['编号'] + '_轨迹')
    q4 = [r for r in runs if r['题号'] == 4]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    xs = np.arange(6)
    bottom = np.zeros(6)
    for part, color in [('移动', '#438CB0'), ('检测', '#56B8B3'), ('切换', '#DBAC51'), ('清除成功', '#229E86')]:
        values = np.array([r['诊断']['时间分项_秒'][part] for r in q4])
        axes[0].bar(xs, values, bottom=bottom, color=color, label=part)
        bottom += values
    for i, v in enumerate(bottom):
        axes[0].text(i, v+70, f'{v:.0f}', ha='center')
    axes[0].set(title='虚拟时间构成', ylabel='秒', xticks=xs, xticklabels=[r['编号'] for r in q4], ylim=(0, 6800))
    axes[0].legend(ncol=4, fontsize=9, loc='upper left')
    bottom = np.zeros(6)
    for label, color in [('扫描中', '#438CB0'), ('扫描后', '#9B5B9D')]:
        tail = np.array([r['诊断']['阶段及移动用途']['扫描后收尾']['路程_米']/1000 for r in q4])
        values = tail if label == '扫描后' else np.array([r['诊断']['路程_米']/1000 for r in q4])-tail
        axes[1].bar(xs, values, bottom=bottom, color=color, label=label)
        if label == '扫描后':
            for i, v in enumerate(values):
                if v > .2:
                    axes[1].text(i, bottom[i]+v/2, f'{v:.2f}', ha='center', va='center', color='white', fontsize=10)
        bottom += values
    axes[1].axhline(11.4, ls='--', color='#856A3E', label='只访问13站：11.4公里')
    axes[1].set(title='实际路程及扫完固定站后的收尾', ylabel='公里', xticks=xs, xticklabels=[r['编号'] for r in q4], ylim=(0, 26))
    axes[1].legend(ncol=2, fontsize=9, loc='upper left')
    fig.suptitle('第四问六局耗时与尾程诊断', fontsize=18)
    fig.text(.5, .025, '扫描中包含固定站、智能补测和顺路清除；11.4公里不是完整任务的最优路程', ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, .94))
    paths += save(fig, '第四问_耗时与尾程')
    return paths


def report(runs):
    q4 = [r for r in runs if r['题号'] == 4]
    n = len(q4)
    avg = lambda key: np.mean([r['诊断'][key] for r in q4])
    parts = {k: np.mean([r['诊断']['时间分项_秒'][k] for r in q4]) for k in q4[0]['诊断']['时间分项_秒']}
    lines = ['# 目录4：第四问实测结果与轨迹', '',
             '本报告只分析用户提供的既有演练文件，未运行模拟器或修改策略。目录共有15组文件：6局第四问、9局第三问旧记录，各含jlog、psum、result.json。编号按各题结束时间排序；两题分别统计。', '',
             '## 1. 第四问结果', '',
             f'6局均全清，共清除79/79个干扰源，均完成13站扫描，无清除失败、通信重试或通信异常。平均路程{avg("路程_米")/1000:.3f}公里，平均虚拟时间{avg("时间_秒"):.3f}秒。这里的全清由结果文件目标总数与HTTP成功清除的不同频道数核对得出，不是程序仅凭13站自行证明。', '',
             '| 编号 | 案例码 | 全向/定向 | 清除/目标 | 路程（公里） | 虚拟时间（秒） | 检测次数 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in q4:
        d, o = r['诊断'], r['官方结果']
        lines.append(f'| {r["编号"]} | {o["case_code"]} | {o["omnidirectional_jammer_count"]}/{o["directional_jammer_count"]} | {d["清除数"]}/{d["目标数"]} | {d["路程_米"]/1000:.3f} | {d["时间_秒"]:.3f} | {r["结果"]["检测次数"]} |')
    lines += ['', '![第四问六局轨迹](../../results/figures/第四问/实测4目录/第四问_六局轨迹总览.png)', '',
              '图上固定方块是检测站，橙色三角形是追加测向点，绿色圆是成功清除时机器狗的位置；空心圆表示固定扫描结束后清除。紫色路径为扫描后收尾，箭头为实际方向。清除位置与干扰源真值可有20米以内的差异，未用清除位置冒充精确目标坐标。', '',
              '## 2. 路程与时间花在哪里', '',
              f'平均移动时间{parts["移动"]:.1f}秒，占总时间{parts["移动"]/avg("时间_秒")*100:.2f}%；检测与切换合计{parts["检测"]+parts["切换"]:.1f}秒，占{(parts["检测"]+parts["切换"])/avg("时间_秒")*100:.2f}%。主要成本仍是移动，检测也不能忽略。', '',
              '| 编号 | 扫描中清除 | 扫描后清除 | 尾程（公里） | 尾程时间（秒） | 主动补测次数 | 其中无信号 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in q4:
        d = r['诊断']; t = d['阶段及移动用途']['扫描后收尾']
        lines.append(f'| {r["编号"]} | {d["扫描中清除数"]} | {d["扫描后清除数"]} | {t["路程_米"]/1000:.3f} | {t["时间_秒"]:.3f} | {d["检测用途次数"].get("追加测向",0)} | {d["补测无信号次数"]} |')
    lines += ['', '![费用与尾程](../../results/figures/第四问/实测4目录/第四问_耗时与尾程.png)', '',
              '尾程以“固定扫描完成”事件为界，包含此后对已发现源的补测、清除及相关移动；它不是返回起点的路程。移动用途按目的动作归类，只用于费用归属，不能把所有去补测的线段都称为可删除的浪费。', '',
              '以下逐局数据用于定位具体问题：', '']
    for r in q4:
        d = r['诊断']; tail = d['阶段及移动用途']['扫描后收尾']
        channels = '、'.join(str(x['频道']) for x in d['尾程逐频道']) or '无'
        first = '、'.join(map(str, d['顺路首次发现频道'])) or '无'
        lines.append(f'- **{r["编号"]}**：扫完13站后清除频道{channels}，尾程{tail["路程_米"]/1000:.3f}公里；主动补测{d["检测用途次数"].get("追加测向",0)}次，其中{d["补测无信号次数"]}次无信号。首次由顺路检测发现的频道：{first}。')
    lines += ['', '## 3. 如何解释这组结果', '',
              '1. **13站加现有自适应机制，在本次6局中全部成功。** 这支持继续保留当前实验版本；不是任意位置、朝向都能发现的证明，也不能据此认定每个目标均被13个固定站中的两个接收到。顺路检测和主动补测也参与了发现、定位，目标清除后不会再在后续固定站接收。',
              '2. **顺路清除确实工作，但收尾仍有明显差异。** P4-4只有5个目标在扫描中清除，8个留到末尾；P4-5的14个目标均在扫描中清除。固定站顺序虽然达到纯遍历11.4公里下界，却没有保证加入动态目标后的总路线最短。',
              '3. **定向条件下，主动选出的几何补测点可能无信号。** 无信号动作仍要付移动、检测和切换费用。P4-2、P4-3、P4-4、P4-6均出现多次这种反馈；但不能仅凭无信号判断超距还是位于发射背侧，也不能凭6个不同案例断言定向源数量直接决定时间。',
              '4. **结果字段要区分在线证明与事后核对。** 只有清除16源的P4-3具有在线全清证据。其余5局的“运行成功=false”代表未获得策略定义的全清证据；它们流程正常且经目标总数核对实际全清，不是异常退出。未发现频道可以实际不存在。', '',
              '**两个具体瓶颈：** P4-4尾程6.913公里、耗时1466.585秒；其中频道8、9、13、17在扫描结束时已定位，最小覆盖圆半径分别约17.77、17.59、16.90、14.76米，均低于19.5米安全阈值，其余4个仍需补测。P4-3尾程更长，为7.084公里、1501.807秒，6个收尾目标中3个在扫描结束时已定位。这里只证明扫描结束时的状态，并未证明它们在更早经过附近时也已经满足清除条件。', '',
              'P4-6尾程只处理频道11和16，却走了4.945公里、耗时1125.954秒。频道11在末尾连续7次主动补测无信号，第8次收到方向；频道16连续6次主动补测无信号，第7次收到方向（期间可能夹有其他频道的顺路检测）。合计15次主动补测中13次无信号，说明补测接收效率低。P4-2频道3及P4-6频道16还进入了有限方格光学覆盖，均在第一处清除便成功；这两次不带单点覆盖证书，是正常回退分支，不属于清除失败。其余77次成功清除的覆盖证书均已重算通过。', '',
              f'六局合计主动补测{sum(r["诊断"]["检测用途次数"].get("追加测向",0) for r in q4)}次，其中无信号{sum(r["诊断"]["补测无信号次数"] for r in q4)}次。六局有{sum(r["诊断"]["首次发现用途"].get("顺路检测",0) for r in q4)}个目标首次通过顺路检测发现；这说明实测验证的是“13站＋自适应机制”的组合，不能把成果全部归因于13个固定站。', '',
              '后续若优化，优先考虑“更早处理预计会留在身后的目标”和“利用已有正负反馈约束定向朝向，提高补测接收概率”，并用相同场景对照验证。未作反事实计算前，不能把本表尾程全部当作可节约路程。本轮不改参数、不增加外围站、不修改既有策略。', '',
              '## 4. 图集与复现', '',
              '| 第四问 | PNG | SVG |', '|---|---|---|']
    for r in q4:
        prefix = '../../results/figures/第四问/实测4目录/第四问/' + r['编号'] + '_轨迹'
        lines.append(f'| {r["编号"]} | [查看]({prefix}.png) | [矢量图]({prefix}.svg) |')
    lines += ['', '从项目根目录执行：', '', '```powershell',
              r'.\.venv\Scripts\python.exe -X utf8 scripts/analyze_q4_practice_4.py --verify',
              '# 使用归档动作重绘全部图表，不连接模拟器',
              r'.\.venv\Scripts\python.exe -X utf8 scripts/analyze_q4_practice_4.py',
              '# 在保留本地原日志的电脑上，额外核对原文件SHA256',
              r'.\.venv\Scripts\python.exe -X utf8 scripts/analyze_q4_practice_4.py --verify --source-check',
              '```', '',
              '首次采集使用`--collect`；归档后的重绘和费用复核无需原始私有HTTP日志。统计见[评估JSON](../../results/tables/第四问/实测4目录_评估.json)，脱敏动作见[归档目录](../../results/models/第四问/实测4目录)。', '',
              '## 5. 数据来源与验证边界', '',
              '第四问运行版本均为`eec554a`，运行时Git工作区干净；逐文件源码与配置SHA256已核对。jlog公开头、psum公开头及result.json的题号、案例码、演练号一致，结果文件所记包哈希与jlog一致；同队号和唯一退出时间匹配本地HTTP记录。公开头不是密文内容证明，本报告未解密正文、未验证包签名。', '',
              'HTTP请求逐动作与策略的动作、频道、位置、结果、示向度及虚拟时间核对；独立按距离÷5、检测5秒、切换1秒、清除成功5秒/失败3秒复算。第四问额外重建观测外包并复核清除证书，确认无信号未生成错误的1000米排除圆。官方文件没有提供可读取的精确目标位置与发射轴，不能重算隐藏真值物理反馈或在图中标出真值。', '',
              '本次全部为既有演练，不是正式测试成绩，不是与第三问相同地图的配对实验。以下第三问只作目录完整性归档。', '',
              '## 附录：同目录9局第三问旧日志', '',
              '这些记录来自2026-09-11，均为第三问方案一路线版，与2026-09-12的六局第四问分别统计。', '',
              '| 编号 | 案例码 | 清除/目标 | 路程（公里） | 虚拟时间（秒） | 轨迹 |', '|---|---|---:|---:|---:|---|']
    for r in runs:
        if r['题号'] != 3:
            continue
        d, o = r['诊断'], r['官方结果']
        prefix = '../../results/figures/第四问/实测4目录/第三问旧日志/' + r['编号'] + '_轨迹'
        lines.append(f'| {r["编号"]} | {o["case_code"]} | {d["清除数"]}/{d["目标数"]} | {d["路程_米"]/1000:.3f} | {d["时间_秒"]:.3f} | [PNG]({prefix}.png) / [SVG]({prefix}.svg) |')
    q3 = [r for r in runs if r['题号'] == 3]
    lines += ['', f'第三问合计清除{sum(r["诊断"]["清除数"] for r in q3)}/{sum(r["诊断"]["目标数"] for r in q3)}个源，平均路程{np.mean([r["诊断"]["路程_米"] for r in q3])/1000:.3f}公里，平均虚拟时间{np.mean([r["诊断"]["时间_秒"] for r in q3]):.3f}秒。第三问尾程以最后一次固定站检测为界，与其原日志结构一致。', '',
              '![第三问九局旧日志轨迹](../../results/figures/第四问/实测4目录/第三问旧日志_九局轨迹总览.png)', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def verify_sources(runs):
    for run in runs:
        for name, expected in run['来源SHA256'].items():
            assert digest(ROOT / name) == expected, name
        if run['题号'] == 4:
            for name, expected in run['结果']['运行来源']['源码_SHA256'].items():
                assert matches_source(ROOT, name, expected), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--source-check', action='store_true')
    args = parser.parse_args()
    if args.collect and args.verify:
        parser.error('--collect与--verify不能同时使用')
    if args.collect:
        runs = collect()
    else:
        old = read(TABLE)
        for name, expected in old['归档动作SHA256'].items():
            assert digest(ROOT / name) == expected, name
        runs = [read(ROOT / name) for name in old['归档动作SHA256']]
        for run in runs:
            assert analyze(run) == run['诊断'], run['编号']
    if args.source_check:
        verify_sources(runs)
    if args.verify:
        for name, expected in {**old['图表SHA256'], **old['分析代码SHA256'], **old['报告SHA256']}.items():
            assert matches_source(ROOT, name, expected), name
        print('核验通过：15局脱敏动作与费用、6局第四问外包与清除证书、图表及报告哈希。')
        return
    figures = plot(runs)
    report(runs)
    data = {'数据范围': {'第四问': 6, '第三问旧记录': 9, '原目录文件数': 45},
            '说明': '既有演练；无HTTP调用；题号分组；轨迹为机器狗位置，未知目标真值未绘制。',
            '逐局摘要': [{'编号': r['编号'], '题号': r['题号'], '案例码': r['官方结果']['case_code'],
                        **{k: v for k, v in r['诊断'].items() if k not in ('移动线段', '逐频道')}} for r in runs],
            '归档动作SHA256': {relative(MODELS / (r['编号'] + '.json')): digest(MODELS / (r['编号'] + '.json')) for r in runs},
            '图表SHA256': {relative(p): digest(p) for p in figures},
            '报告SHA256': {relative(REPORT): digest(REPORT)},
            '分析代码SHA256': {relative(p): digest(p) for p in [Path(__file__), ROOT / 'scripts/analyze_q3_practice_log.py', ROOT / 'scripts/evaluate_q3_routed_practice.py']}}
    write(TABLE, data)
    print('已生成15张逐局轨迹、两张总览及一张耗时图，各含PNG/SVG；未运行模拟器。')
    for r in runs:
        d = r['诊断']
        print(r['编号'], d['清除数'], '/', d['目标数'], f'{d["路程_米"]/1000:.3f}公里 {d["时间_秒"]:.3f}秒', d['阶段及移动用途']['扫描后收尾'])


if __name__ == '__main__':
    main()
