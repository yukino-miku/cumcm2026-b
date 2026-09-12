#!/usr/bin/env python
"""只读分析新建文件夹(2)六局第四问演练，重算费用、绘制轨迹及评估改进空间。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
from PIL import Image
from analyze_q4_practice_4 import (matching_logs, public_header, read, write, digest, relative,
                                   audit_actions, audit_trace, is_fixed, RESULT_KEYS, COLORS)
from analyze_q4_spacing_practice import summarize, draw, handles, extra_analysis
from build_q4_inner13_assets import setup_font
from q4_archive_sources import matches_source, verify_manifest
from cumcm2026_b.q4_strategy import DirectionalRegion, DirectionalChannel, search_stations
from cumcm2026_b.q3_geometry import max_distance, min_distance, minimum_circle, sweep_path
from cumcm2026_b.q3_geometry import principal_axes, finish_bound, angular_interval, add_wedge, DELTA
from cumcm2026_b.q4_feedback_strategy import joint_hypotheses, reception_support, task_projection

SOURCE = ROOT / 'materials/original/simulator/CUMCM2026B/Jammers-simulator-win64/Jammers-simulator/JammersSimulatorData/behavior-logs/新建文件夹 (2)'
MODELS = ROOT / 'results/models/第四问/实测方向反馈_新建文件夹2'
FIG = ROOT / 'results/figures/第四问/实测方向反馈_新建文件夹2'
TABLE = ROOT / 'results/tables/第四问/实测方向反馈_新建文件夹2_评估.json'
MANIFEST = ROOT / 'results/tables/第四问/实测方向反馈_新建文件夹2_来源校验.json'
REPORT = ROOT / 'docs/第四问/实测方向反馈_新建文件夹2分析.md'
PAPER = ROOT / 'paper/sections/第四问_方向反馈六局实测_论文备用说明.md'
EXPECTED_COMMIT = '6dddfd59bc359af87685c22b7b2cd79fab42919f'


# 费用审计沿用此前实测脚本；新增达到16源上界后提前终止的合法分支。
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
    regions = {c: DirectionalRegion() for c in range(1, 21)}  # 本批七局实际都执行Q4策略；题号仍按官方文件分类。
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
        assert r['流程正常完成']
        completed = r['已完成搜索站']
        assert completed == list(range(len(completed))) and 1 <= len(completed) <= 13
        assert len(completed) == 13 or len(cleared) == 16  # 达16源上界可提前结束，不能伪补第13站。
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
        # 光学失败只增加排除圆，不改变凸外包顶点；该半径仍是忽略排除圆后的安全上界。
        vertices = latest.get(state['频道'])
        radius = minimum_circle(vertices)[1] if vertices is not None else None
        state['扫描结束时外包覆盖圆半径_米'] = radius
        state['扫描结束时已可安全清除'] = radius is not None and radius <= 19.5
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


def extension_audit(run):
    """只用观测历史复算新增机制，未知源真值不参与。"""
    r = run['结果']; events = r['动作记录']
    actions = [e for e in events if e['动作'] in ('检测', '清除')]
    spacing = extra_analysis(r)
    states = {c: DirectionalChannel(c) for c in range(1, 21)}
    current = np.array([0., 0.]); tuned = 1; pending = []
    extra_scans, feedback, small, delays, impossible = [], [], [], [], []
    plan = None
    for e in events:
        c = e.get('频道')
        if e['动作'] == '补齐顺路未知频道':
            assert not pending
            pending = list(e['候选频道'])
            for channel in pending:
                s = states[channel]
                assert s.status == 'unknown'
                assert not s.measured_sites or min(math.dist(e['位置'], p) for p in s.measured_sites) >= 600 - 1e-6
        elif e['动作'] == '规划':
            plan = e
        elif e['动作'] == '延后目标':
            delays.append(e)
        elif e['动作'] == '顺路小范围覆盖规划':
            vertices = states[c].region.vertices
            assert np.allclose(e['外包顶点'], vertices, rtol=0, atol=1e-7)
            points = np.array(e['位置序列'])
            original = sweep_path(vertices, current)
            assert len(points) == len(original) == e['覆盖点数'] <= 4
            assert all(np.linalg.norm(original - p, axis=1).min() < 1e-6 for p in points)
            assert len({tuple(p) for p in points}) == len(points)
            length = math.dist(current, points[0]) + np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            detour = length + math.dist(points[-1], e['下一固定站']) - math.dist(current, e['下一固定站'])
            assert abs(detour - e['完整绕行_米']) < 1e-6 and detour <= 800 + 1e-6
            following = [a for a in actions if a['步骤'] > e['步骤'] and a['频道'] == c and a['用途'] == '顺路小范围覆盖']
            assert following[-1]['结果'] == 'success'
            assert np.allclose(points[:len(following)], [a['位置'] for a in following], atol=1e-6, rtol=0)
            small.append({'频道': c, '步骤': e['步骤'], '计划点数': len(points), '实际清除尝试': len(following),
                          '完整绕行上界_米': detour, '额外费用上界_秒': e['增加费用上界_秒'],
                          '实际任务时间_秒': following[-1]['虚拟时间_秒'] - e['虚拟时间_秒']})
        elif e['动作'] == '检测':
            s = states[c]
            if s.status == 'found' and (is_fixed(e) or e['用途'] == '顺路检测'):
                lower_distance = min_distance(s.region.vertices, e['位置'])
                if lower_distance > 1500 + 1e-6:
                    assert e['结果'] == 'no_signal'
                    impossible.append({'步骤': e['步骤'], '频道': c, '用途': e['用途'],
                                       '当时距离下界_米': lower_distance,
                                       '单次检测费用_秒': 5, '原单次切换费用_秒': int(c != tuned)})
            if pending:
                assert c == pending.pop(0) and e['用途'] == '顺路检测' and s.status == 'unknown'
                extra_scans.append({'步骤': e['步骤'], '频道': c, '位置': e['位置'], '结果': e['结果'],
                                    '实际费用_秒': 5 + int(c != tuned),
                                    '首次发现': e['结果'] != 'no_signal'})
            if e['用途'] == '追加测向':
                assert plan is not None and plan['频道'] == c
                p = plan['方案']
                bound = max_distance(s.region.vertices, e['位置'])
                feedback.append({'步骤': e['步骤'], '频道': c, '结果': e['结果'],
                                 '方向反馈已启用': p.get('方向反馈已启用', False),
                                 '接收支持度': p.get('接收样本支持度'), '当时距离上界_米': bound,
                                 '无信号且可排除超距': e['结果'] == 'no_signal' and bound <= 999.5 + 1e-6})
                s.localization_count += 1
            s.region.observe(e['位置'], e['结果'], e.get('示向度'))
            s.measured_sites.append(e['位置'])
            if e['结果'] != 'no_signal': s.status = 'found'
            tuned = c; current = np.asarray(e['位置'])
        elif e['动作'] == '清除':
            if e['结果'] == 'success': states[c].status = 'cleared'
            else: states[c].region.failed_clear(e['位置'])
            current = np.asarray(e['位置'])
    assert not pending
    assert all(s.localization_count <= 8 for s in states.values())
    unknown = []
    for c in r['未发现频道']:
        measured = [e for e in actions if e['动作'] == '检测' and e['频道'] == c]
        assert all(e['结果'] == 'no_signal' for e in measured)
        assert sum(is_fixed(e) for e in measured) == len(r['已完成搜索站'])
        unknown.append({'频道': c, '固定站检测数': sum(is_fixed(e) for e in measured),
                        '顺路检测数': sum(e['用途'] == '顺路检测' for e in measured),
                        '总检测数': len(measured)})
    return {'间距与有限覆盖': spacing, '小范围顺路任务': small, '逐次主补测': feedback,
            '方向反馈选点数': sum(f['方向反馈已启用'] for f in feedback),
            '方向反馈选点无信号数': sum(f['方向反馈已启用'] and f['结果'] == 'no_signal' for f in feedback),
            '补齐未知频道实际检测': extra_scans, '补齐检测数': len(extra_scans),
            '补齐检测实际费用_秒': sum(e['实际费用_秒'] for e in extra_scans),
            '补齐首次发现频道': [e['频道'] for e in extra_scans if e['首次发现']],
            '延后目标': delays, '已知目标可证明超距的附带检测': impossible,
            '结束时未知频道检测情况': unknown}


def collect():
    index = matching_logs(); runs = []
    files = sorted(SOURCE.glob('*.result.json'), key=lambda p: read(p)['ended_at_utc'])
    assert len(files) == 6 and len(list(SOURCE.iterdir())) == 18
    for i, file in enumerate(files, 1):
        official = read(file); stem = file.name.removesuffix('.result.json')
        jlog, psum = SOURCE / (stem + '.jlog'), SOURCE / (stem + '.psum')
        header, public = public_header(jlog); raw = psum.read_bytes()
        assert raw[:8] == b'JMBPSUM1' and int.from_bytes(raw[8:10], 'big') == 1
        length = int.from_bytes(raw[10:14], 'big'); assert 0 < length <= len(raw) - 14
        ph = json.loads(raw[14:14 + length])
        assert header['package_type'] == 'practice_behavior_log' and ph['package_type'] == 'practice_summary'
        assert header['formal_index'] is None and ph['formal_index'] is None
        assert header['team_no'] == ph['team_no']
        for key in ['problem_no', 'practice_run_no', 'case_code']:
            assert official[key] == header[key] == ph[key]
        assert official['problem_no'] == 4 and official['package_sha256'] == digest(jlog)
        assert official['jammer_count'] == official['omnidirectional_jammer_count'] + official['directional_jammer_count']
        timestamp = datetime.fromisoformat(header['created_at_utc'].replace('Z', '+00:00')).timestamp() * 1000
        matches = [x for x in index if x[1] == header['team_no'] and abs(timestamp - x[2]) <= 2000]
        assert len(matches) == 1, (stem, len(matches))
        number, _, exited, folder, items = matches[0]
        r = read(folder / '运行结果.json')
        audit = audit_actions(items, r); actions = audit_trace(items, r)
        responses = {x['request_id']: x['response'] for x in items if x.get('类型') == '响应'}
        requests = {x['payload']['request_id']: x for x in items if x.get('类型') == '请求' and x['path'] in ('/measure', '/clear')}
        for (rid, req), event in zip(requests.items(), actions):
            if req['path'] == '/measure': assert event['示向度'] == responses[rid].get('svd_deg')
        meta = r['运行来源']
        assert meta['Git提交'] == EXPECTED_COMMIT and meta['Git工作区干净'] and meta['策略入口'] == 'feedback'
        for name, value in meta['源码_SHA256'].items(): assert matches_source(ROOT, name, value), name
        assert r['配置'] == read(ROOT / 'configs/q4_feedback.json')
        assert number == 4 and np.allclose(r['固定站坐标'], search_stations(), atol=1e-8, rtol=0)
        record = {'编号': f'F4-{i}', '题号': 4, '官方结果': official, '公开头摘要': public,
                  '关联': {'本地目录': relative(folder), '同队号': True, '生成减退出响应_毫秒': timestamp - exited,
                           '依据': '同队号、唯一退出时间及协议逐动作核查；未解密正文或验证包签名'},
                  '协议核验': audit, '计划固定站坐标': r['固定站坐标'],
                  '固定站坐标': [r['固定站坐标'][k] for k in r['已完成搜索站']],
                  '结果': {k: r[k] for k in RESULT_KEYS},
                  '来源SHA256': {relative(p): digest(p) for p in [file, jlog, psum, folder / '运行结果.json', folder / '请求响应日志.jsonl']}}
        record['诊断'] = analyze(record)
        record['改进机制核验'] = extension_audit(record)
        write(MODELS / (record['编号'] + '.json'), record); runs.append(record)
    return runs


def aggregate(runs):
    summary = summarize(runs)
    summary.update({'方向反馈选点数': sum(r['改进机制核验']['方向反馈选点数'] for r in runs),
                    '方向反馈无信号数': sum(r['改进机制核验']['方向反馈选点无信号数'] for r in runs),
                    '小范围顺路清除目标数': sum(len(r['改进机制核验']['小范围顺路任务']) for r in runs),
                    '补齐未知频道检测数': sum(r['改进机制核验']['补齐检测数'] for r in runs),
                    '补齐检测费用_秒': sum(r['改进机制核验']['补齐检测实际费用_秒'] for r in runs),
                    '补齐扫描首次发现数': sum(len(r['改进机制核验']['补齐首次发现频道']) for r in runs)})
    return summary


def candidate_diagnostic(run, channel, step):
    """重建被延后时的候选池，比较先约束再评分；不调用机器人或引入后续观测。"""
    state = DirectionalChannel(channel); current = np.zeros(2)
    events = run['结果']['动作记录']
    for e in events:
        if e['步骤'] >= step: break
        if e['动作'] in ('检测', '清除'): current = np.asarray(e['位置'])
        if e['动作'] == '检测' and e['频道'] == channel:
            state.region.observe(e['位置'], e['结果'], e.get('示向度'))
            state.measured_sites.append(e['位置'])
            state.localization_count += e['用途'] == '追加测向'
    last_route = next(e for e in reversed(events) if e['步骤'] < step and e['动作'] == '路线规划')
    remaining = np.asarray([p['位置'] for p in last_route['计划路径'] if p['类型'] == '固定站'])
    hypotheses = joint_hypotheses(state, 41); assert hypotheses and len(remaining)
    vertices = state.region.vertices; center, axes = principal_axes(vertices)
    anchor = np.asarray([o for o in state.region.observations if o['结果'] == 'direction'][-1]['位置'])
    ray = center - anchor; ray /= max(np.linalg.norm(ray), 1e-9); lateral = np.array([-ray[1], ray[0]])
    candidates = [center + axes @ np.array([math.cos(t), math.sin(t)]) * d
                  for t in np.arange(8) * math.pi / 4 for d in (75, 150, 300, 500, 650)]
    candidates += [anchor + ray * forward + lateral * side for forward in (0, 100, 250, 450)
                   for side in (-300, -150, -75, 75, 150, 300)]
    feasible = []
    for p in candidates:
        if min_distance(vertices, p) <= 5.5 or min_distance(vertices, p) > 1000: continue
        if any(np.linalg.norm(p - q) < .01 for q in state.measured_sites + feasible): continue
        if np.linalg.norm(p - current) < 100 - 1e-7 or np.linalg.norm(p - state.measured_sites[-1]) < 100 - 1e-7: continue
        feasible.append(p)
    masses = [sum(b - a for a, b in arcs) for _, arcs in hypotheses]
    representative = np.average([p for p, _ in hypotheses], axis=0, weights=masses)
    rows = []
    for p in feasible:
        support = reception_support(hypotheses, p)
        if support <= 1e-8: continue
        proj = task_projection(current, p, representative, remaining)
        low, high = angular_interval(vertices, p); predictions = []
        for theta in np.linspace(low - math.radians(DELTA), high + math.radians(DELTA), 9):
            branch = add_wedge(vertices, p, math.degrees(theta))
            if len(branch): predictions.append(finish_bound(branch, p))
        if not predictions: continue
        detour = math.dist(current, p) + math.dist(p, remaining[0]) - math.dist(current, remaining[0])
        score = (math.dist(current, p) / 5 + 6 + support * max(predictions)
                 + (1 - support) * finish_bound(vertices, p)
                 + (math.dist(representative, remaining[0]) - math.dist(current, remaining[0])) / 5)
        admissible = detour <= 800 and (proj['预测完整任务绕行_米'] <= 800
                                       or proj['预测完整任务绕行_米'] <= proj['预测延后最小绕行_米'] + 1e-6)
        rows.append({'位置': p.tolist(), '补测绕行_米': detour, '候选评分_秒': score,
                     '接收支持度': support, '距离上界_米': max_distance(vertices, p),
                     '初筛评分_秒': proj['预测完整任务绕行_米'] / 5 + (1 - support) * finish_bound(vertices, p),
                     '满足当前两项路线门槛': bool(admissible), **proj})
    rows.sort(key=lambda x: (x['初筛评分_秒'], -x['接收支持度'], *x['位置']))
    old = min(rows[:12], key=lambda x: x['候选评分_秒'])
    rejection = next(e for e in events if e['步骤'] == step)
    assert abs(old['补测绕行_米'] - rejection['本次测向绕行_米']) < 1e-5
    filtered = [p for p in rows if p['满足当前两项路线门槛']]
    best = min(filtered[:12], key=lambda x: x['候选评分_秒']) if filtered else None
    return {'编号': run['编号'], '频道': channel, '决策步骤': step, '当前位置': current.tolist(),
            '下一固定站': remaining[0].tolist(), '当前正观测外包': vertices.tolist(),
            '位置假设数': len(hypotheses), '全部候选数': len(rows), '可行候选数': len(filtered),
            '重现原被拒候选': old, '先筛路线约束再评分的候选': best,
            '性质': '仅使用该步骤之前的历史重算候选；没有测量替代点，不承诺接收、清除或整局节省'}


def skipped_scan_counterexample(run):
    """寻找与全部负反馈一致的构造源，证明600米相近不代表方向信息重复。"""
    r = run['结果']; actions = [e for e in r['动作记录'] if e['动作'] in ('检测', '清除')]
    groups = []
    for e in actions:
        if not groups or math.dist(groups[-1][0]['位置'], e['位置']) > 1e-6: groups.append([])
        groups[-1].append(e)
    unknown = r['未发现频道']; prior = {c: [] for c in unknown}; skipped = []
    for group in groups:
        point = group[0]['位置']
        opportunity = any(e['用途'] == '追加测向' or e['动作'] == '清除' and e['结果'] == 'success' for e in group)
        if opportunity and not any(is_fixed(e) for e in group):
            measured = {e['频道'] for e in group if e['动作'] == '检测'}
            for c in unknown:
                if c in measured or not prior[c]: continue
                nearest = min(prior[c], key=lambda p: math.dist(p, point))
                gap = math.dist(nearest, point)
                if gap < 600:
                    skipped.append({'步骤': group[0]['步骤'], '频道': c, '停点': point,
                                    '最近该频道旧测量站': nearest, '距离_米': gap})
        for e in group:
            if e['动作'] == '检测' and e['频道'] in prior: prior[e['频道']].append(e['位置'])
    positions = np.array([(x, y) for x in range(-1700, 1701, 100) for y in range(-1700, 1701, 100)
                          if x*x + y*y <= 1790**2], dtype=float)
    angles = np.arange(72) * 5.; axes = np.stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))], axis=1)
    for c in unknown:
        sites = np.array(prior[c]); possible = np.ones((len(positions), len(axes)), dtype=bool)
        for site in sites:
            vec = site - positions; near = np.linalg.norm(vec, axis=1) <= 1000.5
            # 保留严格背侧或明确超距，避免把边界舍入误差当反例。
            possible &= ~near[:, None] | (vec @ axes.T < -1.)
        for stop in (p for p in skipped if p['频道'] == c):
            vec = np.array(stop['停点']) - positions
            heard = (np.linalg.norm(vec, axis=1) <= 999.5)[:, None] & (vec @ axes.T > 1.)
            index = np.argwhere(possible & heard)
            if len(index):
                p, u = index[0]; source = positions[p]; axis = axes[u]
                details = [{'测量站': site.tolist(), '距离_米': math.dist(site, source),
                            '前向投影_米': float((site - source) @ axis)} for site in sites]
                assert all(d['距离_米'] > 1000 or d['前向投影_米'] < 0 for d in details)
                assert math.dist(stop['停点'], source) < 1000 and (np.array(stop['停点']) - source) @ axis > 0
                return {'候选未扫描频道停点对数': len(skipped), '例子': {**stop,
                        '构造源位置': source.tolist(), '构造发射轴_度': float(angles[u]), '构造半径_米': 1000,
                        '全部实际该频道测量': details},
                        '性质': '这是与全部该频道无信号记录相容的假设反例，绝不是实际遗漏源的坐标或频道识别'}
    return {'候选未扫描频道停点对数': len(skipped), '例子': None,
            '性质': '有限位置与朝向采样未找到反例不等于信息重复或完整发现保证'}


def evaluation(runs):
    old = [read(ROOT / f'results/models/第四问/实测100米间距/N4-{i}.json') for i in range(1, 7)]
    assert {r['官方结果']['case_code'] for r in runs}.isdisjoint(r['官方结果']['case_code'] for r in old)
    candidates = [candidate_diagnostic(runs[2], 10, 197), candidate_diagnostic(runs[3], 8, 264)]
    skipped = {r['编号']: skipped_scan_counterexample(r) for r in (runs[1], runs[5])}
    unnecessary = [dict(编号=r['编号'], **e) for r in runs for e in r['改进机制核验']['已知目标可证明超距的附带检测']]
    return {'性质': '六局第四问实际演练；只读已有协议，无新增模拟器调用；旧批地图不同',
            '本批汇总': aggregate(runs), '上批汇总': summarize(old),
            '两批定向源数量': {'上批': sum(r['官方结果']['directional_jammer_count'] for r in old),
                               '本批': sum(r['官方结果']['directional_jammer_count'] for r in runs)},
            '被延后时可行候选核查': candidates, '600米间隔的局限反例': skipped,
            '可证明超距的已知频道附带检测': unnecessary,
            '纯检测费用_秒': 5 * len(unnecessary),
            '逐局': [{'编号': r['编号'], '案例码': r['官方结果']['case_code'],
                     **{k: r['诊断'][k] for k in ['目标数', '清除数', '实际全清', '路程_米', '时间_秒',
                                                 '检测用途次数', '补测无信号次数', '扫描后清除数', '时间分项_秒',
                                                 '阶段及移动用途']},
                     '固定站访问数': len(r['固定站坐标']),
                     '小范围清除数': len(r['改进机制核验']['小范围顺路任务']),
                     '补齐未知检测数': r['改进机制核验']['补齐检测数'],
                     '补齐首次发现数': len(r['改进机制核验']['补齐首次发现频道'])} for r in runs]}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        path = FIG / (name + suffix)
        fig.savefig(path, dpi=165, metadata={'Date': None} if suffix == '.svg' else None)
        if suffix == '.svg': path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    plt.close(fig)


def draw_actual(ax, run, detailed=False):
    draw(ax, run, False)
    if detailed:
        for i, point in enumerate(run['固定站坐标'], 1):
            ax.annotate(f'站{i}', point, xytext=(-6, -14), textcoords='offset points', fontsize=8)
        clears = [e for e in run['结果']['动作记录'] if e['动作'] == '清除' and e['结果'] == 'success']
        for e in clears:
            point = e['位置']
            neighbors = sorted([a for a in clears if math.dist(point, a['位置']) < 150],
                               key=lambda a: (a['位置'][0], a['位置'][1], a['频道']))
            if len(neighbors) > 1:
                rank = next(i for i, a in enumerate(neighbors) if a['频道'] == e['频道'])
                theta = math.radians(150 - 120 * rank / (len(neighbors) - 1))
                offset = (22 * math.cos(theta), 22 * math.sin(theta))
            else:
                near_station = min(math.dist(point, s) for s in run['固定站坐标']) < 180
                offset = (-18, -16) if near_station else (5, 5)
            ax.annotate(str(e['频道']), point, xytext=offset, textcoords='offset points', fontsize=8,
                        color='#206F5F', ha='right' if offset[0] < 0 else 'left',
                        arrowprops=dict(arrowstyle='-', color='#6da593', lw=.5) if len(neighbors) > 1 else None)
    unvisited = run['计划固定站坐标'][len(run['固定站坐标']):]
    for p in unvisited:
        ax.scatter(*p, marker='s', s=28, facecolors='none', edgecolors='#b3b3b3', zorder=4)
        if detailed: ax.annotate('站13未访问', p, xytext=(-20, -15), textcoords='offset points', fontsize=8, color='#777777')
    d = run['诊断']; n = len(run['固定站坐标'])
    ax.set_title(f"{run['编号']} · 清除{d['清除数']}/{d['目标数']} · 固定站{n}/13\n"
                 f"{d['路程_米']/1000:.2f}公里｜{d['时间_秒']:.1f}秒｜收尾{d['扫描后清除数']}个", fontsize=13 if detailed else 11)


def figures(runs, data):
    setup_font(); plt.rcParams['svg.hashsalt'] = 'q4-feedback-practice-6dddfd5'
    legend = handles() + [Line2D([], [], marker='*', color='#172F42', ls='none', markersize=11, label='中心起点'),
                          Line2D([], [], marker='s', color='#b3b3b3', markerfacecolor='none', ls='none', label='未访问固定站')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    for ax, run in zip(axes.flat, runs): draw_actual(ax, run)
    fig.suptitle('方向反馈改进版六局实际演练：75/77清除，4局全清', fontsize=18)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .035), ncol=6, fontsize=10)
    fig.text(.5, .016, '紫线为固定扫描后收尾；清除位置不是源的精确坐标。第1局达到16源上界，提前结束。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.06, right=.98, bottom=.17, top=.91, wspace=.28, hspace=.37)
    save(fig, '01_六局轨迹总览')
    for run in runs:
        fig, ax = plt.subplots(figsize=(9, 10.4)); draw_actual(ax, run, True)
        fig.suptitle(run['官方结果']['case_code'], fontsize=11, y=.96)
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .04), ncol=3, fontsize=9)
        fig.text(.5, .013, '圆点旁数字为清除频道；空心圆为扫描后清除；按真实动作连接，不人为闭合。', ha='center', fontsize=9)
        fig.subplots_adjust(left=.11, right=.96, bottom=.25, top=.88)
        save(fig, run['编号'] + '_轨迹')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4)); x = np.arange(6); bottom = np.zeros(6)
    for key, color in [('移动', '#438CB0'), ('检测', '#56B8B3'), ('切换', '#DBAC51'), ('清除成功', '#229E86'), ('清除失败', '#b95050')]:
        v = np.array([r['诊断']['时间分项_秒'][key] for r in runs]); axes[0].bar(x, v, bottom=bottom, color=color, label=key); bottom += v
    axes[0].bar_label(axes[0].containers[-1], labels=[f'{v:.0f}' for v in bottom], padding=4)
    axes[0].set(title='总耗时分项', ylabel='虚拟时间（秒）', ylim=(0, 6800)); axes[0].legend(fontsize=9)
    counts = np.array([len(r['改进机制核验']['逐次主补测']) for r in runs]); misses = np.array([r['诊断']['补测无信号次数'] for r in runs])
    bars = axes[1].bar(x, counts - misses, color='#229E86', label='收到方向/near')
    axes[1].bar(x, misses, bottom=counts - misses, color='#c87957', label='无信号')
    for i, (n, m) in enumerate(zip(counts, misses)): axes[1].text(i, n + .4, f'{n}次\n无信号{m}次', ha='center', fontsize=9)
    axes[1].set(title='主补测65次，其中9次无信号', ylabel='检测次数', ylim=(0, 22)); axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [r['编号'] for r in runs]); ax.grid(axis='y', alpha=.15)
    fig.tight_layout(); save(fig, '02_耗时与主补测反馈')
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
    for ax, run, diagnostic in zip(axes, runs[2:4], data['被延后时可行候选核查']):
        draw_actual(ax, run)
        current = diagnostic['当前位置']; next_station = diagnostic['下一固定站']
        p = diagnostic['先筛路线约束再评分的候选']['位置']
        ax.plot([current[0], p[0], next_station[0]], [current[1], p[1], next_station[1]], ls='--', color='#111111', lw=1.4)
        ax.scatter(*p, marker='*', s=110, c='#eab83a', edgecolors='#222222', zorder=10)
        ax.annotate('离线可行候选\n实际未执行', p, xytext=(22, 12), textcoords='offset points', fontsize=9,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        old = diagnostic['重现原被拒候选']['补测绕行_米']; new = diagnostic['先筛路线约束再评分的候选']['补测绕行_米']
        ax.set_title(f"{run['编号']} 频道{diagnostic['频道']}：原选点被{800}米门槛拒绝\n原候选绕行{old:.0f}米；可行替代{new:.0f}米", fontsize=12)
    fig.text(.5, .04, '彩色实线为真实轨迹；黑虚线/金星仅为当时信息下重算的候选，不是已执行路线或收益保证。\n第3、4局实际尾程分别3.784、3.797公里。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, 1)); save(fig, '03_长尾程与可行候选诊断')
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for ax, label in zip(axes, ('F4-2', 'F4-6')):
        example = data['600米间隔的局限反例'][label]['例子']; assert example
        source = example['构造源位置']; angle = example['构造发射轴_度']
        ax.add_patch(Circle((0, 0), 1800, fill=False, ls='--', color='#836096'))
        ax.add_patch(Wedge(source, 1000, angle - 90, angle + 90, color='#77c3ad', alpha=.25))
        sites = np.array([d['测量站'] for d in example['全部实际该频道测量']])
        ax.scatter(sites[:, 0], sites[:, 1], s=24, color='#438CB0', label='该频道真实无信号测量站')
        ax.scatter(*source, marker='D', c='#229E86', s=55, label='假设源位置（非真值）')
        ax.scatter(*example['停点'], marker='*', s=140, c='#e8a739', edgecolors='#555555', label='真实到达但未扫描此频道的点')
        near = example['最近该频道旧测量站']; stop = example['停点']
        ax.plot([near[0], stop[0]], [near[1], stop[1]], color='#cf5b5b', lw=2)
        ax.annotate(f"仅相距{example['距离_米']:.0f}米", stop, xytext=(-85, 32), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#cf5b5b'), fontsize=9)
        ax.set(aspect='equal', xlim=(-2000, 2400), ylim=(-2000, 2000), xlabel='东向 x（米）', ylabel='北向 y（米）',
               title=f'{label} 信息不重复的构造反例\n绿色为假设源1000米前向接收半圆')
        ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, bbox_to_anchor=(.5, .045), fontsize=10)
    fig.text(.5, .013, '两图均为与全部无信号记录相容的假设，用于评估600米规则；不是实际遗漏源定位结果。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.86, wspace=.2)
    save(fig, '04_600米距离规则的局限')


def documents(runs, data):
    a, b = data['上批汇总'], data['本批汇总']
    time_change = 100 * (b['平均虚拟时间_秒'] / a['平均虚拟时间_秒'] - 1)
    distance_change = 100 * (b['平均路程_米'] / a['平均路程_米'] - 1)
    table = '| 编号 | 案例码 | 全向/定向 | 清除/目标 | 路程/公里 | 时间/秒 | 主补测/无信号 | 尾程/公里 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    per_case = []
    for r in runs:
        d, o = r['诊断'], r['官方结果']; tail = d['阶段及移动用途']['扫描后收尾']['路程_米']
        table += f"| [{r['编号']}](../../results/figures/第四问/实测方向反馈_新建文件夹2/{r['编号']}_轨迹.png) | {o['case_code']} | {o['omnidirectional_jammer_count']}/{o['directional_jammer_count']} | {d['清除数']}/{d['目标数']} | {d['路程_米']/1000:.3f} | {d['时间_秒']:.1f} | {d['检测用途次数']['追加测向']}/{d['补测无信号次数']} | {tail/1000:.3f} |\n"
        x = r['改进机制核验']
        per_case.append(f"| {r['编号']} | {len(x['小范围顺路任务'])} | {x['方向反馈选点数']}/{x['方向反馈选点无信号数']} | {x['补齐检测数']} | {len(x['补齐首次发现频道'])} | {len(x['已知目标可证明超距的附带检测'])} |")
    comparison = '| 指标 | 上批100米/400米 | 本批反馈/800米 |\n|---|---:|---:|\n'
    for title, key, scale, digits in [('清除数量', '清除总数', 1, 0), ('全清场次', '全清局数', 1, 0),
                                     ('平均时间/秒', '平均虚拟时间_秒', 1, 1), ('平均路程/公里', '平均路程_米', 1000, 3),
                                     ('平均尾程/公里', '平均尾程_米', 1000, 3), ('主补测次数', '主补测次数', 1, 0),
                                     ('主补测无信号', '主补测无信号次数', 1, 0), ('清除未命中', '清除失败次数', 1, 0),
                                     ('扫描后清除目标数', '扫描后清除数', 1, 0), ('顺路首次发现目标数', '顺路首次发现数', 1, 0)]:
        comparison += f"| {title} | {a[key]/scale:.{digits}f} | {b[key]/scale:.{digits}f} |\n"
    candidates = '| 案例/频道 | 被拒候选绕行/米 | 合格候选数 | 次优可行绕行/米 | 接收支持度 |\n|---|---:|---:|---:|---:|\n'
    for d in data['被延后时可行候选核查']:
        p = d['先筛路线约束再评分的候选']
        candidates += f"| {d['编号']}/频道{d['频道']} | {d['重现原被拒候选']['补测绕行_米']:.1f} | {d['可行候选数']} | {p['补测绕行_米']:.1f} | {p['接收支持度']:.3f} |\n"
    text = f'''# 方向反馈改进版六局实际演练：结果、轨迹与后续优化

本批六局清除75/77个源，4局全清；F4-2、F4-6各漏1源。平均时间{b['平均虚拟时间_秒']:.1f}秒，平均路程{b['平均路程_米']/1000:.3f}公里。已发现目标全部清除，发现缺口仍未解决；不存在“大量已定位目标未处理”的现象。

## 1. 来源、版本与真实轨迹

输入为“新建文件夹 (2)”的18份文件（六组jlog、psum、result.json），按官方结束时间编号F4-1至F4-6，与旧批N4及初版P4编号区分。六局均为第四问演练，记录提交`6dddfd5`、工作区干净；当前配置800米绕行、100米优先补测间距、方向反馈、小范围覆盖和未知频道补全均开启。已核对包SHA256、两个公开头和结果字段，并唯一关联本地HTTP记录逐动作复算。正文未解密、包签名未验证；原始材料未修改，没有新增模拟器调用。

{table}

F4-1在12个固定站后已清除16个不同频道，达到题目目标数上界，合法提前结束；图中站13为空心灰方块。其余五局访问13站。F4-2、F4-6均正常完成程序流程，但实际未全清。未知频道列表同时包含不存在的空频道，不能把列表长度当剩余源数，也不能从现有日志识别究竟哪个未知频道或坐标对应漏源。

![六局实际轨迹总览](../../results/figures/第四问/实测方向反馈_新建文件夹2/01_六局轨迹总览.png)

表格编号链接到六张单局大图，均标注1800米边界、真实行进方向、固定站、补测点、清除频道、起终点及扫描后紫色尾程。圆点是机器人成功清除位置，真实源只能确定在其20米内，不能当成精确源坐标。

## 2. 哪些问题已经减轻

{comparison}

两批案例码全部不同，虽然同为77个源、定向源数量也接近（上批43、本批42），空间位置、发射朝向及目标数量分布仍不同。平均时间变化{time_change:.2f}%、路程变化{distance_change:.2f}%是两批描述性差值，不是同地图因果收益；全清数量仍为4/6，没有证据称本批发现能力已完全解决。

本批65次主补测有9次无信号，比例13.85%；上批45/93，48.39%。方向反馈实际启用32次，其中31次得到有效方向、1次无信号；这是本批选择过的检测结果，不能把31/32解释为候选支持度的概率校准。小范围顺路覆盖处理12个目标，全批仅4次清除未命中，未触发末尾大面积有限网格清除。

未知频道补全增加230次检测与1380秒检测/切换费用（平均每局230秒），其中18个源在这些新增检测中首次发现；全批顺路首次发现22个源。这说明附带扫描很有价值，但“首次在此发现”不等于关闭机制后必然少清除18源，其他后续站是否发现需要同地图对照。不能为降低时间直接关闭未知频道补扫。

| 编号 | 小范围清除目标 | 方向反馈次数/无信号 | 补齐未知检测 | 补齐首次发现 | 已知超距附带检测 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_case)}

![时间与反馈](../../results/figures/第四问/实测方向反馈_新建文件夹2/02_耗时与主补测反馈.png)

## 3. 优先优化：在可行补测点中选最优

F4-3和F4-4的长尾程分别为3.784公里/767.8秒、3.797公里/775.4秒，各只为一个早已发现的目标收尾。F4-3频道10在第7站首次发现，补测绕行989.8米被800米门槛拒绝；末尾从东南向西北返回，第一段就走3495.1米。F4-4频道8也在第7站发现，第8站后一次补测无信号，之后候选绕行1086.4米被拒，末尾第一段返回3347.9米。

代码当前先选综合评分最好的补测点，再在执行层检查绕行门槛；被拒后直接延后整个频道，没有在剩余合格点中重新选择。仅用当时已有观测、当前位置、剩余固定站离线重建候选池，可精确复现上述两次被拒绕行，并找到仍符合原两项路线规则的替代点：

{candidates}

建议先把800米补测约束及完整任务延后条件用于候选筛选，再在合格集合中评分。若合格集合为空，再比较是否延后或有限放宽门槛。这样优先修正选点与路线约束的衔接，避免只把800米统一加大。

替代点没有实际测量，支持度是启发式权重，并非接收概率或保证。特别是F4-4替代点对整个外包的最远距离约1292米，无法保证接收；不能承诺因此省下全部约770秒尾程。F4-3原被拒点的完整任务预测1492米、延后3247米，也提示应在无合格点时审查门槛与延后费用，而非一律把目标留到最后。

![尾程与候选核查](../../results/figures/第四问/实测方向反馈_新建文件夹2/03_长尾程与可行候选诊断.png)

## 4. 同时优化：未知频道不能只按600米判定重复

F4-2和F4-6末尾分别有11、5个未知频道，每个都在13个固定站测过，实际只分别残留1源。由于缺少漏源真值，无法断言这两个源的具体位置、频道或发射轴；现有全部观测只证明其从未给出有效信号，不能判定是站点空间缺口还是顺路停点未扫描造成。

现有补全修复了4频道数量上限，但仍保留距该频道所有旧测量站至少600米的规则。对定向源，两个相距很近的点也可能分居前后半平面；“距离近”不能证明测量信息重复。已在两局真实停点找到构造反例：F4-2有未扫描停点距旧测量站仅250米；F4-6类似距离262.9米。为各自未知频道构造一个圆内位置、1000米半径与朝向，可同时满足所有真实无信号记录，却在该停点能够接收。位置与朝向是验证规则局限的假设，绝非实际漏源定位。

建议把600米改为常规稀疏扫描条件，并允许“此停点可区分仍相容的位置/朝向假设”的例外；只复用已到达停点，设置额外检测费用预算。已知目标朝向约束和未知频道假设应分开维护，未知频道没有第一条方向，不能直接套用已发现目标的反馈定位。该例外的收益仍需本地同图对照，不能据构造反例承诺补齐这两源。

若需要任意朝向发现保证，仅优化现有停点扫描仍不充分，13站及实际停点可能整体不经过某接收区。是否最终增加少量发现站应作为独立方案再评估；本轮未启用外围搜索或修改布局。

![600米规则反例](../../results/figures/第四问/实测方向反馈_新建文件夹2/04_600米距离规则的局限.png)

## 5. 次优先级：跳过可严格证明超距的已知目标检测

根据每次测量之前的连续正观测外包，发现{len(data['可证明超距的已知频道附带检测'])}次附带检测对已知目标的最短可能距离已超过1500米，即便最大检测半径也不可能接收。这些检测的纯检测费用合计{data['纯检测费用_秒']}秒，平均每局{data['纯检测费用_秒']/6:.1f}秒；切换顺序变化及整体路线需要重算，不能简单加上全部原切换费用。

建议先核验严格距离下界再跳过，仅用于已经发现、外包有效的频道。未知频道不能以此排除。此项节省幅度小于长尾程问题，但判断依据明确，不需要放宽检测或清除安全条件。

## 6. 建议验证顺序与复现

优先比较“可行候选内选点”，再单独加入“未知频道信息增量补扫”，最后检查“已知目标确定超距免测”。预先固定地图，记录清除数量、全清、总虚拟时间及分位数、尾程、检测量、主补测无信号；保留更慢与漏源案例，不只追求一次绕圈或更少点数。当前800米版本作为基线，参数扫描与新验证场景分离；当前策略源码和配置未改。

```powershell
Set-Location 'D:\\mywork\\code\\cumcm2026-b'
.\\.venv\\Scripts\\python.exe -X utf8 scripts/analyze_q4_feedback_practice.py --verify --source-check
```

不带参数时从已归档脱敏动作重绘；`--collect`从现有六组原始文件及对应本地HTTP记录重新采集，不调用模拟器。保存6份脱敏逐动作JSON、费用及机制核验、两处候选重算、两个几何反例、10张中文图各PNG/SVG和来源清单。来源核验包括原始包、关联记录、运行代码与图文依赖。
'''
    REPORT.write_text(text, encoding='utf8', newline='\n')
    PAPER.write_text(f'''# 方向反馈改进版六局实际演练：论文备用说明

六局第四问演练均运行6dddfd5版本，800米绕行门槛和100米优先补测间距。合计清除75/77个源，4局全清，第2、6局各漏1源；平均时间5008.7秒、路程17.177公里。第1局清除16个源后在12站结束；其余访问13站。所有已发现目标清除完成，两个遗漏发生于发现阶段，不能把正常流程结束当成全清。

主补测65次，其中9次无信号；方向反馈启用32次，31次得到有效方向。小范围联合覆盖顺路清除12源，仅4次未命中清除，未发生末尾大范围网格搜索。补齐未知频道实际增加230次检测、1380秒切换及测量费用，18个源首次通过该部分发现；这不是关闭机制后必然少发现18源的因果估计。

与前批不同地图的平均时间相比下降9.80%、路程下降13.68%，但全清同为4/6，两个批次均75/77，不能据此宣称总体发现能力提升或因果时间收益。第3、4局各为一个已知源产生约3.79公里尾程。重建当时信息发现，先评分再检查800米门槛导致合格次优候选未被利用；另以与负反馈相容的假设证明600米距离规则不能确保定向测量信息重复。已知目标存在{len(data['可证明超距的已知频道附带检测'])}次可严格证明超距的附带检测，纯检测费用{data['纯检测费用_秒']}秒。

下一步应按候选可行性、未知频道信息增量和已知目标超距免测分别消融验证。替代候选未实际测量，构造源不是真实漏源坐标；13站完整发现仍无保证。原始资料及当前策略未改，本轮只分析已有演练，没有新增模拟器调用。

[完整分析、流程建议与全部轨迹](../../docs/第四问/实测方向反馈_新建文件夹2分析.md)。
''', encoding='utf8', newline='\n')


def manifest(runs):
    files = list(MODELS.glob('*.json')) + list(FIG.glob('*')) + [TABLE, REPORT, PAPER, Path(__file__)]
    files += [ROOT / 'scripts' / name for name in ['analyze_q4_spacing_practice.py', 'analyze_q4_practice_4.py',
              'analyze_q3_practice_log.py', 'evaluate_q3_routed_practice.py', 'build_q4_inner13_assets.py', 'q4_archive_sources.py']]
    files += [ROOT / f'results/models/第四问/实测100米间距/N4-{i}.json' for i in range(1, 7)]
    files += [ROOT / name for name in runs[0]['结果']['运行来源']['源码_SHA256']]
    return {relative(p): digest(p) for p in sorted(set(files))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect', action='store_true'); p.add_argument('--verify', action='store_true')
    p.add_argument('--source-check', action='store_true'); args = p.parse_args()
    if args.collect and args.verify: p.error('--collect和--verify不同时使用')
    runs = collect() if args.collect else [read(MODELS / f'F4-{i}.json') for i in range(1, 7)]
    for r in runs:
        assert r['诊断'] == analyze(r) and r['改进机制核验'] == extension_audit(r)
        if args.source_check:
            for path, value in r['来源SHA256'].items(): assert digest(ROOT / path) == value, path
            for path, value in r['结果']['运行来源']['源码_SHA256'].items(): assert matches_source(ROOT, path, value), path
    data = evaluation(runs)
    if args.verify:
        assert read(TABLE) == data
        expected = read(MANIFEST)
        assert expected.keys() == manifest(runs).keys()
        verify_manifest(ROOT, expected)
        assert len(list(FIG.glob('*.png'))) == len(list(FIG.glob('*.svg'))) == 10
        for path in FIG.glob('*'):
            if path.suffix == '.png':
                with Image.open(path) as image: image.verify()
            elif path.suffix == '.svg': ET.parse(path)
        print('核验通过：六局费用、协议来源、提前结束、改进机制、历史信息候选、反例及20份中文图。')
    else:
        write(TABLE, data); figures(runs, data); documents(runs, data); write(MANIFEST, manifest(runs))
    print(json.dumps(data['本批汇总'], ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
