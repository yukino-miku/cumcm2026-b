#!/usr/bin/env python
"""核验用户提供的s11/s22新版演练，保存脱敏统计与完整实际行动坐标。"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from analyze_q3_practice_log import public_header, matched_run, audit_actions
from cumcm2026_b.q3_geometry import search_stations
import numpy as np
from scipy.stats import t

BASE = ROOT / 'materials/original/simulator/CUMCM2026B/Jammers-simulator-win64/Jammers-simulator/JammersSimulatorData/behavior-logs'
REPORT = ROOT / 'results/tables/第三问/新版演练评估_s11_s22.json'
TRACKS = ROOT / 'results/tables/第三问/新版十二局实际轨迹_s11_s22.json'
OLD = ROOT / 'results/tables/第三问/两组演练评估_s1_s2.json'
GROUPS = {'s11': '方案一加路线规划', 's22': '方案二路线规划与停止重复扫描'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def audit_trace(items, result):
    """逐个核对明文请求/反馈与用于绘图的程序动作，不能只比较总路程。"""
    requests = {x['payload']['request_id']: x for x in items if x.get('类型') == '请求'}
    responses = {x['request_id']: x['response'] for x in items if x.get('类型') == '响应'}
    moves = [x for x in requests.values() if x['path'] in ('/measure', '/clear')]
    actions = [e for e in result['动作记录'] if e['动作'] in ('检测', '清除')]
    assert len(moves) == len(actions)
    for request, event in zip(moves, actions):
        p = request['payload']; response = responses[p['request_id']]
        assert event['动作'] == ('检测' if request['path'] == '/measure' else '清除')
        assert event['频道'] == p['channel']
        assert math.dist(event['位置'], [p['position']['x'], p['position']['y']]) < 1e-8
        assert event['结果'] == response['measure_result' if event['动作'] == '检测' else 'clear_result']
        assert abs(event['虚拟时间_秒'] - response['virtual_time_s']) < .001
    return actions


def gather(base):
    rows, tracks, sources = [], [], {}
    for group, name in GROUPS.items():
        files = sorted((base / group).glob('*.result.json'))
        assert len(files) == 6, (group, '应为用户指定的六局')
        stems = {p.name.removesuffix('.result.json') for p in files}
        assert {p.stem for p in (base / group).glob('*.jlog')} == stems
        assert {p.stem for p in (base / group).glob('*.psum')} == stems
        for path in files:
            official = json.loads(path.read_text(encoding='utf-8-sig'))
            stem = path.name.removesuffix('.result.json')
            jlog, psum = path.with_name(stem + '.jlog'), path.with_name(stem + '.psum')
            header, _ = public_header(jlog)
            raw = psum.read_bytes()
            assert raw[:8] == b'JMBPSUM1' and int.from_bytes(raw[8:10], 'big') == 1
            length = int.from_bytes(raw[10:14], 'big')
            assert 0 < length <= len(raw) - 14
            summary = json.loads(raw[14:14 + length])
            assert official['package_sha256'] == digest(jlog)
            assert header['package_type'] == 'practice_behavior_log' and summary['package_type'] == 'practice_summary'
            assert header['formal_index'] is None and summary['formal_index'] is None
            for key in ['problem_no', 'case_code', 'practice_run_no']:
                assert official[key] == header[key] == summary[key]
            assert header['problem_no'] == 3 and header['team_no'] == summary['team_no']
            assert official['jammer_count'] == official['omnidirectional_jammer_count']
            assert official['directional_jammer_count'] == 0 and 10 <= official['jammer_count'] <= 16
            folder, items, delta = matched_run(header)
            result_path = folder / '运行结果.json'
            result = json.loads(result_path.read_text(encoding='utf-8'))
            expected = json.loads((ROOT / f'configs/q3_scheme{1 if group == "s11" else 2}_routed.json').read_text(encoding='utf-8'))
            assert result['配置'] == expected, (group, '运行配置不同于本次默认改进版')
            assert result['方案'].startswith('方案一：' if group == 's11' else '方案二：')
            assert result['启用路线规划'] is True
            if group == 's22': assert result['停止已定位频道扫描'] is True
            audit = audit_actions(items, result)
            actions = audit_trace(items, result)
            success = [e for e in actions if e['动作'] == '清除' and e['结果'] == 'success']
            failed = [e for e in actions if e['动作'] == '清除' and e['结果'] != 'success']
            assert len({e['频道'] for e in success}) == result['清除数']
            assert all(not e['具有覆盖证书'] for e in failed)
            full = bool(result['运行成功'] and result['正常退出'] and result['清除数'] == official['jammer_count'])
            fixed = search_stations() if group == 's11' else np.asarray(result['布局']['站点'])
            fixed_events = [e for e in actions if e['动作'] == '检测' and e['用途'].startswith(('搜索站', '固定扫描站'))]
            visits = [i for i, p in enumerate(fixed) if any(math.dist(p, e['位置']) < 1e-6 for e in fixed_events)]
            assert visits == result['已完成搜索站']
            cutoff = max(e['步骤'] for e in fixed_events)
            portions = {'末次固定检测及之前': 0., '末次固定检测之后': 0.}
            purposes = {'前往固定站': 0., '前往追加测向点': 0., '前往清除点': 0.}
            previous = np.zeros(2)
            for e in actions:
                distance = math.dist(previous, e['位置']); previous = np.asarray(e['位置'])
                portions['末次固定检测之后' if e['步骤'] > cutoff else '末次固定检测及之前'] += distance
                key = '前往清除点' if e['动作'] == '清除' else '前往追加测向点' if e['用途'] == '追加测向' else '前往固定站'
                if key == '前往固定站' and distance > 1e-7: assert e['用途'].startswith(('搜索站', '固定扫描站'))
                purposes[key] += distance
            assert abs(sum(portions.values()) - result['总路程_米']) < 1e-5
            route = {'重规划次数': sum(e['动作'] == '路线规划' for e in result['动作记录']),
                     '延后决策次数': sum(e['动作'] == '延后目标' for e in result['动作记录']),
                     '顺路证书清除次数': sum(e.get('用途') == '顺路证书清除' for e in actions)}
            assert route == result['路线调度统计']
            skips = [e for e in result['动作记录'] if e['动作'] == '跳过固定检测']
            assert len(skips) == result.get('跳过已定位频道检测次数', 0)
            certificates = {e['频道']: e for e in result['动作记录'] if e['动作'] == '定位完成'}
            for e in skips:
                c = certificates[e['频道']]
                assert e['证书步骤'] == c['步骤'] < e['步骤'] and c['不同方向站数'] >= 2
                assert np.linalg.norm(np.array(c['外包顶点']) - c['覆盖圆心'], axis=1).max() <= 19.5
            row = {'组': group, '方案': name, '案例码': official['case_code'], '演练编号': str(official['practice_run_no']),
                   '结束时间UTC': official['ended_at_utc'], '客户端版本': header['client_version'],
                   '官方结果文件目标总数': official['jammer_count'], '成功清除数': result['清除数'],
                   '全清核对通过': full, '正常退出': result['正常退出'], '异常': result['异常'],
                   '总虚拟时间_秒': result['虚拟总时间_秒'],
                   '每目标虚拟时间_秒': result['虚拟总时间_秒'] / official['jammer_count'],
                   '总路程_米': result['总路程_米'], '检测次数': result['检测次数'], '切换次数': result['频道切换次数'],
                   '专门追加测向次数': sum(s['追加测向次数'] for s in result['频道记录']),
                   '失败清除次数': result['清除失败次数'], '失败清除用途': [e['用途'] for e in failed],
                   '本地程序时间_秒': result['程序运行时间_秒'], '时间分项_秒': result['时间分项_秒'],
                   '配置': result['配置'], '请求响应核验': audit, '逐动作坐标反馈核对数': len(actions),
                   '关联时间差_毫秒': delta, '关联本地目录': folder.relative_to(ROOT).as_posix(),
                   '官方结果文件': path.relative_to(ROOT).as_posix(), '路线调度统计': route,
                   '定位完成证书数': len(certificates), '跳过固定检测次数': len(skips),
                   '检测用途计数': dict(Counter(e['用途'] if not e['用途'].startswith(('搜索站', '固定扫描站')) else '固定站检测'
                                             for e in actions if e['动作'] == '检测')),
                   '清除用途计数': dict(Counter(e['用途'] for e in success)),
                   '延后原因计数': dict(Counter(e['原因'] for e in result['动作记录'] if e['动作'] == '延后目标')),
                   '已完成固定站': visits, '末次固定检测步骤': cutoff,
                   '末次固定检测前成功清除数': sum(e['步骤'] < cutoff for e in success),
                   '按阶段路程_米': portions, '按终点用途路程_米': purposes,
                   '末次固定检测时间_秒': fixed_events[-1]['虚拟时间_秒'],
                   '失败清除记录': [{k: e[k] for k in ['步骤', '频道', '用途', '位置', '虚拟时间_秒']} for e in failed]}
            assert abs(row['每目标虚拟时间_秒'] - result['平均定位清除时间_秒']) < 1e-8
            rows.append(row)
            tracks.append({'案例码': row['案例码'], '组': group, '目标数': official['jammer_count'], '全清': full,
                           '总虚拟时间_秒': result['虚拟总时间_秒'], '总路程_米': result['总路程_米'],
                           '检测次数': result['检测次数'], '清除成功次数': result['清除成功次数'],
                           '清除失败次数': result['清除失败次数'], '固定站点': fixed.tolist(),
                           '已完成固定扫描站序号': visits, '终止依据': result['终止依据'],
                           '末次固定检测步骤': cutoff, '路线调度统计': route,
                           '实际动作': [{k: e[k] for k in ['步骤', '虚拟时间_秒', '动作', '用途', '频道', '位置', '结果']}
                                        for e in actions]})
            for p in [path, jlog, psum, result_path, folder / '请求响应日志.jsonl']:
                sources[p.relative_to(ROOT).as_posix()] = digest(p)
    rows.sort(key=lambda r: (r['组'], r['结束时间UTC']))
    assert len({r['关联本地目录'] for r in rows}) == len(rows)
    for group in GROUPS:
        subset = [r for r in rows if r['组'] == group]
        assert len({json.dumps(r['配置'], sort_keys=True) for r in subset}) == 1
        for i, row in enumerate(subset, 1): row['显示编号'] = f'{group.upper()}-{i}'
    indices = {r['案例码']: r['显示编号'] for r in rows}
    for track in tracks: track['编号'] = indices[track['案例码']]
    tracks.sort(key=lambda r: r['编号'])
    return rows, tracks, sources


def statistics(rows):
    output = {}
    numeric = ['官方结果文件目标总数', '总虚拟时间_秒', '每目标虚拟时间_秒', '总路程_米', '检测次数',
               '切换次数', '专门追加测向次数', '本地程序时间_秒', '末次固定检测前成功清除数', '跳过固定检测次数']
    for group in GROUPS:
        rs = [r for r in rows if r['组'] == group]
        output[group] = {'演练数': len(rs), '全清次数': sum(r['全清核对通过'] for r in rs),
                        **{f'平均{k}': float(np.mean([r[k] for r in rs])) for k in numeric},
                        '清除目标总数': sum(r['成功清除数'] for r in rs),
                        '失败清除总次数': sum(r['失败清除次数'] for r in rs),
                        '最大总虚拟时间_秒': max(r['总虚拟时间_秒'] for r in rs),
                        '每目标时间中位数_秒': float(np.median([r['每目标虚拟时间_秒'] for r in rs])),
                        '每目标时间样本标准差_秒': float(np.std([r['每目标虚拟时间_秒'] for r in rs], ddof=1)),
                        '合计时间除以合计目标_秒': sum(r['总虚拟时间_秒'] for r in rs) / sum(r['官方结果文件目标总数'] for r in rs),
                        '目标数分布': dict(sorted(Counter(str(r['官方结果文件目标总数']) for r in rs).items())),
                        '平均时间分项_秒': {k: float(np.mean([r['时间分项_秒'][k] for r in rs])) for k in rs[0]['时间分项_秒']},
                        '路线调度合计': {k: sum(r['路线调度统计'][k] for r in rs) for k in rs[0]['路线调度统计']},
                        '平均按阶段路程_米': {k: float(np.mean([r['按阶段路程_米'][k] for r in rs])) for k in rs[0]['按阶段路程_米']},
                        '平均按终点用途路程_米': {k: float(np.mean([r['按终点用途路程_米'][k] for r in rs])) for k in rs[0]['按终点用途路程_米']}}
    return output


def comparison(first, second):
    x, y = [np.asarray([r['每目标虚拟时间_秒'] for r in rows]) for rows in [first, second]]
    common = sorted({r['官方结果文件目标总数'] for r in first} & {r['官方结果文件目标总数'] for r in second})
    strata = []
    for count in common:
        row = {'目标数': count}
        for name, rows in [('第一组', first), ('第二组', second)]:
            sub = [r for r in rows if r['官方结果文件目标总数'] == count]
            row[name] = {'样本数': len(sub), '平均每目标时间_秒': float(np.mean([r['每目标虚拟时间_秒'] for r in sub]))}
        row['第一组相对第二组降低百分比'] = (1 - row['第一组']['平均每目标时间_秒'] / row['第二组']['平均每目标时间_秒']) * 100
        strata.append(row)
    equal = {name: float(np.mean([r[name]['平均每目标时间_秒'] for r in strata])) for name in ['第一组', '第二组']} if strata else None
    a, b = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (a + b)**2 / (a*a/(len(x)-1) + b*b/(len(y)-1))
    interval = (x.mean() - y.mean() + np.array([-1, 1]) * t.ppf(.975, df) * np.sqrt(a+b)).tolist()
    return {'相同案例码数': len({r['案例码'] for r in first} & {r['案例码'] for r in second}),
            '平均总时间降低百分比': (1 - np.mean([r['总虚拟时间_秒'] for r in first]) / np.mean([r['总虚拟时间_秒'] for r in second])) * 100,
            '每局每目标均值降低百分比': float((1 - x.mean() / y.mean()) * 100),
            '共同目标数分层': strata, '共同层等权均值_秒': equal,
            '共同层等权降低百分比': (1 - equal['第一组'] / equal['第二组']) * 100 if equal else None,
            '第一组减第二组T除N均值差_秒': float(x.mean()-y.mean()), '探索性Welch95区间_秒': interval,
            '限制': '各组6局且未确认随机分配；共同目标数不代表相同位置难度。区间仅为近似独立样本描述，不是纯算法因果效应区间。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    rows, tracks, sources = gather(args.base.resolve())
    old = json.loads(OLD.read_text(encoding='utf-8'))
    assert all(digest(ROOT / p) == h for p, h in old['来源SHA256'].items()), '原版对照来源变化'
    sources[OLD.relative_to(ROOT).as_posix()] = digest(OLD)
    for p in [Path(__file__), ROOT/'scripts/analyze_q3_practice_log.py', ROOT/'src/cumcm2026_b/q3_geometry.py',
              ROOT/'configs/q3_scheme1_routed.json', ROOT/'configs/q3_scheme2_routed.json']:
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
    track_data = {'说明': 's11/s22各6局实际HTTP动作坐标；逐动作与原始请求响应核对。无目标真值、队号、票据。',
                  '连线口径': '从原点依实际检测/清除执行顺序连直线，不添加回原点路线；清除位置为机器狗位置。',
                  '来源SHA256': sources, '逐局轨迹': tracks}
    write_json(TRACKS, track_data)
    history = {}
    for new_group, old_group in [('s11', 's1'), ('s22', 's2')]:
        history[new_group] = {'第一组': new_group, '第二组': old_group,
                             **comparison([r for r in rows if r['组'] == new_group], [r for r in old['逐局记录'] if r['组'] == old_group])}
    report = {'说明': '新版s11/s22实际演练；未解密jlog/psum正文或验证数字签名。以结果JSON核对目标数，以唯一时间关联HTTP记录重算行为。',
              '版本识别': '运行配置与当前默认JSON完全一致，并记录路线决策与跳过扫描；原运行未记录源码commit，不能据此反推逐字节源码版本。',
              '统计': statistics(rows), '两方案比较': {'第一组': 's11', '第二组': 's22',
                  **comparison([r for r in rows if r['组'] == 's11'], [r for r in rows if r['组'] == 's22'])},
              '历史对照': history, '原版历史统计': old['统计'], '逐局记录': rows,
              '核验合计': {'测试文件数': len(rows)*3, '不同请求响应动作数': sum(r['请求响应核验']['不同动作数'] for r in rows),
                           '实际检测清除坐标核对数': sum(r['逐动作坐标反馈核对数'] for r in rows),
                           '重试次数': sum(r['请求响应核验']['重试请求数'] for r in rows),
                           '通信异常数': sum(r['请求响应核验']['通信异常数'] for r in rows)},
              '轨迹数据': TRACKS.relative_to(ROOT).as_posix(), '轨迹数据SHA256': digest(TRACKS), '来源SHA256': sources}
    write_json(REPORT, report)
    print(json.dumps({k: report[k] for k in ['统计', '两方案比较', '历史对照', '核验合计']}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
