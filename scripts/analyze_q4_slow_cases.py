#!/usr/bin/env python
"""复核第四问第4—6局慢因与第5局两点清除的局部插入费用，不运行策略或模拟器。"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cumcm2026_b.q3_geometry import minimum_circle, sweep_path

MODELS = ROOT / 'results/models/第四问/实测100米间距'
OUTPUT = ROOT / 'results/tables/第四问/实测100米间距_慢局优化依据.json'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def analysis():
    files = [MODELS / f'N4-{i}.json' for i in (4, 5, 6)]
    runs = [read(p) for p in files]
    summaries = []
    for r in runs:
        d = r['诊断']
        x = r['间距与覆盖诊断']
        events = r['结果']['动作记录']
        active_misses = [e for e in events if e.get('用途') == '追加测向'
                         and e.get('结果') == 'no_signal']
        limits = []
        for e in active_misses:
            history = [v for v in events if v['步骤'] < e['步骤']
                       and v.get('频道') == e['频道'] and '外包顶点' in v]
            vertices = np.asarray(history[-1]['外包顶点'])
            farthest = float(np.linalg.norm(vertices - np.asarray(e['位置']), axis=1).max())
            assert farthest <= 999.5 + 1e-6
            limits.append({'步骤': e['步骤'], '频道': e['频道'],
                           '当时全部可能位置的距离上界_米': farthest})
        summaries.append({
            '编号': r['编号'], '目标数': d['目标数'], '清除数': d['清除数'],
            '虚拟时间_秒': d['时间_秒'], '总路程_米': d['路程_米'],
            '时间分项_秒': d['时间分项_秒'],
            '扫描后收尾': d['阶段及移动用途']['扫描后收尾'],
            '尾程逐频道': d['尾程逐频道'],
            '主补测数': d['检测用途次数']['追加测向'],
            '主补测无信号数': d['补测无信号次数'],
            '无信号在当时即可排除超距': limits,
            '有限网格清除': [{k: v for k, v in g.items()
                             if k not in ('计划位置', '实际位置')}
                            for g in x['有限覆盖核验']],
            '阈值延后': [e for e in events if e['动作'] == '延后目标'
                         and '本次测向绕行_米' in e],
        })
    r = runs[1]
    events = r['结果']['动作记录']
    obs = next(e for e in events if e['步骤'] == 156)
    assert obs['频道'] == 15 and obs['结果'] == 'direction'
    current = np.asarray(obs['位置'])
    vertices = np.asarray(obs['外包顶点'])
    radius = minimum_circle(vertices)[1]
    assert abs(radius - 20.540405635898445) < 1e-7
    next_station = next(e for e in events if e['步骤'] > obs['步骤']
                        and e.get('用途', '').startswith('固定搜索站'))
    destination = np.asarray(next_station['位置'])
    path = sweep_path(vertices, current)
    assert len(path) == 2
    points = [current, *path, destination]
    visit = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    direct = math.dist(current, destination)
    detour = visit - direct
    # 两个20米网格点覆盖整个当时外包；源存在，最迟第二次成功。
    # 不使用最终成功位置选择顺序；最多一次失败3秒及一次成功5秒。
    clear_cost = 3 * (len(path) - 1) + 5
    late = next(g for g in r['间距与覆盖诊断']['有限覆盖核验'] if g['频道'] == 15)
    local = {
        '场次': 'N4-5', '频道': 15, '决策步骤': 156,
        '当时外包覆盖圆半径_米': radius, '当前位置': current.tolist(),
        '下一固定站': next_station['用途'], '下一固定站坐标': destination.tolist(),
        '按当时外包生成的清除点': path.tolist(), '清除点数': len(path),
        '直接到下一站_米': direct, '插入两点后到下一站_米': visit,
        '最多增加绕行_米': detour, '最多清除动作时间_秒': clear_cost,
        '局部增加时间上界_秒': detour / 5 + clear_cost,
        '原日志最后处理该频道时间_秒': late['总时间_秒'],
        '解释': '只比较局部几何与清除费用；不含可选顺路扫描，不预测全局路线和整局收益。没有重跑策略或调用模拟器。',
    }
    events6 = runs[2]['结果']['动作记录']
    obs6 = next(e for e in events6 if e['步骤'] == 78)
    assert obs6['频道'] == 11
    start6 = np.asarray(obs6['位置'])
    next6 = next(e for e in events6 if e['步骤'] > 78
                 and e.get('用途', '').startswith('固定搜索站')
                 and e['位置'] != start6.tolist())
    end6 = np.asarray(next6['位置'])
    grid6 = sweep_path(np.asarray(obs6['外包顶点']), start6)
    sequence6 = [start6, *grid6, end6]
    detour6 = sum(math.dist(a, b) for a, b in zip(sequence6, sequence6[1:])) - math.dist(start6, end6)
    comparison = {'场次': 'N4-6', '频道': 11, '观测步骤': 78,
                  '当前位置': start6.tolist(), '下一站': next6['用途'],
                  '清除点数': len(grid6), '清除点位置': grid6.tolist(),
                  '完整插入额外路程_米': detour6,
                  '局部增加时间上界_秒': detour6 / 5 + 3 * (len(grid6) - 1) + 5,
                  '解释': '虽只需少量格点，但以第3站为起点返回下一站的绕行仍很大；不能只按格点数量强制提前处理。'}
    dependencies = files + [Path(__file__), ROOT / 'src/cumcm2026_b/q3_geometry.py',
                           ROOT / 'src/cumcm2026_b/q4_strategy.py',
                           ROOT / 'src/cumcm2026_b/q4_spacing_strategy.py',
                           ROOT / 'src/cumcm2026_b/q3_route_planning.py']
    return {'性质': '既有日志慢因核验与待实施建议的局部计算，不是新版试验结果',
            '逐局依据': summaries, '第五局小范围清除插入': local,
            '第六局小范围也可能不顺路': comparison,
            '来源SHA256': {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                           for p in dependencies}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = analysis()
    if args.verify:
        assert read(OUTPUT) == result
        print('核验通过：第4—6局在线距离上界及第5局两点清除局部费用。')
    else:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf8', newline='\n')
    print(json.dumps(result['第五局小范围清除插入'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
