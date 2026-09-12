"""第四问反馈改进的预设消融与门槛对照；16场选参、8场独立验证，仅本地构造。"""
import argparse
from dataclasses import asdict
from hashlib import sha256

import numpy as np

from evaluate_q4_zigzag import ROOT, PROFILES, read, dump, aggregate, summary
from evaluate_q4_spacing import sources as spacing_sources
from cumcm2026_b.q4_spacing_strategy import SpacingConfig, SpacedFour
from cumcm2026_b.q4_feedback_strategy import FeedbackConfig, FeedbackFour
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_geometry import min_distance
from build_q4_inner13_assets import verify_run

RUNS = ROOT / 'results/models/第四问/方向反馈与小范围覆盖对照'
TABLE = ROOT / 'results/tables/第四问/方向反馈与小范围覆盖对照.json'
# 在本轮任何实验运行前确定设置、场景种子和排名规则。
VARIANTS = {
    '基线100米': None,
    '仅小范围400': dict(small_sweep_limit=4, direction_feedback=False, task_routing=False),
    '仅方向反馈400': dict(small_sweep_limit=0, direction_feedback=True, task_routing=False),
    '两项组合400': dict(small_sweep_limit=4, direction_feedback=True, task_routing=False),
    '完整任务400': dict(localization_detour_limit_m=400),
    '完整任务600': dict(localization_detour_limit_m=600),
    '完整任务800': dict(localization_detour_limit_m=800),
}


def config(name):
    return SpacingConfig(probe_spacing_m=100) if VARIANTS[name] is None else FeedbackConfig(**VARIANTS[name])


def sources():
    result = spacing_sources()
    for rel in ('src/cumcm2026_b/q4_feedback_strategy.py', 'scripts/evaluate_q4_feedback.py'):
        result[rel] = sha256((ROOT / rel).read_bytes()).hexdigest()
    return result


def spec(stage, i):
    _, count, layout, radius, fraction, orientation = PROFILES[i % 8]
    return dict(seed=(101000 if stage == '选参' else 102000) + i, count=count, layout=layout,
                radius_mode=radius, error_mode='hash', directional_fraction=fraction, orientation=orientation)


def summarize(result, stage, i, name):
    row = summary(result, stage, i, name, config(name).localization_detour_limit_m)
    ev = result['动作记录']
    row['小范围顺路清除任务数'] = sum(e['动作'] == '顺路小范围覆盖规划' for e in ev)
    row['小范围顺路清除目标数'] = sum(e.get('用途') == '顺路小范围覆盖' and e.get('结果') == 'success' for e in ev)
    row['方向反馈选点次数'] = sum(e['动作'] == '规划' and e['方案'].get('方向反馈已启用', False) for e in ev)
    row['方向反馈选点无信号次数'] = sum(
        next(a for a in ev if a['步骤'] > e['步骤'] and a.get('用途') == '追加测向')['结果'] == 'no_signal'
        for e in ev if e['动作'] == '规划' and e['方案'].get('方向反馈已启用', False))
    return row


def verify_extensions(r):
    source_positions = {s['频道']: np.asarray(s['位置']) for s in r['离线真值核验']['目标']}
    for e in r['动作记录']:
        if e['动作'] == '顺路小范围覆盖规划':
            points = np.asarray(e['位置序列'])
            assert len(points) == e['覆盖点数'] <= r['配置']['small_sweep_limit']
            assert np.linalg.norm(points - source_positions[e['频道']], axis=1).min() <= 20
            assert min_distance(e['外包顶点'], source_positions[e['频道']]) < 1e-5
            assert e['完整绕行_米'] <= r['配置']['localization_detour_limit_m'] + 1e-6
        if e['动作'] == '规划' and not e['方案'].get('间距约束已放宽', False):
            p = e['方案']
            assert p['距当前机器人_米'] >= 100 - 1e-6
            assert p['距该频道上次观测_米'] >= 100 - 1e-6
    assert all(s['追加测向次数'] <= 8 for s in r['频道记录'])


def path_for(stage, i, name):
    return RUNS / stage / f'{name}-{i:02d}.json.gz'


def run(stage, i, name):
    path = path_for(stage, i, name)
    if path.exists():
        result = read(path)
        assert result['实验来源SHA256'] == sources() and result['构造参数'] == spec(stage, i)
    else:
        env = make_case(**spec(stage, i))
        cfg = config(name)
        client = RobotClient(env, 'local-robot')
        try:
            result = (SpacedFour if name == '基线100米' else FeedbackFour)(client, cfg).run()
            result.update(离线真值核验=env.truth_for_evaluation(), 构造参数=spec(stage, i),
                          实验来源SHA256=sources(), 实验配置=asdict(cfg),
                          运行类型='固定种子本地配对构造，非官方演练或正式成绩')
            verify_run(result)
            verify_extensions(result)
            dump(path, result)
        finally:
            client.close()
    return summarize(result, stage, i, name)


def choose(rows):
    keys = []
    for name in VARIANTS:
        group = [r for r in rows if r['阶段'] == '选参' and r['方法'] == name]
        assert len(group) == 16
        g = aggregate(group)
        keys.append((g['异常局数'], g['漏源总数'], -g['全清局数'], g['平均虚拟时间_秒'], name))
    return min(keys)[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['选参', '验证', '核验'], required=True)
    args = p.parse_args()
    data = read(TABLE) if TABLE.exists() else {
        '规则': '16场101000—101015选参；8场102000—102007独立验证。先少异常、少漏源、多全清，再比较平均总时间。7种设置；100米间距与600米顺路检测间隔保持，未自动加未知源搜索站。',
        '设置': {k: asdict(config(k)) for k in VARIANTS}, '逐局': []}
    if args.stage == '核验':
        truth = {}
        for row in data['逐局']:
            stage, i, name = row['阶段'], row['场景编号'], row['方法']
            r = read(path_for(stage, i, name))
            assert r['实验来源SHA256'] == sources() and r['构造参数'] == spec(stage, i)
            assert r['配置'] == r['实验配置'] == asdict(config(name))
            verify_run(r)
            verify_extensions(r)
            assert summarize(r, stage, i, name) == row
            target = r['离线真值核验']['目标']
            if (stage, i) in truth:
                assert truth[stage, i] == target
            truth[stage, i] = target
        assert data['选参设置'] == choose(data['逐局'])
        print(f'核验通过：{len(data["逐局"])}次完整动作、物理反馈、清除、费用、同图真值与扩展规则。')
        return
    variants = list(VARIANTS) if args.stage == '选参' else list(dict.fromkeys(['基线100米', choose(data['逐局']), '两项组合400']))
    jobs = [(i, name) for i in range(16 if args.stage == '选参' else 8) for name in variants]
    for n, (i, name) in enumerate(jobs, 1):
        row = run(args.stage, i, name)
        data['逐局'] = [r for r in data['逐局'] if (r['阶段'], r['场景编号'], r['方法']) != (args.stage, i, name)] + [row]
        data['分组汇总'] = [{'阶段': stage, '方法': method,
                           **aggregate([r for r in data['逐局'] if (r['阶段'], r['方法']) == (stage, method)])}
                          for stage, method in sorted({(r['阶段'], r['方法']) for r in data['逐局']})]
        if sum(r['阶段'] == '选参' for r in data['逐局']) == 16 * len(VARIANTS):
            data['选参设置'] = choose(data['逐局'])
        dump(TABLE, data)
        print(f'{args.stage} {n}/{len(jobs)} 场景{i:02d} {name}: {row["清除数"]}/{row["目标数"]}，{row["虚拟时间_秒"]:.1f}秒，无信号{row["补测无信号次数"]}，小范围{row["小范围顺路清除目标数"]}', flush=True)
    print('选参设置:', data.get('选参设置'))


if __name__ == '__main__':
    main()
