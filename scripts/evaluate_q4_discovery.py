"""修正顺路四频道上限后的诊断复查与新一批验证；800米先固定，不重新选参。"""
import argparse
from dataclasses import asdict
from hashlib import sha256

from evaluate_q4_feedback import (ROOT, PROFILES, read, dump, aggregate, verify_run,
                                 verify_extensions, sources as feedback_sources,
                                 path_for as previous_path, summary, run as previous_run)
from cumcm2026_b.q4_discovery_strategy import DiscoveryConfig, DiscoveryFour
from cumcm2026_b.q4_spacing_strategy import SpacingConfig, SpacedFour
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient

RUNS = ROOT / 'results/models/第四问/未知频道补全验证'
TABLE = ROOT / 'results/tables/第四问/未知频道补全验证.json'
BASELINE = '基线100米'
IMPROVED = '完整任务800与未知补全'


def config(name):
    return SpacingConfig(probe_spacing_m=100) if name == BASELINE else DiscoveryConfig(localization_detour_limit_m=800)


def spec(stage, i):
    _, count, layout, radius, fraction, orientation = PROFILES[i % 8]
    return dict(seed=(102000 if stage == '诊断复查' else 103000) + i, count=count, layout=layout,
                radius_mode=radius, error_mode='hash', directional_fraction=fraction, orientation=orientation)


def sources():
    result = feedback_sources()
    for path in ('src/cumcm2026_b/q4_discovery_strategy.py', 'scripts/evaluate_q4_discovery.py'):
        result[path] = sha256((ROOT / path).read_bytes()).hexdigest()
    return result


def run_path(stage, i, name):
    return RUNS / stage / f'{name}-{i:02d}.json.gz'


def summarize(r, stage, i, name):
    row = summary(r, stage, i, name, config(name).localization_detour_limit_m)
    events = r['动作记录']
    row['未知补全停点数'] = sum(e['动作'] == '补齐顺路未知频道' for e in events)
    row['未知补全检测数'] = sum(len(e['候选频道']) for e in events if e['动作'] == '补齐顺路未知频道')
    row['小范围顺路清除数'] = sum(e.get('用途') == '顺路小范围覆盖' and e.get('结果') == 'success' for e in events)
    return row


def run(stage, i, name):
    path = run_path(stage, i, name)
    if path.exists():
        result = read(path)
        assert result['实验来源SHA256'] == sources() and result['构造参数'] == spec(stage, i)
    else:
        env = make_case(**spec(stage, i))
        cfg = config(name)
        client = RobotClient(env, 'local-robot')
        try:
            result = (SpacedFour if name == BASELINE else DiscoveryFour)(client, cfg).run()
            result.update(离线真值核验=env.truth_for_evaluation(), 构造参数=spec(stage, i), 实验来源SHA256=sources(),
                          实验配置=asdict(cfg), 运行类型='本地构造；诊断集不能冒充新验证集，非官方成绩')
            verify_run(result)
            verify_extensions(result)
            dump(path, result)
        finally:
            client.close()
    return summarize(result, stage, i, name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['诊断复查', '新验证', '核验'], required=True)
    args = p.parse_args()
    data = read(TABLE) if TABLE.exists() else {
        '规则': '原800米配置在102000—102007验证新增遗漏，查明实际可接收停点被4频道上限跳过。只补齐合格未知频道，600米重复扫描距离规则不变；800米和其他参数固定。旧8场只作诊断复查，新8场103000—103007与基线配对，未据其结果重调参。',
        '逐局': [], '改进配置': asdict(config(IMPROVED))}
    if args.stage == '核验':
        truth = {}
        for row in data['逐局']:
            stage, i, name = row['阶段'], row['场景编号'], row['方法']
            r = read(run_path(stage, i, name))
            assert r['实验来源SHA256'] == sources() and r['构造参数'] == spec(stage, i)
            assert r['配置'] == r['实验配置'] == asdict(config(name))
            verify_run(r)
            verify_extensions(r)
            assert summarize(r, stage, i, name) == row
            if stage == '诊断复查':
                assert r['离线真值核验']['目标'] == read(previous_path('验证', i, BASELINE))['离线真值核验']['目标']
            if (stage, i) in truth:
                assert truth[stage, i] == r['离线真值核验']['目标']
            truth[stage, i] = r['离线真值核验']['目标']
        print(f'核验通过：{len(data["逐局"])}次补全版配对运行、真值、动作和费用。')
        return
    jobs = [(i, name) for i in range(8) for name in ([IMPROVED] if args.stage == '诊断复查' else [BASELINE, IMPROVED])]
    for n, (i, name) in enumerate(jobs, 1):
        row = run(args.stage, i, name)
        data['逐局'] = [r for r in data['逐局'] if (r['阶段'], r['场景编号'], r['方法']) != (args.stage, i, name)] + [row]
        data['分组汇总'] = [{'阶段': stage, '方法': method,
                           **aggregate([r for r in data['逐局'] if (r['阶段'], r['方法']) == (stage, method)])}
                          for stage, method in sorted({(r['阶段'], r['方法']) for r in data['逐局']})]
        dump(TABLE, data)
        print(f'{args.stage} {n}/{len(jobs)} 场景{i:02d} {name}: {row["清除数"]}/{row["目标数"]}，{row["虚拟时间_秒"]:.1f}秒，补全检测{row["未知补全检测数"]}', flush=True)


if __name__ == '__main__':
    main()
