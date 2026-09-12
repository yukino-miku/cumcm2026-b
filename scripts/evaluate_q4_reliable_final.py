"""缩减发现保障后的新场景对照；不复用原型107000组进行性能验证。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_q4_reliable import CASES, read, dump, aggregate, summarize
from cumcm2026_b.q4_discovery_strategy import DiscoveryFour, DiscoveryConfig
from cumcm2026_b.q4_reliable_strategy import ReliableFour, ReliableConfig
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from verify_q4_reliable import verify_run

RUNS = ROOT / 'results/models/第四问/清除优先最终验证'
TABLE = ROOT / 'results/tables/第四问/清除优先最终验证.json'
VARIANTS = ['上一版', '仅发现保障', '完整改进']


def config(name):
    if name == '上一版':
        return DiscoveryConfig(localization_detour_limit_m=800)
    if name == '仅发现保障':
        return ReliableConfig(route_feasible_candidates=False, skip_out_of_range=False,
                              unknown_scan_spacing_m=600, unknown_novelty=False)
    return ReliableConfig()


def spec(i):
    _, count, layout, radius, fraction, orientation = CASES[i]
    return dict(seed=108000 + i, count=count, layout=layout, radius_mode=radius,
                error_mode=('hash' if i < 8 else ['plus', 'minus', 'alternating', 'hash'][i - 8]),
                directional_fraction=fraction, orientation=orientation)


def sources():
    files = list((ROOT / 'src/cumcm2026_b').glob('*.py')) + [Path(__file__), ROOT / 'scripts/verify_q4_reliable.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def path_for(i, name):
    return RUNS / f'{i + 1:02d}_{name}.json.gz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    rows = []
    for i in range(len(CASES)):
        truth = None
        for name in VARIANTS:
            path = path_for(i, name)
            if path.exists():
                r = read(path)
                assert r['实验来源SHA256'] == sources() and r['构造参数'] == spec(i)
            else:
                assert not args.verify, str(path)
                env = make_case(**spec(i))
                client = RobotClient(env, 'local-robot')
                try:
                    cls = DiscoveryFour if name == '上一版' else ReliableFour
                    r = cls(client, config(name)).run()
                    r.update(离线真值核验=env.truth_for_evaluation(), 构造参数=spec(i),
                             实验来源SHA256=sources(), 运行类型='新固定场景配对本地构造，非官方成绩')
                    dump(path, r)
                finally:
                    client.close()
            assert r['配置'] == asdict(config(name))
            verify_run(r)
            if truth is not None:
                assert r['离线真值核验']['目标'] == truth
            truth = r['离线真值核验']['目标']
            rows.append(summarize(r, i, name))
            if not args.verify:
                print(f'{i + 1:02d} {name} {r["清除数"]}/{spec(i)["count"]} {r["虚拟总时间_秒"]:.1f}秒，保障{len(r.get("发现保障站坐标", []))}站', flush=True)
    table = {'性质': '12场×3版本同地图本地构造；8一般布局、4边界全朝外压力。不是官方成绩。',
             '规则': '107000组暴露原型耗时后，预先固定compact保障、合格旧候选保留、近站采样增益规则；108000—108011为新组，未据该组重调参数。保证优先于时间。',
             '逐场': rows, '分组': [{'组别': group, '方法': name, **aggregate([r for r in rows if r['组别'] == group and r['方法'] == name])}
                                  for group in ['一般构造', '朝外压力'] for name in VARIANTS]}
    if args.verify:
        assert read(TABLE) == table
        print('核验通过：36次新场景同图运行、接收与清除物理、连续证书、费用和源码。')
    else:
        dump(TABLE, table)


if __name__ == '__main__':
    main()
