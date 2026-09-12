"""清除优先改进的四版本同地图对照；固定12场，先比较漏源，再比较费用。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_q4_zigzag import PROFILES, read, dump
from cumcm2026_b.q4_discovery_strategy import DiscoveryFour, DiscoveryConfig
from cumcm2026_b.q4_reliable_strategy import ReliableFour, ReliableConfig
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from verify_q4_reliable import verify_run

RUNS = ROOT / 'results/models/第四问/清除优先对照'
TABLE = ROOT / 'results/tables/第四问/清除优先对照.json'
VARIANTS = ['上一版', '仅发现保障', '仅效率改进', '完整改进']
CASES = PROFILES + [('边界朝外10源', 10, 'boundary', 'min', 1., 'outward'),
                    ('边界朝外16源', 16, 'boundary', 'min', 1., 'outward'),
                    ('边界朝外变半径', 13, 'boundary', 'mixed', 1., 'outward'),
                    ('边界朝外最大半径', 12, 'boundary', 'max', 1., 'outward')]


def config(name):
    if name == '上一版':
        return DiscoveryConfig(localization_detour_limit_m=800)
    if name == '仅发现保障':
        return ReliableConfig(route_feasible_candidates=False, skip_out_of_range=False, unknown_scan_spacing_m=600)
    return ReliableConfig(discovery_safeguard=name == '完整改进')


def spec(i):
    _, count, layout, radius, fraction, orientation = CASES[i]
    return dict(seed=107000 + i, count=count, layout=layout, radius_mode=radius,
                error_mode=('hash' if i < 8 else ['plus', 'minus', 'alternating', 'hash'][i - 8]),
                directional_fraction=fraction, orientation=orientation)


def sources():
    files = list((ROOT / 'src/cumcm2026_b').glob('*.py')) + [Path(__file__), ROOT / 'scripts/verify_q4_reliable.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def path_for(i, name):
    return RUNS / f'{i + 1:02d}_{name}.json.gz'


def summarize(r, i, name):
    events = r['动作记录']
    begin = next((e['虚拟时间_秒'] for e in events if e['动作'] == '开始发现保障'), r['虚拟总时间_秒'])
    return {'编号': i + 1, '场景': CASES[i][0], '组别': '一般构造' if i < 8 else '朝外压力', '方法': name,
            '目标数': r['离线真值核验']['目标数'], '清除数': r['清除数'], '全清': r['离线真值核验']['全部清除'],
            '有全清证据': r['全部完成证据'], '流程正常': r['流程正常完成'],
            '时间_秒': r['虚拟总时间_秒'], '路程_米': r['总路程_米'],
            '发现保障阶段时间_秒': r['虚拟总时间_秒'] - begin,
            '保障站数': len(r.get('发现保障站坐标', [])),
            '保障检测数': sum(e.get('用途') == '发现保障补扫' for e in events),
            '顺路检测数': sum(e.get('用途') == '顺路检测' for e in events),
            '跳过超距数': sum('距离下界_米' in e for e in events),
            '主补测数': sum(e.get('用途') == '追加测向' for e in events),
            '主补测无信号': sum(e.get('用途') == '追加测向' and e.get('结果') == 'no_signal' for e in events),
            '失败清除数': r['清除失败次数']}


def aggregate(rows):
    return {'场次': len(rows), '全清场次': sum(r['全清'] for r in rows),
            '有证据场次': sum(r['有全清证据'] for r in rows), '目标总数': sum(r['目标数'] for r in rows),
            '清除总数': sum(r['清除数'] for r in rows), '遗漏总数': sum(r['目标数'] - r['清除数'] for r in rows),
            '平均时间_秒': sum(r['时间_秒'] for r in rows) / len(rows),
            '平均路程_米': sum(r['路程_米'] for r in rows) / len(rows),
            '平均保障阶段时间_秒': sum(r['发现保障阶段时间_秒'] for r in rows) / len(rows),
            '平均保障站数': sum(r['保障站数'] for r in rows) / len(rows)}


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
                             实验来源SHA256=sources(), 运行类型='固定场景配对本地构造，非官方成绩')
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
                print(f'{i + 1:02d} {name} {r["清除数"]}/{spec(i)["count"]} {r["虚拟总时间_秒"]:.1f}秒', flush=True)
    table = {'性质': '12场×4版本同地图本地构造；8一般布局、4边界全朝外压力。不是官方成绩。',
             '规则': '在运行前固定种子107000—107011、800米绕行与100米间距。优先全清；压力与一般布局分开统计。无结果驱动的选参。',
             '逐场': rows, '分组': [{'组别': group, '方法': name, **aggregate([r for r in rows if r['组别'] == group and r['方法'] == name])}
                                  for group in ['一般构造', '朝外压力'] for name in VARIANTS]}
    if args.verify:
        assert read(TABLE) == table
        print('核验通过：48次同地图运行、物理反馈、连续证书、费用与源码。')
    else:
        dump(TABLE, table)


if __name__ == '__main__':
    main()
