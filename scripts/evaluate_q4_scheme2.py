"""19站方案二与当前13站方案一的8场同图本地对照；--verify只审计已有结果。"""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q4_reliable_strategy import ReliableConfig, ReliableFour
from cumcm2026_b.q4_scheme2 import SchemeTwo, SchemeTwoConfig
from cumcm2026_b.q4_scheme2.route import shortest_route
from verify_q4_scheme2 import verify_run

RUNS = ROOT / 'results/models/第四问/方案二十九站'
TABLE = ROOT / 'results/tables/第四问/方案二十九站_同图对照.json'
ROUTE = ROOT / 'results/tables/第四问/方案二十九站_最短路线.json'
CASES = [
    ('均匀混合', 10, 'uniform', 'min', .5, 'random'),
    ('均匀定向', 16, 'uniform', 'mixed', 1., 'random'),
    ('边界混合', 13, 'boundary', 'min', .5, 'random'),
    ('边界切向', 12, 'boundary', 'min', 1., 'tangent'),
    ('聚集混合', 14, 'cluster', 'mixed', .5, 'random'),
    ('中心定向', 13, 'center', 'min', 1., 'random'),
    ('边界全部朝外10源', 10, 'boundary', 'min', 1., 'outward'),
    ('边界全部朝外16源', 16, 'boundary', 'min', 1., 'outward'),
]


def read(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def sources():
    files = (list((ROOT / 'src/cumcm2026_b').glob('*.py'))
             + list((ROOT / 'src/cumcm2026_b/q4_scheme2').glob('*.py'))
             + [Path(__file__).resolve(), ROOT / 'scripts/verify_q4_scheme2.py',
                ROOT / 'configs/q4_reliable.json', ROOT / 'configs/q4_scheme2.json'])
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def path_for(i, scheme):
    return RUNS / f'{i+1:02d}_方案{scheme}.json.gz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    parameters = read(ROOT / 'configs/q4_reliable.json')
    assert parameters == read(ROOT / 'configs/q4_scheme2.json')
    manifest = sources()
    rows = []
    for i, (name, count, layout, radius, fraction, orientation) in enumerate(CASES):
        spec = dict(seed=110000+i, count=count, layout=layout, radius_mode=radius,
                    error_mode='hash', directional_fraction=fraction, orientation=orientation)
        truth = None
        for scheme in [1, 2]:
            path = path_for(i, scheme)
            if path.exists():
                result = read(path)
            else:
                assert not args.verify, str(path)
                env = make_case(**spec)
                client = RobotClient(env, 'local-robot')
                try:
                    config = (ReliableConfig if scheme == 1 else SchemeTwoConfig)(**parameters)
                    result = (ReliableFour if scheme == 1 else SchemeTwo)(client, config).run()
                    result.update(离线真值核验=env.truth_for_evaluation(), 构造参数=spec,
                                  实验来源SHA256=manifest, 运行类型='同图本地构造对照，非官方成绩')
                    if scheme == 1:
                        result['方案'] = '方案一：13站，已取消自动补漏，保留路线修复及超距免测'
                        result['末尾增站评估']['本版处理'] = '13站及已发现源处理后退出，不自动未知源补漏'
                    dump(path, result)
                finally:
                    client.close()
            assert result['实验来源SHA256'] == manifest
            assert result['构造参数'] == spec and result['配置'] == parameters
            verify_run(result)
            targets = result['离线真值核验']['目标']
            assert truth is None or targets == truth
            truth = targets
            rows.append({'场景': i+1, '名称': name, '方案': scheme, '目标数': count,
                         '清除数': result['清除数'], '实际全清': result['离线真值核验']['全部清除'],
                         '遗漏频道': result['离线真值核验']['遗漏频道'],
                         '虚拟时间_秒': result['虚拟总时间_秒'], '路程_米': result['总路程_米'],
                         '检测次数': result['检测次数'], '固定站完成数': len(result['已完成搜索站']),
                         '主动补测次数': sum(e.get('用途') == '追加测向' for e in result['动作记录']),
                         '补测无信号次数': result['路线调度统计']['补测无信号次数'],
                         '末次清除后剩余流程时间_秒': (result['虚拟总时间_秒'] - max(
                             (e['虚拟时间_秒'] for e in result['动作记录']
                              if e['动作'] == '清除' and e['结果'] == 'success'), default=0))
                             if result['清除数'] else None})
            if not args.verify:
                print(f'{i+1:02d} {name} 方案{scheme}：{result["清除数"]}/{count}，'
                      f'{result["虚拟总时间_秒"]:.1f}秒，{result["总路程_米"]/1000:.3f}公里', flush=True)
    groups = []
    for scheme in [1, 2]:
        group = [r for r in rows if r['方案'] == scheme]
        groups.append({'方案': scheme, '总目标数': sum(r['目标数'] for r in group),
                       '总清除数': sum(r['清除数'] for r in group),
                       '实际全清局数': sum(r['实际全清'] for r in group),
                       '平均时间_秒': sum(r['虚拟时间_秒'] for r in group) / len(group),
                       '平均路程_米': sum(r['路程_米'] for r in group) / len(group)})
    table = {'性质': '8场预定同图本地对照，含2场全部朝外压力构造；没有调参择优，不是官方成绩。',
             '比较规则': '先看遗漏和全清，再比较时间；漏源提前退出的短时间不能算效率胜出。',
             '配置': parameters, '逐场': rows, '汇总': groups}
    if args.verify:
        assert read(TABLE) == table and read(ROUTE) == shortest_route()[1]
        print('16次同图运行逐动作物理、费用、外包、站序、来源及汇总核验通过。')
    else:
        dump(TABLE, table)
        dump(ROUTE, shortest_route()[1])
        print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
