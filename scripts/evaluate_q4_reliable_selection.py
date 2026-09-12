"""密集补扫变慢后的配置收敛：12场比较，冻结选择，再用8个新场景复核。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from evaluate_q4_reliable_final import (ROOT, CASES, read, dump, aggregate, summarize,
                                        config as previous_config, spec as previous_spec, path_for as previous_path)
from cumcm2026_b.q4_reliable_strategy import ReliableConfig, ReliableFour
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from verify_q4_reliable import verify_run

RUNS = ROOT / 'results/models/第四问/清除优先配置收敛'
TABLE = ROOT / 'results/tables/第四问/清除优先配置收敛.json'
NAMES = ['路线修复与免测', '仅超距免测']


def config(name):
    if name == '仅发现保障':
        return previous_config(name)
    return ReliableConfig(unknown_scan_spacing_m=600, unknown_novelty=False,
                          route_feasible_candidates=name == '路线修复与免测')


def sources():
    files = list((ROOT / 'src/cumcm2026_b').glob('*.py')) + [Path(__file__), ROOT / 'scripts/verify_q4_reliable.py',
             ROOT / 'scripts/evaluate_q4_reliable_final.py', ROOT / 'scripts/evaluate_q4_reliable.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def spec(stage, i):
    r = previous_spec(i if stage == '配置比较' or i < 6 else i + 2)
    if stage == '独立复核':
        r['seed'] = 109000 + i
    return r


def path_for(stage, i, name):
    return RUNS / stage / f'{i + 1:02d}_{name}.json.gz'


def run(stage, i, name, verify):
    path = path_for(stage, i, name)
    if path.exists():
        r = read(path)
        assert r['实验来源SHA256'] == sources()
    else:
        assert not verify, str(path)
        env = make_case(**spec(stage, i))
        client = RobotClient(env, 'local-robot')
        try:
            r = ReliableFour(client, config(name)).run()
            r.update(离线真值核验=env.truth_for_evaluation(), 构造参数=spec(stage, i),
                     实验来源SHA256=sources(), 运行类型='本地配置收敛与新场景复核，非官方成绩')
            dump(path, r)
        finally:
            client.close()
    assert r['配置'] == asdict(config(name)) and r['构造参数'] == spec(stage, i)
    verify_run(r)
    index = i if stage == '配置比较' or i < 6 else i + 2
    row = {**summarize(r, index, name), '阶段': stage, '组内编号': i}
    if not verify:
        print(f'{stage} {i + 1:02d} {name} {r["清除数"]}/{r["离线真值核验"]["目标数"]} {r["虚拟总时间_秒"]:.1f}秒', flush=True)
    return row, r['离线真值核验']['目标']


def choose(rows):
    def rank(name):
        group = [r for r in rows if r['方法'] == name]
        return (sum(not r['流程正常'] for r in group), sum(r['目标数'] - r['清除数'] for r in group),
                -sum(r['有全清证据'] for r in group), sum(r['时间_秒'] for r in group), name)
    return min(['仅发现保障'] + NAMES, key=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    rows = []
    for i in range(12):
        previous = read(previous_path(i, '仅发现保障'))
        rows.append({**summarize(previous, i, '仅发现保障'), '阶段': '配置比较', '组内编号': i})
        for name in NAMES:
            row, truth = run('配置比较', i, name, args.verify)
            assert truth == previous['离线真值核验']['目标']
            rows.append(row)
    selected = choose(rows)
    names = list(dict.fromkeys(['仅发现保障', selected]))
    for i in range(8):
        truth = None
        for name in names:
            row, targets = run('独立复核', i, name, args.verify)
            assert truth is None or truth == targets
            truth = targets
            rows.append(row)
    data = {'规则': '108000组为配置比较，不能再冒充独立验证；比较仅发现保障、路线修复加免测、仅免测，按异常、遗漏、证据、总时间顺序选定。选择冻结后109000—109007作8场独立复核，含6个一般布局与2个全朝外压力布局，不根据复核结果重调参数。',
            '选择': selected, '默认配置': asdict(config(selected)), '逐场': rows,
            '分组': [{'阶段': stage, '方法': name, **aggregate([r for r in rows if r['阶段'] == stage and r['方法'] == name])}
                     for stage, name in sorted({(r['阶段'], r['方法']) for r in rows})]}
    if args.verify:
        assert read(TABLE) == data
        print('配置比较、选择依据与独立复核全部核验通过。')
    else:
        dump(TABLE, data)
        print('选择：' + selected, flush=True)


if __name__ == '__main__':
    main()
