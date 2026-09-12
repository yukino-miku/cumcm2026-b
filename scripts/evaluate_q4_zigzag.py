"""固定种子配对比较折线补测与绕行门槛；仅运行本地物理环境。"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cumcm2026_b.q4_strategy import Q4Config, SchemeFour
from cumcm2026_b.q4_zigzag_strategy import ZigzagConfig, ZigzagFour
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from build_q4_inner13_assets import verify_run

MODELS = ROOT/'results/models/第四问/折线门槛对照'
TABLE = ROOT/'results/tables/第四问/折线门槛对照.json'
THRESHOLDS = (400, 600, 800, 1000, 1200, 1600)
# 在读取任何本轮对照结果前固定场景、种子、门槛和评价顺序。
PROFILES = [
    ('均匀混合', 10, 'uniform', 'min', .5, 'random'),
    ('均匀定向变半径', 16, 'uniform', 'mixed', 1., 'random'),
    ('边界切向定向', 13, 'boundary', 'min', 1., 'tangent'),
    ('边界混合变半径', 16, 'boundary', 'mixed', .5, 'random'),
    ('聚集定向', 12, 'cluster', 'min', 1., 'random'),
    ('中心混合', 14, 'center', 'min', .5, 'random'),
    ('均匀定向最小半径', 13, 'uniform', 'min', 1., 'random'),
    ('边界朝内混合', 12, 'boundary', 'min', .5, 'inward'),
]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(raw, mtime=0))
    else:
        path.write_bytes(raw)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sources():
    paths = [ROOT/'src/cumcm2026_b'/name for name in ['q4_strategy.py', 'q4_zigzag_strategy.py',
              'q4_local_env.py', 'q3_local_env.py', 'q3_strategy.py', 'q3_geometry.py',
              'q3_route_planning.py', 'q3_routed_strategies.py', 'q3_protocol.py',
              'q2_physical_geometry.py', 'q2_active_localization.py', 'q1_geometry.py']]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}


def spec(stage, i):
    label, count, layout, radius, fraction, orientation = PROFILES[i % len(PROFILES)]
    return {'seed': (96000 if stage == '选参' else 97000)+i, 'count': count, 'layout': layout,
            'radius_mode': radius, 'error_mode': 'hash', 'directional_fraction': fraction, 'orientation': orientation}


def summary(result, stage, i, method, threshold):
    events = result['动作记录']
    mark = next(e for e in events if e['动作'] == '固定扫描完成')
    tail_distance = 0.; position = [0., 0.]
    for e in events:
        if e['动作'] not in ('检测', '清除'): continue
        if e['步骤'] > mark['步骤']: tail_distance += math.dist(position, e['位置'])
        position = e['位置']
    probes = [e for e in events if e.get('用途') == '追加测向']
    truth = result['离线真值核验']
    return {'阶段': stage, '场景编号': i, '场景名称': PROFILES[i % 8][0], '方法': method,
            '门槛_米': threshold, '目标数': truth['目标数'], '清除数': result['清除数'],
            '实际全清': truth['全部清除'], '漏源数': len(truth['遗漏频道']),
            '流程正常': result['流程正常完成'], '虚拟时间_秒': result['虚拟总时间_秒'],
            '路程_米': result['总路程_米'], '尾程_米': tail_distance,
            '补测次数': len(probes), '补测无信号次数': sum(e['结果'] == 'no_signal' for e in probes),
            '清除失败次数': result['清除失败次数'], '时间分项_秒': result['时间分项_秒'],
            '检测次数': result['检测次数'], '有限覆盖次数': sum(e['动作'] == '有限覆盖' for e in events),
            '延后次数': sum(e['动作'] == '延后目标' for e in events), '现实运行时间_秒': result['程序运行时间_秒']}


def run_one(stage, i, method, threshold):
    path = MODELS/stage/f'{method}-{threshold}-{i:02d}.json.gz'
    settings = spec(stage, i)
    if path.exists():
        result = read(path)
        assert result['实验来源SHA256'] == sources(), '运行源码变更；不可混用旧实验'
        assert result['构造参数'] == settings
        return summary(result, stage, i, method, threshold)
    env = make_case(**settings)
    config = (ZigzagConfig if method == '折线' else Q4Config)(localization_detour_limit_m=float(threshold))
    client = RobotClient(env, 'local-robot')
    try:
        result = (ZigzagFour if method == '折线' else SchemeFour)(client, config).run()
        result['离线真值核验'] = env.truth_for_evaluation()
        result.update(构造参数=settings,实验来源SHA256=sources(),实验配置=asdict(config),
                      运行类型='本地固定种子配对构造；非官方演练或正式成绩')
        verify_run(result)  # 独立核对真值、观测、清除及所有费用。
        dump(path, result)
        return summary(result, stage, i, method, threshold)
    finally:
        client.close()


def aggregate(rows):
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows)/n
    return {'局数': n, '全清局数': sum(r['实际全清'] for r in rows), '目标总数': sum(r['目标数'] for r in rows),
            '清除总数': sum(r['清除数'] for r in rows), '漏源总数': sum(r['漏源数'] for r in rows),
            '异常局数': sum(not r['流程正常'] for r in rows),
            **{'平均'+k: avg(k) for k in ['虚拟时间_秒', '路程_米', '尾程_米', '补测次数', '补测无信号次数', '清除失败次数', '检测次数']}}


def selection(data):
    selected = {}
    for method in ('几何', '折线'):
        groups = []
        for threshold in THRESHOLDS:
            group = [r for r in data['逐局'] if r['阶段'] == '选参' and r['方法'] == method and r['门槛_米'] == threshold]
            assert len(group) == 16
            a = aggregate(group)
            groups.append((a['异常局数'], a['漏源总数'], -a['全清局数'], a['平均虚拟时间_秒'], threshold))
        selected[method] = min(groups)[-1]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['选参', '验证', '核验'], required=True)
    args = parser.parse_args()
    data = read(TABLE) if TABLE.exists() else {'规则': '先最少异常、最少漏源、最多全清局，再比较平均虚拟时间；仅在选参16场选门槛，8场独立验证不参与调参', '逐局': []}
    if args.stage == '核验':
        for r in data['逐局']:
            path = MODELS/r['阶段']/f'{r["方法"]}-{r["门槛_米"]}-{r["场景编号"]:02d}.json.gz'
            result = read(path)
            assert result['实验来源SHA256'] == sources()
            verify_run(result)
            assert summary(result, r['阶段'], r['场景编号'], r['方法'], r['门槛_米']) == r
        assert data['选参门槛'] == selection(data)
        print(f'已核验{len(data["逐局"])}局原始动作、物理反馈、虚拟费用和选参规则。')
        return
    if args.stage == '选参':
        jobs = [(i, method, threshold) for i in range(16) for method in ('几何', '折线') for threshold in THRESHOLDS]
    else:
        chosen = selection(data)
        variants = sorted({('几何', 400), ('几何', chosen['几何']), ('折线', chosen['折线'])})
        jobs = [(i, method, threshold) for i in range(8) for method, threshold in variants]
    for index, (i, method, threshold) in enumerate(jobs, 1):
        row = run_one(args.stage, i, method, threshold)
        data['逐局'] = [r for r in data['逐局'] if (r['阶段'], r['场景编号'], r['方法'], r['门槛_米']) != (args.stage, i, method, threshold)]+[row]
        data['分组汇总'] = [{'阶段': stage, '方法': m, '门槛_米': t, **aggregate([r for r in data['逐局'] if (r['阶段'],r['方法'],r['门槛_米']) == (stage,m,t)])}
                        for stage,m,t in sorted({(r['阶段'],r['方法'],r['门槛_米']) for r in data['逐局']})]
        if sum(r['阶段'] == '选参' for r in data['逐局']) == 192:
            data['选参门槛'] = selection(data)
        dump(TABLE, data)
        print(f'{args.stage} {index}/{len(jobs)} 场景{i:02d} {method}{threshold}: {row["清除数"]}/{row["目标数"]}，{row["虚拟时间_秒"]:.1f}秒，补测{row["补测次数"]}次/无信号{row["补测无信号次数"]}次', flush=True)
    print('当前选参门槛：', data.get('选参门槛'))


if __name__ == '__main__': main()
