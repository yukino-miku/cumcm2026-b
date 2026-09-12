"""原几何补测间距对照：预设16场选参和8场验证，仅本地构造。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json

from evaluate_q4_zigzag import ROOT, PROFILES, read, dump, aggregate, summary, sources as original_sources
from cumcm2026_b.q4_spacing_strategy import SpacingConfig, SpacedFour
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from build_q4_inner13_assets import verify_run

RUNS = ROOT/'results/models/第四问/几何补测间距对照'
TABLE = ROOT/'results/tables/第四问/几何补测间距对照.json'
GAPS = (0, 100, 200, 300)


def sources():
    result = original_sources()
    result.pop('src/cumcm2026_b/q4_zigzag_strategy.py')
    result['src/cumcm2026_b/q4_spacing_strategy.py'] = sha256((ROOT/'src/cumcm2026_b/q4_spacing_strategy.py').read_bytes()).hexdigest()
    return result


def spec(stage, i):
    _, count, layout, radius, fraction, orientation = PROFILES[i % 8]
    return dict(seed=(98000 if stage == '选参' else 99000)+i, count=count, layout=layout,
                radius_mode=radius, error_mode='hash', directional_fraction=fraction, orientation=orientation)


def summarize(r, stage, i, gap):
    row = summary(r, stage, i, '原几何间距', 400)
    plans = [e['方案'] for e in r['动作记录'] if e['动作'] == '规划']
    probes = [e for e in r['动作记录'] if e.get('用途') == '追加测向']
    row.update(间距_米=gap,间距放宽次数=sum(p.get('间距约束已放宽', False) for p in plans),
               顺路首次发现数=sum(e.get('用途') == '顺路检测' for e in first_positive(r)),
               补测位置数=len({tuple(e['位置']) for e in probes}))
    return row


def first_positive(r):
    seen = set()
    for e in r['动作记录']:
        if e['动作'] == '检测' and e.get('结果') in ('direction', 'near') and e['频道'] not in seen:
            seen.add(e['频道']); yield e


def run(stage, i, gap):
    path = RUNS/stage/f'间距{gap}-{i:02d}.json.gz'
    if path.exists():
        r = read(path)
        assert r['实验来源SHA256'] == sources() and r['构造参数'] == spec(stage, i)
        return summarize(r, stage, i, gap)
    env = make_case(**spec(stage, i))
    cfg = SpacingConfig(probe_spacing_m=gap)
    client = RobotClient(env, 'local-robot')
    try:
        r = SpacedFour(client, cfg).run()
        r.update(离线真值核验=env.truth_for_evaluation(), 构造参数=spec(stage, i),
                 实验来源SHA256=sources(), 实验配置=asdict(cfg), 运行类型='本地构造，不是官方成绩')
        verify_run(r)
        dump(path, r)
        return summarize(r, stage, i, gap)
    finally: client.close()


def choose(rows, nonzero_only=False):
    choices = []
    for gap in GAPS:
        if nonzero_only and gap == 0: continue
        group = [r for r in rows if r['阶段'] == '选参' and r['间距_米'] == gap]
        assert len(group) == 16
        g = aggregate(group)
        choices.append((g['异常局数'],g['漏源总数'],-g['全清局数'],g['平均虚拟时间_秒'],gap))
    return min(choices)[-1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['选参','验证','核验'],required=True); args=p.parse_args()
    data=read(TABLE) if TABLE.exists() else {'规则':'原几何候选及评分；0表示不加间距；16场选参先少漏源再比时间，8场验证不参与选参。绕行门槛400米、顺路扫描间隔600米不变。','逐局':[]}
    if args.stage == '核验':
        truth = {}
        for row in data['逐局']:
            stage,i,gap = row['阶段'],row['场景编号'],row['间距_米']
            r=read(RUNS/stage/f'间距{gap}-{i:02d}.json.gz')
            assert r['实验来源SHA256'] == sources() and r['构造参数']==spec(stage,i)
            assert r['配置']==r['实验配置'] and r['配置']['probe_spacing_m']==gap
            assert summarize(r,stage,i,gap)==row
            t=r['离线真值核验']['目标']
            if (stage,i) in truth: assert truth[stage,i]==t
            truth[stage,i]=t
            verify_run(r)
            if gap:
                for e in r['动作记录']:
                    if e['动作']=='规划' and not e['方案']['间距约束已放宽']:
                        assert e['方案']['距当前机器人_米']>=gap-1e-6
                        assert e['方案']['距该频道上次观测_米']>=gap-1e-6
        assert data['选参间距_米']==choose(data['逐局'])
        print(f'核验通过：{len(data["逐局"])}次完整动作、费用、真值、间距规则。')
        return
    jobs=[(i,gap) for i in range(16 if args.stage=='选参' else 8)
          for gap in (GAPS if args.stage=='选参' else sorted({0,choose(data['逐局'], nonzero_only=True)}))]
    for n,(i,gap) in enumerate(jobs,1):
        row=run(args.stage,i,gap)
        data['逐局']=[r for r in data['逐局'] if (r['阶段'],r['场景编号'],r['间距_米'])!=(args.stage,i,gap)]+[row]
        data['分组汇总']=[{'阶段':s,'间距_米':g,**aggregate([r for r in data['逐局'] if (r['阶段'],r['间距_米'])==(s,g)])}
                           for s,g in sorted({(r['阶段'],r['间距_米']) for r in data['逐局']})]
        if sum(r['阶段']=='选参' for r in data['逐局'])==64:
            data['选参间距_米']=choose(data['逐局'])
            data['非零备选间距_米']=choose(data['逐局'],nonzero_only=True)
        dump(TABLE,data)
        print(f'{args.stage}{n}/{len(jobs)} 场景{i:02d} 间距{gap}: {row["清除数"]}/{row["目标数"]}，{row["虚拟时间_秒"]:.1f}秒',flush=True)
    print('选参间距',data.get('选参间距_米'))


if __name__=='__main__':main()
