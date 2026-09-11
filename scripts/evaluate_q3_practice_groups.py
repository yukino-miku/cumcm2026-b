#!/usr/bin/env python
"""评估用户提供的s1/s2演练组：核对目标总数、日志归属、费用与比较口径。"""
from __future__ import annotations
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from analyze_q3_practice_log import public_header, matched_run, audit_actions
from build_q3_scheme1_assets import style, save
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

DEFAULT = ROOT/"materials/original/simulator/CUMCM2026B/Jammers-simulator-win64/Jammers-simulator/JammersSimulatorData/behavior-logs"
OUTPUT = ROOT/"results/tables/第三问/两组演练评估_s1_s2.json"
FIGURES = ["两组演练_每目标时间与费用", "两组演练_同目标数分组与逐局结果"]


def digest(path): return sha256(path.read_bytes()).hexdigest()


def gather(base):
    rows = []; provenance = {}
    for group,name in [("s1","方案一"),("s2","方案二")]:
        files = sorted((base/group).glob("*.result.json"))
        if not files: raise ValueError(f"{group}无结果文件")
        stems = {p.name[:-len('.result.json')] for p in files}
        assert {p.stem for p in (base/group).glob('*.jlog')} == stems
        assert {p.stem for p in (base/group).glob('*.psum')} == stems
        for path in files:
            official = json.loads(path.read_text(encoding="utf-8-sig"))
            stem = path.name[:-len('.result.json')]
            jlog = path.with_name(stem+'.jlog'); psum = path.with_name(stem+'.psum')
            header,_ = public_header(jlog)
            raw = psum.read_bytes()
            assert raw[:8] == b"JMBPSUM1" and int.from_bytes(raw[8:10],'big') == 1
            length = int.from_bytes(raw[10:14],'big')
            assert 0 < length <= len(raw)-14
            ph = json.loads(raw[14:14+length])
            assert official['package_sha256'] == digest(jlog)
            assert header['package_type'] == 'practice_behavior_log' and ph['package_type'] == 'practice_summary'
            assert header['formal_index'] is None and ph['formal_index'] is None
            for key in ['problem_no','case_code','practice_run_no']:
                assert official[key] == header[key] == ph[key]
            assert header['problem_no'] == 3 and ph['team_no'] == header['team_no']
            assert official['jammer_count'] == official['omnidirectional_jammer_count']
            assert official['directional_jammer_count'] == 0
            assert 10 <= official['jammer_count'] <= 16
            folder,items,time_delta = matched_run(header)
            result_path = folder/'运行结果.json'
            result = json.loads(result_path.read_text(encoding='utf-8'))
            assert result['方案'].startswith(name+'：'),(group,result['方案'])
            audit = audit_actions(items,result)
            complete = result['运行成功'] and result['正常退出'] and result['清除数'] == official['jammer_count']
            positives = [e for e in result['动作记录'] if e['动作']=='清除' and e['结果']=='success']
            assert len({e['频道'] for e in positives}) == result['清除数']
            false_clear = [e for e in result['动作记录'] if e['动作']=='清除' and e['结果']!='success']
            assert all(not e['具有覆盖证书'] for e in false_clear)
            row = {'组':group,'方案':name,'案例码':official['case_code'],'演练编号':str(official['practice_run_no']),
                '结束时间UTC':official['ended_at_utc'],'客户端版本':header['client_version'],
                '官方结果文件目标总数':official['jammer_count'],'成功清除数':result['清除数'],
                '全清核对通过':complete,'本地判定成功':result['运行成功'],'异常':result['异常'],
                '总虚拟时间_秒':result['虚拟总时间_秒'],
                '每目标虚拟时间_秒':result['虚拟总时间_秒']/official['jammer_count'],
                '总路程_米':result['总路程_米'],'检测次数':result['检测次数'],'切换次数':result['频道切换次数'],
                '专门追加测向次数':sum(s['追加测向次数'] for s in result['频道记录']),
                '失败清除次数':result['清除失败次数'],'失败清除用途':[e['用途'] for e in false_clear],
                '本地程序时间_秒':result['程序运行时间_秒'],'时间分项_秒':result['时间分项_秒'],
                '配置':result['配置'],'请求响应核验':audit,'关联时间差_毫秒':time_delta,
                '关联本地目录':folder.relative_to(ROOT).as_posix(),
                '官方结果文件':path.relative_to(ROOT).as_posix(),'jlog_SHA256':digest(jlog),
                '扫描阶段':{k:v for k,v in result.get('扫描阶段',{}).items() if k!='频道快照'}}
            assert abs(row['每目标虚拟时间_秒']-result['平均定位清除时间_秒']) < 1e-8
            rows.append(row)
            for source in [path,jlog,psum,result_path,folder/'请求响应日志.jsonl']:
                provenance[source.relative_to(ROOT).as_posix()] = digest(source)
    rows.sort(key=lambda r:(r['组'],r['结束时间UTC']))
    for group in ('s1','s2'):
        subset = [r for r in rows if r['组']==group]
        assert len({json.dumps(r['配置'],sort_keys=True) for r in subset}) == 1,'同组配置不一致'
        for i,row in enumerate(subset,1): row['显示编号'] = f'{group.upper()}-{i}'
    assert len({r['关联本地目录'] for r in rows}) == len(rows),'本地运行被重复使用'
    return rows,provenance


def statistics(rows):
    stats = {}
    keys = ['官方结果文件目标总数','总虚拟时间_秒','每目标虚拟时间_秒','总路程_米','检测次数','切换次数','专门追加测向次数','本地程序时间_秒']
    for group in ('s1','s2'):
        rs = [r for r in rows if r['组']==group]
        stats[group] = {'演练数':len(rs),'全清次数':sum(r['全清核对通过'] for r in rs),
            **{f'平均{k}':float(np.mean([r[k] for r in rs])) for k in keys},
            '清除目标总数':sum(r['成功清除数'] for r in rs),'失败清除总次数':sum(r['失败清除次数'] for r in rs),
            '重试总次数':sum(r['请求响应核验']['重试请求数'] for r in rs),
            '通信异常总次数':sum(r['请求响应核验']['通信异常数'] for r in rs),
            '最大总虚拟时间_秒':max(r['总虚拟时间_秒'] for r in rs),
            '每目标时间中位数_秒':float(np.median([r['每目标虚拟时间_秒'] for r in rs])),
            '每目标时间样本标准差_秒':float(np.std([r['每目标虚拟时间_秒'] for r in rs],ddof=1)),
            '合计时间除以合计目标_秒':sum(r['总虚拟时间_秒'] for r in rs)/sum(r['官方结果文件目标总数'] for r in rs),
            '平均时间分项_秒':{k:float(np.mean([r['时间分项_秒'][k] for r in rs])) for k in rs[0]['时间分项_秒']}}
    common = sorted(set(r['官方结果文件目标总数'] for r in rows if r['组']=='s1') &
                    set(r['官方结果文件目标总数'] for r in rows if r['组']=='s2'))
    strata = []
    for count in common:
        row = {'目标数':count}
        for group in ('s1','s2'):
            rs = [r for r in rows if r['组']==group and r['官方结果文件目标总数']==count]
            row[group] = {'样本数':len(rs),'平均每目标时间_秒':float(np.mean([r['每目标虚拟时间_秒'] for r in rs]))}
        row['方案一相对方案二减少百分比'] = (1-row['s1']['平均每目标时间_秒']/row['s2']['平均每目标时间_秒'])*100
        strata.append(row)
    standardized = {g:float(np.mean([r[g]['平均每目标时间_秒'] for r in strata])) for g in ('s1','s2')}
    x = np.array([r['每目标虚拟时间_秒'] for r in rows if r['组']=='s1'])
    y = np.array([r['每目标虚拟时间_秒'] for r in rows if r['组']=='s2'])
    a,b = x.var(ddof=1)/len(x),y.var(ddof=1)/len(y)
    df = (a+b)**2/(a*a/(len(x)-1)+b*b/(len(y)-1))
    interval = (x.mean()-y.mean()+np.array([-1,1])*t.ppf(.975,df)*np.sqrt(a+b)).tolist()
    comparison = {'相同案例码数':len(set(r['案例码'] for r in rows if r['组']=='s1') & set(r['案例码'] for r in rows if r['组']=='s2')),
        '平均总时间减少百分比':(1-stats['s1']['平均总虚拟时间_秒']/stats['s2']['平均总虚拟时间_秒'])*100,
        '每局每目标时间均值减少百分比':(1-x.mean()/y.mean())*100,
        '共同目标数分层':strata,'共同层等权每目标均值_秒':standardized,
        '共同层等权减少百分比':(1-standardized['s1']/standardized['s2'])*100,
        '非配对均值差S1减S2_秒':float(x.mean()-y.mean()),'探索性Welch95区间_秒':interval,
        '区间假设与限制':'独立样本的近似Welch均值差区间；每组仅6局、未确认随机分配、目标数分布不同，不是算法因果效应区间'}
    return stats,comparison


def figures(rows,stats,comparison):
    style(); plt.rcParams['svg.hashsalt'] = 'q3-practice-groups'
    colors = {'s1':'#347CA0','s2':'#C77B39'}
    note = '用户提供的两组实际演练，各6局；案例码不重合，未确认同场景配对；目标总数已与结果文件核对'
    fig,axes = plt.subplots(1,2,figsize=(13,6.5))
    for i,group in enumerate(('s1','s2')):
        rs = [r for r in rows if r['组']==group]
        values = np.array([r['每目标虚拟时间_秒'] for r in rs])
        axes[0].scatter(i+np.linspace(-.14,.14,len(rs)),values,color=colors[group],s=55,zorder=3)
        axes[0].hlines(values.mean(),i-.32,i+.32,color=colors[group],lw=3)
        axes[0].text(i+.35,values.mean(),f'均值{values.mean():.2f}',ha='left',va='center',color=colors[group],fontsize=9)
    axes[0].set(xticks=[0,1],xticklabels=['方案一：6局','方案二：6局'],ylabel='每局总虚拟时间 / 目标数（秒）',
                title='每个点是一局；短横线为均值',xlim=(-.65,1.85))
    axes[0].grid(axis='y',alpha=.2)
    bottom = np.zeros(2)
    parts = ['移动','检测','切换','清除成功','清除失败']
    for part,color in zip(parts,['#347CA0','#62A9AE','#D1AE59','#57976F','#B55748']):
        values = [stats[g]['平均时间分项_秒'][part] for g in ('s1','s2')]
        axes[1].bar([0,1],values,bottom=bottom,color=color,label=part,width=.52)
        bottom += values
    for i,v in enumerate(bottom):axes[1].text(i,v+75,f'{v:.2f}',ha='center')
    axes[1].set(xticks=[0,1],xticklabels=['方案一','方案二'],ylabel='平均每局虚拟时间（秒）',title='检测与切换费用抵消了方案二的路程节省',ylim=(0,bottom.max()*1.2))
    axes[1].legend(loc='upper left',fontsize=8,ncol=3)
    fig.suptitle('两组演练：每目标耗时分布与完整费用构成',fontsize=15)
    fig.text(.5,.015,note,ha='center',fontsize=8,color='#586A73')
    fig.tight_layout(rect=(0,.07,1,.92)); save(fig,FIGURES[0])

    fig,axes = plt.subplots(1,2,figsize=(14,7))
    strata = comparison['共同目标数分层']
    for i,group in enumerate(('s1','s2')):
        values = [r[group]['平均每目标时间_秒'] for r in strata]
        x = np.arange(len(strata))+(i-.5)*.36
        axes[0].bar(x,values,width=.34,color=colors[group],label='方案一' if i==0 else '方案二')
        for xx,value,r in zip(x,values,strata):axes[0].text(xx,value+5,f'{value:.1f}\nn={r[group]["样本数"]}',ha='center',fontsize=8)
    axes[0].set_xticks(range(len(strata)),[f'{r["目标数"]}个目标' for r in strata])
    axes[0].set(ylabel='分组平均每目标虚拟时间（秒）',title='按相同目标数量分组后仍倾向方案一\n同目标数不等于同位置、同难度',ylim=(0,570))
    axes[0].legend(loc='upper left',fontsize=9)
    y = np.arange(len(rows))
    axes[1].barh(y,[r['总虚拟时间_秒'] for r in rows],color=[colors[r['组']] for r in rows])
    axes[1].set_yticks(y,[f'{r["显示编号"]}（{r["官方结果文件目标总数"]}目标）' for r in rows],fontsize=8)
    axes[1].invert_yaxis()
    for i,r in enumerate(rows):axes[1].text(r['总虚拟时间_秒']+35,i,f'{r["总虚拟时间_秒"]:.1f}',va='center',fontsize=8)
    axes[1].set(xlabel='总虚拟时间（秒）',title='全部12局公开；组内按结束时间编号',xlim=(0,max(r['总虚拟时间_秒'] for r in rows)*1.14))
    fig.suptitle('两组演练：共同目标数分层与全部逐局成绩',fontsize=15)
    fig.text(.5,.015,note,ha='center',fontsize=8,color='#586A73')
    fig.tight_layout(rect=(0,.07,1,.92)); save(fig,FIGURES[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT)
    args = parser.parse_args()
    rows,sources = gather(args.base.resolve())
    stats,comparison = statistics(rows)
    figures(rows,stats,comparison)
    for path in [Path(__file__),ROOT/'scripts/analyze_q3_practice_log.py',ROOT/'scripts/build_q3_scheme1_assets.py']:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    report = {'说明':'仅评估用户指定s1/s2两组；官方结果JSON用于核验总目标数，行为费用来自时间吻合的本地HTTP日志；未解密jlog/psum或验证签名',
        '统计':stats,'比较口径':comparison,'逐局记录':rows,'来源SHA256':sources,'图表':FIGURES,
        '结论':'当前数据优先支持方案一；未配对且每组仅6局，控制目标数量后优势缩小，不宣称总体显著或普遍优越'}
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'统计':stats,'比较口径':comparison},ensure_ascii=False,indent=2))
    print(f'评估通过：{len(rows)}局、{len(rows)*3}份测试文件；目标总数、包哈希、组别、时间关联及逐动作费用已核验。')


if __name__ == '__main__': main()
