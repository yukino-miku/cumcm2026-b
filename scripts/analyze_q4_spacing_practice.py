#!/usr/bin/env python
"""分析新建文件夹七局既有演练，核查100米间距及遗漏/网格清除；不调用模拟器。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image
from q4_archive_sources import matches_source, verify_manifest
from analyze_q4_practice_4 import (matching_logs, public_header, audit_actions, audit_trace,
                                  read, write, digest, relative, is_fixed, RESULT_KEYS, COLORS)
from build_q4_inner13_assets import setup_font
from cumcm2026_b.q4_strategy import DirectionalRegion, search_stations
from cumcm2026_b.q3_geometry import max_distance, minimum_circle, sweep_path

SOURCE=ROOT/'materials/original/simulator/CUMCM2026B/Jammers-simulator-win64/Jammers-simulator/JammersSimulatorData/behavior-logs/新建文件夹'
MODELS=ROOT/'results/models/第四问/实测100米间距'
FIG=ROOT/'results/figures/第四问/实测100米间距'
TABLE=ROOT/'results/tables/第四问/实测100米间距_评估.json'
REPORT=ROOT/'docs/第四问/实测100米间距_新建文件夹分析.md'
PAPER=ROOT/'paper/sections/第四问_100米间距六局演练_备用说明.md'
MANIFEST=ROOT/'results/tables/第四问/实测100米间距_来源校验.json'
OLD_MODELS=ROOT/'results/models/第四问/实测4目录'


def analyze(run):
    """从脱敏动作独立复算费用、扫描后尾程与频道过程；未知真值不参与。"""
    r = run['结果']
    events = r['动作记录']
    actions = [e for e in events if e['动作'] in ('检测', '清除')]
    fixed_end = max(e['步骤'] for e in actions if is_fixed(e))
    markers = [e for e in events if e['动作'] == '固定扫描完成']
    if markers:
        fixed_end = markers[0]['步骤']
    position, channel = [0., 0.], 1
    distance = virtual = 0.
    parts = Counter({'移动': 0., '检测': 0., '切换': 0., '清除成功': 0., '清除失败': 0.})
    cleared = set()
    measurements = switches = failures = certificates = 0
    groups = defaultdict(lambda: Counter({'路程_米': 0., '时间_秒': 0., '清除数': 0}))
    scan_counts, feedback = Counter(), Counter()
    channels = defaultdict(lambda: {'首次发现': None, '追加测向次数': 0, '追加无信号次数': 0, '成功清除': None})
    regions = {c: DirectionalRegion() for c in range(1, 21)}  # 本批七局实际都执行Q4策略；题号仍按官方文件分类。
    legs = []
    for e in actions:
        c, p = e['频道'], e['位置']
        travel = math.dist(position, p)
        oldtime = virtual
        distance += travel
        virtual += travel / 5
        parts['移动'] += travel / 5
        tail = e['步骤'] > fixed_end
        kind = '扫描后收尾' if tail else '固定站移动' if is_fixed(e) else '扫描中清除移动' if e['动作'] == '清除' else '追加测向移动'
        groups[kind]['路程_米'] += travel
        if e['动作'] == '检测':
            assert c not in cleared
            measurements += 1
            sw = int(c != channel)
            switches += sw
            channel = c
            virtual += 5 + sw
            parts['检测'] += 5
            parts['切换'] += sw
            feedback[e['结果']] += 1
            purpose = '固定站检测' if is_fixed(e) else e['用途']
            scan_counts[purpose] += 1
            if e['用途'] == '顺路检测':
                assert travel < 1e-6
            if e['结果'] in ('direction', 'near') and channels[c]['首次发现'] is None:
                channels[c]['首次发现'] = {k: e[k] for k in ['步骤', '用途', '位置', '虚拟时间_秒']}
            if e['用途'] == '追加测向':
                channels[c]['追加测向次数'] += 1
                channels[c]['追加无信号次数'] += int(e['结果'] == 'no_signal')
            if regions:
                regions[c].observe(p, e['结果'], e.get('示向度'))
                if '外包顶点' in e:
                    assert np.allclose(regions[c].vertices, e['外包顶点'], rtol=0, atol=1e-7)
        else:
            assert c not in cleared
            success = e['结果'] == 'success'
            virtual += 5 if success else 3
            parts['清除成功' if success else '清除失败'] += 5 if success else 3
            if regions and e['具有覆盖证书']:
                farthest = max_distance(regions[c].vertices, p)
                assert abs(farthest - e['最远可能目标距离_米']) < 1e-5 and farthest < 20
                certificates += 1
            if success:
                cleared.add(c)
                groups[kind]['清除数'] += 1
                channels[c]['成功清除'] = {'步骤': e['步骤'], '位置': p, '虚拟时间_秒': e['虚拟时间_秒'], '扫描后': tail}
            else:
                failures += 1
                if regions:
                    regions[c].failed_clear(p)
        groups[kind]['时间_秒'] += virtual - oldtime
        assert abs(virtual-e['虚拟时间_秒']) < .001
        if travel > 1e-6:
            legs.append({'步骤': e['步骤'], '起点': position, '终点': p, '路程_米': travel, '类别': kind, '频道': c, '用途': e['用途'], '结果': e['结果']})
        position = p
    assert cleared == set(r['已清除频道']) and len(cleared) == r['清除数']
    assert abs(distance-r['总路程_米']) < 1e-5 and abs(virtual-r['虚拟总时间_秒']) < .001
    assert (measurements, switches, failures) == (r['检测次数'], r['频道切换次数'], r['清除失败次数'])
    for name, value in parts.items():
        assert abs(value - r['时间分项_秒'][name]) < 1e-5
    assert r['正常退出'] and r['异常'] is None
    if regions:
        assert r['流程正常完成'] and r['已完成搜索站'] == list(range(13))
        assert r['末尾增站评估']['自动新增未知源搜索站数'] == 0
        assert r['全部完成证据'] == (len(cleared) == 16)
        assert not r['已发现未清除频道'] and not r['已证明不存在频道']
        for state in r['频道记录']:
            assert state['不存在证书'] is None
            assert all(x['来源'] == '清除失败' and x['半径'] == 20 for x in state['排除约束'])
    target = run['官方结果']['jammer_count']
    assert len(cleared) <= target
    n_tail = groups['扫描后收尾']['清除数']
    tail_channels = [{'频道': c, **x} for c, x in sorted(channels.items()) if x['成功清除'] and x['成功清除']['扫描后']]
    latest = {}
    for event in events:
        if event['步骤'] <= fixed_end and '外包顶点' in event:
            latest[event['频道']] = event['外包顶点']
    for state in tail_channels:
        # 光学失败只增加排除圆，不改变凸外包顶点；该半径仍是忽略排除圆后的安全上界。
        state['扫描结束时外包覆盖圆半径_米'] = minimum_circle(latest[state['频道']])[1]
        state['扫描结束时已可安全清除'] = state['扫描结束时外包覆盖圆半径_米'] <= 19.5
    return {'目标数': target, '清除数': len(cleared), '实际全清': len(cleared) == target,
            '路程_米': distance, '时间_秒': virtual, '时间分项_秒': dict(parts), '固定扫描终点步骤': fixed_end,
            '阶段及移动用途': {k: dict(v) for k, v in groups.items()}, '检测用途次数': dict(scan_counts),
            '测量反馈': dict(feedback), '扫描中清除数': len(cleared)-n_tail, '扫描后清除数': n_tail,
            '补测无信号次数': sum(x['追加无信号次数'] for x in channels.values()),
            '首次发现用途': dict(Counter('固定站检测' if x['首次发现']['用途'].startswith(('固定搜索站', '搜索站')) else x['首次发现']['用途'] for x in channels.values() if x['首次发现'])),
            '顺路首次发现频道': [c for c, x in channels.items() if x['首次发现'] and x['首次发现']['用途'] == '顺路检测'],
            '重算通过的清除证书数': certificates, '尾程逐频道': tail_channels,
            '扫描结束时已定位待清除数': sum(x['扫描结束时已可安全清除'] for x in tail_channels),
            '逐频道': [{'频道': c, **x} for c, x in sorted(channels.items())], '移动线段': legs}


def extra_analysis(r):
    events=r['动作记录'];actions=[e for e in events if e['动作'] in ('检测','清除')]
    plans=[e for e in events if e['动作']=='规划']
    previous=[0.,0.];last_channel={};spacing=[]
    success={e['频道']:e for e in actions if e['动作']=='清除' and e['结果']=='success'}
    no_signal=[]
    for e in actions:
        c=e['频道'];point=e['位置']
        if e.get('用途')=='追加测向':
            plan=next(x['方案'] for x in reversed(plans) if x['步骤']<e['步骤'] and x['频道']==c)
            real=math.dist(previous,point);same=math.dist(last_channel[c],point)
            assert abs(real-plan['距当前机器人_米'])<1e-6
            assert abs(same-plan['距该频道上次观测_米'])<1e-6
            if not plan['间距约束已放宽']:assert min(real,same)>=100-1e-6
            spacing.append({'步骤':e['步骤'],'频道':c,'距机器人上一动作位置_米':real,
                            '距该频道上次检测_米':same,'放宽间距':plan['间距约束已放宽']})
            if e['结果']=='no_signal':
                bound=math.dist(point,success[c]['位置'])+20
                no_signal.append({'步骤':e['步骤'],'频道':c,'位置':point,
                                  '距真实源上界_米':bound,'排除超距并归因背侧':bound<1000-1e-6})
        if e['动作']=='检测':last_channel[c]=point
        previous=point
    grids=[]
    for e in events:
        if e['动作']!='有限覆盖':continue
        c=e['频道'];start=next(x['位置'] for x in reversed(actions) if x['步骤']<e['步骤'])
        vertices=next(x['外包顶点'] for x in reversed(events) if x['步骤']<e['步骤'] and x.get('频道')==c and '外包顶点' in x)
        path=sweep_path(np.asarray(vertices),np.asarray(start))
        assert len(path)==e['覆盖点数']
        moves=[x for x in actions if x['步骤']>e['步骤'] and x['频道']==c and x['用途']=='有限方格覆盖']
        assert np.allclose(path[:len(moves)],[x['位置'] for x in moves],rtol=0,atol=1e-6)
        assert moves[-1]['结果']=='success' and all(x['结果']!='success' for x in moves[:-1])
        prev=start;distance=0.
        for x in moves:distance+=math.dist(prev,x['位置']);prev=x['位置']
        direct=3*(len(moves)-1)+5
        duration=moves[-1]['虚拟时间_秒']-e['虚拟时间_秒']
        assert abs(distance/5+direct-duration)<.001
        grids.append({'频道':c,'起始步骤':e['步骤'],'计划点数':len(path),'实际尝试次数':len(moves),
                      '未命中次数':len(moves)-1,'移动_米':distance,'移动时间_秒':distance/5,
                      '清除动作时间_秒':direct,'总时间_秒':duration,'开始位置':start,
                      '计划位置':path.tolist(),'实际位置':[x['位置'] for x in moves]})
    return {'补测间距逐次核验':spacing,'补测间距放宽次数':sum(s['放宽间距'] for s in spacing),
            '主补测无信号逐次距离上界':no_signal,'主补测无信号可排除超距数':sum(s['排除超距并归因背侧'] for s in no_signal),
            '有限覆盖核验':grids,'延后原因':dict(Counter(e['原因'] for e in events if e['动作']=='延后目标')),
            '未发现频道':r['未发现频道'],'漏源是否均未发现':not r['已发现未清除频道']}


def collect():
    index=matching_logs();runs=[];counts=Counter()
    files=sorted(SOURCE.glob('*.result.json'),key=lambda p:read(p)['ended_at_utc'])
    assert len(files)==7 and len(list(SOURCE.iterdir()))==21
    for file in files:
        official=read(file);stem=file.name.removesuffix('.result.json')
        jlog,psum=SOURCE/(stem+'.jlog'),SOURCE/(stem+'.psum')
        header,public=public_header(jlog);raw=psum.read_bytes()
        assert raw[:8]==b'JMBPSUM1' and int.from_bytes(raw[8:10],'big')==1
        length=int.from_bytes(raw[10:14],'big');assert 0<length<=len(raw)-14
        ph=json.loads(raw[14:14+length])
        assert header['package_type']=='practice_behavior_log' and ph['package_type']=='practice_summary'
        assert header['formal_index'] is None and ph['formal_index'] is None
        assert header['team_no']==ph['team_no']
        for k in ['problem_no','practice_run_no','case_code']:assert official[k]==header[k]==ph[k]
        assert official['package_sha256']==digest(jlog)
        assert official['jammer_count']==official['omnidirectional_jammer_count']+official['directional_jammer_count']
        timestamp=datetime.fromisoformat(header['created_at_utc'].replace('Z','+00:00')).timestamp()*1000
        # 不按本地文件夹猜题号：本批官方第三问也由第四问脚本执行并保存在第四问目录。
        matches=[x for x in index if x[1]==header['team_no'] and abs(timestamp-x[2])<=2000]
        assert len(matches)==1,(stem,len(matches))
        folder_problem,_,exited,folder,items=matches[0]
        r=read(folder/'运行结果.json');audit=audit_actions(items,r);actions=audit_trace(items,r)
        responses={x['request_id']:x['response'] for x in items if x.get('类型')=='响应'}
        requests={x['payload']['request_id']:x for x in items if x.get('类型')=='请求' and x['path'] in ('/measure','/clear')}
        for (rid,request),event in zip(requests.items(),actions):
            if request['path']=='/measure':assert event['示向度']==responses[rid].get('svd_deg')
        meta=r['运行来源']
        assert meta['Git提交']=='d4dfa86597d8e66a92c6e4fbb5da46464f0c6464' and meta['Git工作区干净']
        for name,value in meta['源码_SHA256'].items():assert matches_source(ROOT,name,value),name
        assert r['配置']==read(ROOT/'configs/q4_geometric_spacing.json')
        assert folder_problem==4 and np.allclose(r['固定站坐标'],search_stations(),rtol=0,atol=1e-8)
        number=official['problem_no'];counts[number]+=1
        label=f'N4-{counts[number]}' if number==4 else f'混入P3-{counts[number]}'
        record={'编号':label,'题号':number,'执行算法':'第四问13站几何补测100米间距','官方结果':official,
                '公开头摘要':public,'关联':{'本地目录':relative(folder),'同队号':True,
                '生成减退出响应_毫秒':timestamp-exited,'目录标注题号':folder_problem,
                '题号与脚本是否一致':number==4,'依据':'同队号、唯一退出时间；未解密正文或验证包签名'},
                '协议核验':audit,'固定站坐标':r['固定站坐标'],
                '结果':{k:r[k] for k in RESULT_KEYS if k in r},
                '来源SHA256':{relative(p):digest(p) for p in [file,jlog,psum,folder/'运行结果.json',folder/'请求响应日志.jsonl']}}
        record['诊断']=analyze(record);record['间距与覆盖诊断']=extra_analysis(r)
        write(MODELS/(label+'.json'),record);runs.append(record)
    assert counts=={4:6,3:1}
    return runs


def summarize(runs):
    ds=[r['诊断'] for r in runs];n=len(ds)
    return {'局数':n,'目标总数':sum(d['目标数'] for d in ds),'清除总数':sum(d['清除数'] for d in ds),
            '全清局数':sum(d['实际全清'] for d in ds),
            '平均虚拟时间_秒':sum(d['时间_秒'] for d in ds)/n,'平均路程_米':sum(d['路程_米'] for d in ds)/n,
            '平均尾程_米':sum(d['阶段及移动用途'].get('扫描后收尾',{}).get('路程_米',0) for d in ds)/n,
            '主补测次数':sum(d['检测用途次数'].get('追加测向',0) for d in ds),
            '主补测无信号次数':sum(d['补测无信号次数'] for d in ds),
            '顺路首次发现数':sum(len(d['顺路首次发现频道']) for d in ds),
            '清除失败次数':sum(r['结果']['清除失败次数'] for r in runs),
            '扫描后清除数':sum(d['扫描后清除数'] for d in ds),
            '时间分项合计_秒':{k:sum(d['时间分项_秒'][k] for d in ds) for k in ds[0]['时间分项_秒']}}


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ['.png','.svg']:
        p=FIG/(name+ext);fig.savefig(p,dpi=160,metadata={'Date':None} if ext=='.svg' else None)
        if ext=='.svg':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    plt.close(fig)


def draw(ax,run,detailed=False):
    d=run['诊断'];ax.add_patch(Circle((0,0),1800,fill=False,ls='--',color='#836096',lw=1.3))
    for leg in d['移动线段']:
        a,b=np.asarray(leg['起点']),np.asarray(leg['终点']);color=COLORS[leg['类别']]
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.1,alpha=.85)
        if leg['路程_米']>90:ax.annotate('',a+.61*(b-a),a+.44*(b-a),arrowprops=dict(arrowstyle='-|>',color=color,lw=.7))
    actions=[e for e in run['结果']['动作记录'] if e['动作'] in ('检测','清除')]
    for e in actions:
        p=e['位置']
        if e['动作']=='清除':
            if e['结果']=='success':
                ax.scatter(*p,marker='o',s=32,facecolors='white' if e['步骤']>d['固定扫描终点步骤'] else '#229E86',edgecolors='#229E86',zorder=6)
                if detailed:
                    close_station=min(math.dist(p,s) for s in run['固定站坐标'])<180
                    ax.annotate(str(e['频道']),p,xytext=(-18,-16) if close_station else (5,5),
                                textcoords='offset points',fontsize=8,color='#206F5F')
            else:ax.scatter(*p,marker='x',s=12,color='#b95050',alpha=.65,zorder=4)
        elif e['用途']=='追加测向':ax.scatter(*p,marker='^',s=24,c='#D88932',edgecolors='white',lw=.4,zorder=5)
    for i,p in enumerate(run['固定站坐标'],1):
        ax.scatter(*p,marker='s',s=24,c='#283E50',zorder=4)
        if detailed:ax.annotate(f'站{i}',p,xytext=(-6,-14),textcoords='offset points',fontsize=8)
    ax.scatter(0,0,marker='*',s=100,c='#172F42',edgecolors='white',zorder=8)
    ax.scatter(*actions[-1]['位置'],marker='D',s=42,c='#9B5B9D',edgecolors='white',zorder=8)
    bound=max(2000,max(abs(v) for e in actions for v in e['位置'])+180)
    ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal',xlabel='东向 x（米）',ylabel='北向 y（米）')
    ax.grid(alpha=.15)
    ax.set_title(f"{run['编号']} · 清除{d['清除数']}/{d['目标数']} · 扫描后清除{d['扫描后清除数']}个\n{d['路程_米']/1000:.2f}公里｜{d['时间_秒']:.1f}秒｜未命中{run['结果']['清除失败次数']}次",fontsize=10 if not detailed else 13)


def handles():
    h=[Line2D([],[],color=c,lw=2,label=k) for k,c in COLORS.items()]
    h += [Line2D([],[],color=c,marker=m,ls='none',label=k) for k,c,m in
          [('固定站','#283E50','s'),('主补测','#D88932','^'),('清除成功','#229E86','o'),('清除未命中','#b95050','x'),('终点','#9B5B9D','D')]]
    return h


def figures(runs):
    setup_font();plt.rcParams['svg.hashsalt']='q4-spacing-practice'
    q4=[r for r in runs if r['题号']==4]
    fig,axes=plt.subplots(2,3,figsize=(16,11.6))
    for ax,r in zip(axes.flat,q4):draw(ax,r)
    fig.suptitle('100米间距六局第四问实际演练：4局全清、2局各漏1源',fontsize=18)
    fig.legend(handles=handles(),loc='lower center',bbox_to_anchor=(.5,.04),ncol=5,fontsize=10)
    fig.text(.5,.014,'紫线为固定扫描后的收尾；清除点是机器狗位置，未知源的精确位置没有绘入。',ha='center')
    fig.subplots_adjust(left=.065,right=.97,bottom=.16,top=.91,wspace=.28,hspace=.4);save(fig,'六局第四问轨迹总览')
    for r in runs:
        fig,ax=plt.subplots(figsize=(9,10.4));draw(ax,r,True)
        fig.suptitle(r['官方结果']['case_code']+('（官方第三问，实际执行第四问脚本）' if r['题号']==3 else ''),fontsize=11,y=.96)
        fig.legend(handles=handles(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,fontsize=9)
        fig.text(.5,.012,'方块是固定站；圆点旁数字是清除频道；轨迹按实际顺序连接，不人为闭合。',ha='center',fontsize=9)
        fig.subplots_adjust(left=.12,right=.96,bottom=.23,top=.88);save(fig,r['编号']+'_轨迹')
    fig,axes=plt.subplots(1,2,figsize=(14,6.3));x=np.arange(6);bottom=np.zeros(6)
    for key,color in [('移动','#438CB0'),('检测','#56B8B3'),('切换','#DBAC51'),('清除成功','#229E86'),('清除失败','#b95050')]:
        values=np.array([r['诊断']['时间分项_秒'][key] for r in q4]);axes[0].bar(x,values,bottom=bottom,label=key,color=color);bottom+=values
    axes[0].bar_label(axes[0].containers[-1],labels=[f'{v:.0f}' for v in bottom],padding=4)
    axes[0].set(ylabel='虚拟时间（秒）',title='完整费用：包含98次光学清除未命中',ylim=(0,8000))
    axes[0].legend(ncol=3,fontsize=9)
    axes[1].bar(x,[r['诊断']['路程_米']/1000 for r in q4],color='#9bc1d1',label='总路程')
    axes[1].bar(x,[r['诊断']['阶段及移动用途']['扫描后收尾']['路程_米']/1000 for r in q4],color='#9B5B9D',label='其中收尾路程')
    axes[1].set(ylabel='路程（公里）',title='短收尾仍可能漏检：N4-3尾程为0但只清除10/11')
    axes[1].legend()
    for ax in axes:ax.set_xticks(x,[r['编号'] for r in q4]);ax.grid(axis='y',alpha=.12)
    fig.tight_layout();save(fig,'费用与收尾对照')
    r=q4[3];grid=next(g for g in r['间距与覆盖诊断']['有限覆盖核验'] if g['频道']==3)
    ev=r['结果']['动作记录'];positives=[e for e in ev if e.get('频道')==3 and e.get('结果')=='direction']
    origin=np.asarray(positives[0]['位置']);probes=np.array([e['位置'] for e in ev if e.get('频道')==3 and e.get('用途')=='追加测向'])
    plan=np.array(grid['计划位置']);actual=np.array(grid['实际位置'])
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    axes[0].plot(probes[:,0],probes[:,1],'--^',color='#D88932',label='补测站访问次序（省略中途动作）')
    axes[0].scatter(*origin,marker='s',s=65,c='#283E50')
    for i,p in enumerate(probes,1):axes[0].annotate(str(i),p,xytext=(4,5),textcoords='offset points')
    theta=math.radians(positives[0]['示向度']);end=origin+1500*np.array([math.cos(theta),math.sin(theta)])
    axes[0].plot([origin[0],end[0]],[origin[1],end[1]],'--',color='#936291',label='唯一有效观测的方向射线')
    axes[0].scatter(*actual[-1],s=65,c='#229E86',label='最终成功清除位置')
    axes[0].legend(fontsize=8,loc='lower left');axes[0].set_title('频道3：8次主补测全部无信号\n依据距离上界均可排除超距')
    axes[0].margins(x=.13)
    axes[1].scatter(plan[:,0],plan[:,1],s=9,c='#ccd3d8',label='计划228个格点')
    axes[1].plot(actual[:,0],actual[:,1],color='#9B5B9D',lw=1)
    axes[1].scatter(actual[:-1,0],actual[:-1,1],s=13,c='#b95050',marker='x',label='前91次未命中')
    axes[1].scatter(*actual[-1],s=55,c='#229E86',label='第92次成功')
    axes[1].legend(fontsize=9,loc='upper right');axes[1].set_title('网格段耗时801.6秒：移动523.6秒＋动作278秒\n为看清格点横向展宽，纵横比例不同')
    axes[0].set_aspect('equal');axes[1].set_aspect('auto')
    for ax in axes:ax.set(xlabel='东向 x（米）',ylabel='北向 y（米）');ax.grid(alpha=.15)
    fig.text(.5,.015,'成功清除坐标只约束真实源在其20米内；左图虚线表示补测顺序，完整运动路径见单局图。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,1));save(fig,'N4-4频道3_连续无信号与网格清除')


def report(runs,data):
    q4=[r for r in runs if r['题号']==4];now=data['本批第四问'];old=data['此前六局第四问']
    lines=['# 100米间距版本：新建文件夹七局演练分析','',
        '**六局第四问清除75/77个源，4局全清、2局各漏1源；不能认定当前100米间距版本已改善整体结果。** 第4局连续补测无信号及网格清除开销突出。另有一局官方第三问，实际执行了第四问13站脚本，单独统计。','',
        '## 数据与版本核验','',
        '输入目录含21个文件：7组jlog、psum和result.json。核对题号、案例码、演练号、摘要头和包SHA256，按同队号及唯一退出时间关联本地HTTP记录。没有解密日志正文或验证包签名；来源字段与逐动作证据均保存，队号、请求ID及原始HTTP内容不纳入新归档。','',
        '七局均记录提交`d4dfa86`、运行时工作区干净；实际配置均为100米补测优先间距、400米单次绕行门槛，运行源码逐文件SHA256一致。本批没有折线运行，所有场景都执行13个固定站；分析未运行模拟器或修改策略。','',
        '## 六局第四问结果（按结束时间排序）','',
        '| 编号 | 案例码 | 全向/定向 | 清除/目标 | 路程/公里 | 虚拟时间/秒 | 主补测/无信号 | 清除未命中 | 收尾/公里 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in q4:
        d=r['诊断'];o=r['官方结果'];lines.append(f"| {r['编号']} | {o['case_code']} | {o['omnidirectional_jammer_count']}/{o['directional_jammer_count']} | {d['清除数']}/{d['目标数']} | {d['路程_米']/1000:.3f} | {d['时间_秒']:.1f} | {d['检测用途次数']['追加测向']}/{d['补测无信号次数']} | {r['结果']['清除失败次数']} | {d['阶段及移动用途']['扫描后收尾']['路程_米']/1000:.3f} |")
    lines += ['', 'N4-3漏1源，N4-4漏1源。两局都已处理完全部已发现频道，说明遗漏发生在发现阶段，没有把一个已发现目标留着不清除。日志缺少隐藏源真值，不能确定遗漏频道、坐标、类型或发射轴；官方只给本局数量及类型总数，不能直接断言漏的一定是定向源。','',
        '所有运行均正常退出，但程序的严格“运行成功=false”表示没有取得16源在线全清证明，并不等于六局都实际失败。本批实际全清与否按成功清除数和官方目标总数分别判断。','',
        '![六局轨迹总览](../../results/figures/第四问/实测100米间距/六局第四问轨迹总览.png)','',
        '## 与此前六局演练比较','',
        '| 指标 | 此前原几何六局 | 本批100米六局 |','|---|---:|---:|',
        f"| 实际清除 | {old['清除总数']}/{old['目标总数']} | {now['清除总数']}/{now['目标总数']} |",
        f"| 全清局数 | {old['全清局数']}/6 | {now['全清局数']}/6 |",
        f"| 平均时间/秒 | {old['平均虚拟时间_秒']:.1f} | {now['平均虚拟时间_秒']:.1f} |",
        f"| 平均路程/公里 | {old['平均路程_米']/1000:.3f} | {now['平均路程_米']/1000:.3f} |",
        f"| 平均收尾/公里 | {old['平均尾程_米']/1000:.3f} | {now['平均尾程_米']/1000:.3f} |",
        f"| 主补测/无信号 | {old['主补测次数']}/{old['主补测无信号次数']} | {now['主补测次数']}/{now['主补测无信号次数']} |",
        f"| 顺路首次发现数 | {old['顺路首次发现数']} | {now['顺路首次发现数']} |",
        f"| 光学清除未命中 | {old['清除失败次数']} | {now['清除失败次数']} |",'',
        f"平均时间增加{now['平均虚拟时间_秒']-old['平均虚拟时间_秒']:.1f}秒（{(now['平均虚拟时间_秒']/old['平均虚拟时间_秒']-1)*100:.2f}%），平均路程增加{(now['平均路程_米']-old['平均路程_米'])/1000:.3f}公里。这两批案例码全部不同，目标数及类型比例也不同，不是同地图配对，不能据此把退化归因于100米参数，也不能宣称100米提高了发现概率。",'',
        '较短尾程不自动意味着更好：N4-3尾程为0，但漏1个源；N4-4最后只处理2个已知源，却走了8.822公里尾程。','',
        '![费用与收尾](../../results/figures/第四问/实测100米间距/费用与收尾对照.png)','',
        '## N4-4为何最慢','',
        'N4-4共6769.9秒、25.409公里，只清除15/16；23次主补测中15次无信号，光学清除96次未命中。其中频道3占91次，频道17占5次。这些未命中来自有限网格搜索，不能直接称为定位证书错误；本批所有带证书清除均成功。','',
        '- 频道3在固定站7首次收到方向，此后8次主补测全部无信号，达到8次预算；计划228个覆盖格点，实际第92次清除成功。该网格段移动2617.9米、移动时间523.6秒，清除动作278秒，总共801.6秒，占该局约11.84%。',
        '- 频道17在固定站11首次发现，8次主补测中7次无信号；最终计划16个格点，实际6次清除，49.6秒。',
        '- 两频道共96次未命中只直接花288秒；还存在到网格及格点之间的移动、前序无信号补测和跨区收尾。因此不能只减去288秒就当作修复后的耗时。','',
        '用成功清除站C给出的|真实源−C|≤20米，可得任一补测站P到源的距离上界|P−C|+20。频道3八次上界均不超过795.1米；频道17七次均不超过840.9米。由于题目接收半径至少1000米，可以排除超距，按题面物理规则将这15次无信号归因于发射背侧。100米间距拉开了位置，但并没有建立避开发射背侧的选点规则。','',
        '![关键频道轨迹](../../results/figures/第四问/实测100米间距/N4-4频道3_连续无信号与网格清除.png)','',
        '## 当前结论与下一步侧重点','',
        '100米规则确实执行了：本批六局全部93次主补测的实际位移及同频道上次站间距均通过核验，没有放宽间距。无信号仍有45次（48.39%），已知目标处理和未知源发现均存在缺口，不能靠继续机械增大间距解决。','',
        '优先分析如何用已有正负反馈筛掉明显位于发射背侧的候选点，以及何时等待后续固定站，尤其避免同一目标8次补测都无信号。另需区分“更好地处理已发现源”与“发现尚未接收的源”：后者受固定站和途中扫描位置影响，缩短尾程或拉开站距均没有完整发现保证。本轮只记录建议，没有恢复折线、自动加外围站、改变参数或重跑正式测试。','',
        '## 混入的第三问','']
    third=next(r for r in runs if r['题号']==3);d=third['诊断']
    lines += [f"`{third['官方结果']['case_code']}`的官方题号为3、10个全向源；实际日志保存在第四问目录，源码与配置均为第四问13站100米版本。结果10/10全清、{d['时间_秒']:.1f}秒、{d['路程_米']/1000:.3f}公里。它可以证明这次运行完成，但不能混作第三问方案一七站策略的成绩，也不计入上述六局第四问。",'',
        '后续演练应核对模拟器题号与运行入口是否对应。当前协议日志足以识别本次不一致，未据此推测操作者的具体原因。','',
        '## 逐局图与复核','']
    for r in runs:lines.append(f"- [{r['编号']} 高清轨迹](../../results/figures/第四问/实测100米间距/{r['编号']}_轨迹.png)")
    lines += ['', '全部图另存SVG。清除位置是机器狗位置，不是真实源精确坐标；未发现源位置未画出。','',
        '```powershell',r'.\.venv\Scripts\python.exe -X utf8 scripts/analyze_q4_spacing_practice.py --verify --source-check','```','',
        '首次采集使用`--collect`；默认命令从归档重算并重绘。原始日志及HTTP仅用于只读核验，脱敏动作、来源与统计均单独保存。','']
    REPORT.write_text('\n'.join(lines),encoding='utf8',newline='\n')
    PAPER.write_text('''# 100米间距六局实际演练：论文备用说明

采用原几何补测并加入100米优先间距的六局演练，合计清除75/77个目标，4局全清、2局各漏1源；平均5553.1秒、19.900公里。所有已发现目标均最终清除，遗漏发生在未发现阶段，隐藏目标位置及发射轴未取得。

六局主补测93次、无信号45次。最慢局N4-4耗时6769.9秒、25.409公里，其中频道3连续8次主补测无信号，转入有限格点搜索后第92次清除成功，该格点段耗时801.6秒。用成功清除位置的20米半径约束可排除该频道8次补测的超距解释，因此间距调整并未解决发射背侧补测问题。

此前六局为79/79全清、平均5181.4秒，但案例码与目标组成不同，不应将两批差异解释为100米参数的因果效果。此实测不能支持“增大间距必然提高发现概率”的结论。目录另有一局官方第三问由第四问脚本执行，不计入本组结果。

完整逐局表、费用、来源核验和中文轨迹见[本批实测分析](../../docs/第四问/实测100米间距_新建文件夹分析.md)。均为演练，不是正式测试成绩。
''',encoding='utf8',newline='\n')


def manifest(runs):
    paths=list(MODELS.glob('*.json'))+list(FIG.glob('*'))+[TABLE,REPORT,PAPER,Path(__file__),
        ROOT/'scripts/analyze_q4_practice_4.py',ROOT/'scripts/analyze_q3_practice_log.py',
        ROOT/'scripts/evaluate_q3_routed_practice.py',ROOT/'scripts/build_q4_inner13_assets.py']
    paths += [OLD_MODELS/f'P4-{i}.json' for i in range(1,7)]
    runtime={name for r in runs for name in r['结果']['运行来源']['源码_SHA256']}
    paths += [ROOT/name for name in sorted(runtime)]
    return {relative(p):digest(p) for p in sorted(set(paths))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--source-check',action='store_true');args=p.parse_args()
    if args.collect and args.verify:p.error('--collect与--verify不可并用')
    runs=collect() if args.collect else [read(p) for p in sorted(MODELS.glob('*.json'))]
    runs.sort(key=lambda r:(r['题号']!=4,r['编号']))
    assert len(runs)==7
    for r in runs:
        assert analyze(r)==r['诊断'];assert extra_analysis(r['结果'])==r['间距与覆盖诊断']
        if args.source_check:
            for path,value in r['来源SHA256'].items():assert digest(ROOT/path)==value,path
            for path,value in r['结果']['运行来源']['源码_SHA256'].items():assert matches_source(ROOT,path,value),path
    new=[r for r in runs if r['题号']==4];old=[read(OLD_MODELS/f'P4-{i}.json') for i in range(1,7)]
    assert {r['官方结果']['case_code'] for r in new}.isdisjoint(r['官方结果']['case_code'] for r in old)
    data={'数据范围':'6局第四问＋1局第三问，七局实际均运行第四问13站100米间距脚本；只读演练，不是同地图对照',
          '本批第四问':summarize(new),'此前六局第四问':summarize(old),
          '逐局摘要':[{'编号':r['编号'],'题号':r['题号'],'案例码':r['官方结果']['case_code'],
                    **{k:r['诊断'][k] for k in ['目标数','清除数','实际全清','路程_米','时间_秒','时间分项_秒','补测无信号次数','检测用途次数','扫描后清除数','阶段及移动用途']}} for r in runs]}
    if args.verify:
        assert read(TABLE)==data
        saved=read(MANIFEST);assert saved.keys()==manifest(runs).keys();verify_manifest(ROOT,saved)
        for path in FIG.glob('*'):
            if path.suffix=='.png':
                with Image.open(path) as im:im.verify()
            else:ET.parse(path)
        print('核验通过：7局协议关联、动作费用、外包/清除证书、补测间距、网格路径与图文来源。')
    else:
        write(TABLE,data);figures(runs);report(runs,data);write(MANIFEST,manifest(runs))
    print(json.dumps(data['本批第四问'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
