#!/usr/bin/env python
"""核对s31五局演练，绘制实际轨迹并诊断扇形版耗时；不运行模拟器。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from analyze_q3_practice_log import public_header, matched_run, audit_actions
from evaluate_q3_routed_practice import audit_trace, digest, write_json, BASE
from build_q3_scheme1_assets import style, save
from cumcm2026_b.q3_geometry import TargetRegion, search_stations, max_distance, minimum_circle, coverage_certificate
from cumcm2026_b.q3_sector_geometry import intersect_sector, sector_polygon, cover_polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, Polygon
from PIL import Image

DATA = ROOT/'results/tables/第三问/S31_实际轨迹与耗时分析.json'
MANIFEST = ROOT/'results/tables/第三问/S31_图表清单.json'
DOC = ROOT/'docs/第三问/S31_轨迹与耗时诊断.md'
OLD = ROOT/'results/tables/第三问/新版演练评估_s11_s22.json'
FIGDIR = ROOT/'results/figures/第三问'
COLORS = {'固定检测': '#327DA2', '追加测向': '#D18330', '清除': '#25957E', '覆盖补漏': '#9863B1'}


def historical_source(path):
    # 扇形运行入口已撤回，旧运行的源码证据匹配逐字节归档，不冒充当前入口。
    if path == 'scripts/run_q3_scheme1_sector.py':
        return ROOT/'docs/第三问/历史代码/run_q3_scheme1_sector_4e4d7f1.py.txt'
    return ROOT/path


def category(e):
    if e['动作'] == '清除': return '清除'
    if e['用途'].startswith('搜索站'): return '固定检测'
    if e['用途'] == '按需覆盖补漏': return '覆盖补漏'
    return '追加测向'


def apex_exclusion(region, sector):
    """只做离线诊断：整个扇形交集位于某次direction站的4.99米圆内。"""
    clipped = intersect_sector(region.vertices, sector)
    if not len(clipped): return None
    for ob in region.observations:
        if ob['结果'] != 'direction': continue
        distance = max_distance(clipped, ob['位置'])
        if distance < 4.99:
            return {'已有方向观测站': ob['位置'], '交集最远距该站_米': distance,
                    '依据': 'direction排除5米near圆；4.99米为保守诊断阈值',
                    '交集顶点': clipped.tolist()}
    return None


def analyze(events):
    """按事件前缀重建信息，重算费用和证书；不读取目标精确真值。"""
    regions = {c: TargetRegion() for c in range(1, 21)}
    status = {c: 'unknown' for c in regions}
    negative = {c: [] for c in regions}
    phases = {}; parts = Counter(); purposes = Counter(); purpose_fees = Counter()
    state_counts = Counter(); redundant = []; primary = []; first_plans = []; ready_scans = []
    certs = []; actions = []; sector = None; phase = '中心至东方站'; pos = [0., 0.]; channel = 1
    virtual = 0.; distance = 0.; checked = 0; all_cleared = set()
    for e in events:
        if e['动作'] == '开始扇形':
            sector = e['扇形']; phase = f'扇形{sector+1}'
        if e['动作'] == '扇形路线规划':
            findings = []
            for node in e['计划路径']:
                if node['类型'] != '目标': continue
                c = node['频道']; proof = apex_exclusion(regions[c], e['扇形'])
                if proof: findings.append({'频道': c, '预测位置': node['位置'], **proof})
            if not first_plans or e['扇形'] != first_plans[-1]['扇形']:
                first_plans.append({'步骤': e['步骤'], '扇形': e['扇形'], '扇尖可排除目标': findings,
                                    '计划路径': e['计划路径']})
        if e['动作'] == '扇形清空证书':
            assert list(e['逐频道证据']) == [str(c) for c in range(1, 21)]
            for key, proof in e['逐频道证据'].items():
                c = int(key); method = proof['方式']
                if method == 'cleared': assert status[c] == 'cleared'
                elif method == 'absent': assert status[c] == 'absent'
                elif method == '完整可能区域与扇形无交集':
                    assert not len(intersect_sector(regions[c].vertices, sector))
                elif method in ['无信号圆联合覆盖', '扇形交集被该频道无信号圆排除']:
                    v = sector_polygon(sector) if method == '无信号圆联合覆盖' else intersect_sector(regions[c].vertices, sector)
                    proof_check = cover_polygon(v, negative[c])
                    assert proof_check['covered']; checked += proof_check['checked']
                else: raise AssertionError(method)
            certs.append(sector)
            assert certs == list(range(sector+1))
            if len(certs) == 6:
                status.update({c: 'absent' for c in status if status[c] == 'unknown'})
            phase = f'前往站{sector+2}' if sector < 5 else '收尾'
        if e['动作'] not in ['检测', '清除']: continue
        c = e['频道']; previous = pos; segment = math.dist(pos, e['位置']); pos = e['位置']
        distance += segment; fee = segment/5; parts['移动'] += fee
        p = phases.setdefault(phase, {'路程_米': 0., '时间_秒': 0., '检测次数': 0, '清除次数': 0})
        p['路程_米'] += segment
        if e['动作'] == '检测':
            if status[c] == 'found' and minimum_circle(regions[c].vertices)[1] <= 19.5:
                ready_scans.append({'步骤': e['步骤'], '频道': c, '用途': e['用途']})
            kind = '固定检测' if e['用途'].startswith('搜索站') else e['用途']
            switched = int(c != channel); channel = c
            parts['检测'] += 5; parts['切换'] += switched; fee += 5+switched
            purposes[kind] += 1; purpose_fees[kind] += 5+switched
            state_counts[f"{kind}/{status[c]}/{e['结果']}"] += 1; p['检测次数'] += 1
            if e['用途'].startswith('搜索站') and int(e['用途'][3:]) >= 2:
                assert int(e['用途'][3:])-2 in certs, '先到下一站而未清空'
            if e['用途'] == '扇形追加测向':
                proof = apex_exclusion(regions[c], sector)
                if proof: primary.append({'步骤': e['步骤'], '频道': c, '扇形': sector,
                                          '位置': pos, '前一位置': previous, '进入路程_米': segment, **proof})
            if status[c] == 'found' and e['结果'] == 'no_signal':
                # 仅记录同一频道在已无信号的同一站再次取得无信号；不将一般无信号视为无用。
                if any(math.dist(pos, q) < 1e-6 for q in negative[c]): redundant.append(e['步骤'])
            regions[c].observe(pos, e['结果'], e.get('示向度'))
            if e['结果'] == 'no_signal':
                negative[c].append(pos)
                if e['频道状态'] == 'absent':
                    assert status[c] == 'unknown' and coverage_certificate(negative[c]) is not None
                    status[c] = 'absent'
            else: status[c] = 'found'
            assert status[c] == e['频道状态']
            if '外包顶点' in e:
                assert np.asarray(e['外包顶点']).shape == regions[c].vertices.shape
                assert np.max(np.abs(np.asarray(e['外包顶点'])-regions[c].vertices)) < 1e-7
        else:
            assert status[c] != 'cleared'
            if e.get('具有覆盖证书'):
                assert max_distance(regions[c].vertices, pos) <= 19.5+1e-6
            if e['结果'] == 'success':
                fee += 5; parts['清除成功'] += 5; all_cleared.add(c); status[c] = 'cleared'; p['清除次数'] += 1
            else:
                fee += 3; parts['清除失败'] += 3; regions[c].failed_clear(pos)
        virtual += fee; p['时间_秒'] += fee
        assert abs(virtual-e['虚拟时间_秒']) < .001
        actions.append({k: e[k] for k in ['步骤', '虚拟时间_秒', '动作', '用途', '频道', '位置', '结果']})
    cutoff = max(e['步骤'] for e in actions if e['用途'].startswith('搜索站'))
    previous = [0., 0.]; tail = 0.
    for e in actions:
        if e['步骤'] > cutoff: tail += math.dist(previous, e['位置'])
        previous = e['位置']
    return {'时间分项_秒': {k: parts[k] for k in ['移动', '检测', '切换', '清除成功', '清除失败']},
            '总时间_秒': virtual, '总路程_米': distance, '清除数': len(all_cleared),
            '检测用途计数': dict(purposes), '检测用途直接费用_秒': dict(purpose_fees),
            '检测前状态及反馈': dict(state_counts), '同站重复无信号步骤': redundant, '已定位待清除仍检测': ready_scans,
            '分阶段': phases, '首轮规划诊断': first_plans, '可排除扇尖仍触发的主测向': primary,
            '已核验扇形': certs, '覆盖细分核查数': checked, '实际动作': actions,
            '末次固定检测步骤': cutoff, '末段路程_米': tail}


def gather():
    files = sorted((BASE/'s31').glob('*.result.json')); assert len(files) == 5
    sources = {}; runs = []
    for path in files:
        official = json.loads(path.read_text(encoding='utf-8-sig'))
        stem = path.name.removesuffix('.result.json'); jlog = path.with_name(stem+'.jlog'); psum = path.with_name(stem+'.psum')
        h, _ = public_header(jlog); raw = psum.read_bytes()
        assert raw[:8] == b'JMBPSUM1' and int.from_bytes(raw[8:10], 'big') == 1
        n = int.from_bytes(raw[10:14], 'big'); assert 0 < n <= len(raw)-14
        ps = json.loads(raw[14:14+n])
        assert official['package_sha256'] == digest(jlog)
        assert h['package_type'] == 'practice_behavior_log' and ps['package_type'] == 'practice_summary'
        assert h['formal_index'] is None and ps['formal_index'] is None and h['team_no'] == ps['team_no']
        for k in ['problem_no', 'case_code', 'practice_run_no']: assert official[k] == h[k] == ps[k]
        assert h['problem_no'] == 3 and official['directional_jammer_count'] == 0
        assert official['jammer_count'] == official['omnidirectional_jammer_count']
        folder, items, delta = matched_run(h)
        result_path = folder/'运行结果.json'; r = json.loads(result_path.read_text(encoding='utf-8'))
        assert r['Git提交'] == '4e4d7f19a48f787b1bc877eaac5a3267cc0de25f' and r['Git工作区有改动'] is False
        assert r['配置'] == json.loads((ROOT/'configs/q3_scheme1_sector.json').read_text(encoding='utf-8'))
        assert r['正常退出'] and r['运行成功'] and r['清除数'] == official['jammer_count']
        assert all(digest(historical_source(p)) == value for p, value in r['运行源码SHA256'].items())
        audit = audit_actions(items, r); audit_trace(items, r); diagnostic = analyze(r['动作记录'])
        assert abs(diagnostic['总时间_秒']-r['虚拟总时间_秒']) < .001
        assert diagnostic['清除数'] == r['清除数']
        assert all(abs(diagnostic['时间分项_秒'][k]-v) < .001 for k, v in r['时间分项_秒'].items())
        runs.append({'案例码': official['case_code'], '结束时间UTC': official['ended_at_utc'],
                     '目标数': official['jammer_count'], '关联本地目录': folder.relative_to(ROOT).as_posix(),
                     '关联时间差_毫秒': delta, '运行Git提交': r['Git提交'], '运行源码SHA256': r['运行源码SHA256'],
                     '请求响应核验': audit, '完整脱敏事件': r['动作记录'], '诊断': diagnostic,
                     '固定站点': search_stations().tolist(), '已完成固定站': r['已完成搜索站'],
                     '扇形调度统计': r['扇形调度统计'], '程序运行时间_秒': r['程序运行时间_秒']})
        for source in [path, jlog, psum, result_path, folder/'请求响应日志.jsonl']:
            sources[source.relative_to(ROOT).as_posix()] = digest(source)
    runs.sort(key=lambda r: r['结束时间UTC'])
    assert len({r['关联本地目录'] for r in runs}) == len(runs)
    for i, run in enumerate(runs, 1): run['编号'] = f'S31-{i}'
    for p in [Path(__file__), OLD, historical_source('scripts/run_q3_scheme1_sector.py'), ROOT/'configs/q3_scheme1_sector.json', ROOT/'scripts/analyze_q3_practice_log.py',
              ROOT/'scripts/evaluate_q3_routed_practice.py', ROOT/'scripts/build_q3_scheme1_assets.py']:
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
    old = json.loads(OLD.read_text(encoding='utf-8'))
    assert not ({r['案例码'] for r in runs} & {r['案例码'] for r in old['逐局记录'] if r['组']=='s11'})
    return {'性质': '5局实际演练；未解密正文或验证数字签名；无目标精确真值、队号、票据',
            '轨迹口径': '从圆心连接实际检测和清除坐标；箭头为行进方向；清除点为机器狗位置；不强制闭合',
            '来源SHA256': sources, '逐局': runs, '历史S11统计': old['统计']['s11'],
            '历史S11逐局': [{k: r[k] for k in ['案例码','官方结果文件目标总数','总虚拟时间_秒','每目标虚拟时间_秒','时间分项_秒']}
                           for r in old['逐局记录'] if r['组']=='s11']}


def handles():
    return [Line2D([], [], color=c, lw=2, label='前往'+k+'点') for k, c in COLORS.items()] + [
        Line2D([], [], marker='s', color='#263E50', ls='', label='固定检测站'),
        Line2D([], [], marker='^', color=COLORS['追加测向'], ls='', label='追加测向点'),
        Line2D([], [], marker='P', color=COLORS['覆盖补漏'], ls='', label='实际补漏点'),
        Line2D([], [], marker='o', color=COLORS['清除'], mfc='white', ls='', label='成功清除位置'),
        Line2D([], [], marker='*', color='#1D2939', ls='', label='起点（圆心）'),
        Line2D([], [], marker='D', color='#914E86', ls='', label='终点')]


def draw(ax, run, detail=False, first_only=False):
    actions = run['诊断']['实际动作']
    if first_only:
        start = next(e['步骤'] for e in run['完整脱敏事件'] if e['动作']=='开始扇形')
        end = next(e['步骤'] for e in run['完整脱敏事件'] if e['动作']=='扇形清空证书')
        actions = [e for e in actions if start < e['步骤'] < end]
        previous = [1200., 0.]
    else: previous = [0., 0.]
    ax.add_patch(Circle((0,0),1800,fill=False,ls='--',color='#765D91',lw=1.2))
    for i in range(6):
        a=i*math.pi/3; ax.plot([0,1800*math.cos(a)],[0,1800*math.sin(a)],color='#BCC6CE',ls=':',lw=.7)
    for e in actions:
        p=e['位置']; length=math.dist(previous,p); color=COLORS[category(e)]
        if length > 1e-7:
            ax.plot([previous[0],p[0]],[previous[1],p[1]],c=color,lw=1.3,alpha=.85)
            if length > 130:
                a=np.asarray(previous); v=np.asarray(p)-a
                ax.annotate('',xy=a+.59*v,xytext=a+.5*v,arrowprops={'arrowstyle':'-|>','color':color,'lw':.8,'mutation_scale':8})
        if e['动作']=='清除':
            ax.scatter(*p,marker='o',s=38,facecolors='white',edgecolors=color,linewidths=1.3,zorder=5)
            if detail:
                neighbors=[q['位置'] for q in actions if q['动作']=='清除' and q['频道']!=e['频道'] and math.dist(p,q['位置'])<100]
                offset=(-16,6) if neighbors and p[0]<np.mean([q[0] for q in neighbors]) else (6,6)
                ax.annotate(str(e['频道']),p,xytext=offset,textcoords='offset points',fontsize=8,color='#246D5A')
        elif e['用途'] in ['追加测向','扇形追加测向']:
            ax.scatter(*p,marker='^',s=30,c=color,edgecolors='white',linewidths=.4,zorder=4)
        previous=p
    for i,p in enumerate(run['固定站点']):
        ax.scatter(*p,marker='s',s=25,facecolors='white' if first_only and i>1 else '#263E50',edgecolors='#263E50',zorder=4)
        if detail and i: ax.annotate(f'站{i}',p,xytext=(5,-13),textcoords='offset points',fontsize=8)
    for e in run['完整脱敏事件']:
        if e['动作']=='按需补漏站' and (not first_only or e['扇形']==0): ax.scatter(*e['位置'],marker='P',s=60,c=COLORS['覆盖补漏'],zorder=7)
    ax.scatter(0,0,marker='*',s=100,c='#1D2939',edgecolors='white',linewidths=.5,zorder=8)
    ax.scatter(*previous,marker='D',s=45,c='#914E86',edgecolors='white',linewidths=.5,zorder=8)
    ax.set(xlim=(-2050,2050),ylim=(-2050,2050),aspect='equal',xlabel='东向 x（米）',ylabel='北向 y（米）',
           xticks=[-2000,-1000,0,1000,2000],yticks=[-2000,-1000,0,1000,2000])
    ax.grid(color='#DEE5EB',lw=.5)
    d=run['诊断']; ax.set_title(f"{run['编号']} · {run['目标数']}个目标全清\n路程 {d['总路程_米']/1000:.2f}公里 | 时间 {d['总时间_秒']:.1f}秒",fontsize=12,pad=12)


def plot(data):
    style(); plt.rcParams['svg.hashsalt']='q3-s31-practice'; names=[]
    (FIGDIR/'S31实际演练轨迹').mkdir(exist_ok=True)
    runs=data['逐局']
    fig,axes=plt.subplots(2,3,figsize=(17,12))
    for ax,run in zip(axes.flat,runs): draw(ax,run)
    axes.flat[-1].axis('off'); axes.flat[-1].legend(handles=handles(),loc='upper left',fontsize=10,frameon=False,ncol=2,columnspacing=1,handlelength=1.4)
    axes.flat[-1].text(.02,.45,'五局均完成七站扫描、六扇形验收\n虚线圆：1800米目标区域边界\n径向虚线：六个60°扇形边界\n同地点多次扫描不增加线段',transform=axes.flat[-1].transAxes,fontsize=11,linespacing=1.7,va='top')
    fig.suptitle('S31：扇形版五局实际演练轨迹',fontsize=21,y=.985)
    fig.text(.5,.018,'按结束时间编号；箭头为实际移动方向；清除点为机器狗位置，非目标精确真值；路线没有人为闭合',ha='center',fontsize=11)
    fig.subplots_adjust(left=.06,right=.97,bottom=.09,top=.90,wspace=.28,hspace=.35)
    names.append('S31_五局轨迹总览'); save(fig,names[-1])
    for run in runs:
        fig,ax=plt.subplots(figsize=(10,11)); draw(ax,run,True)
        fig.suptitle(f"{run['编号']} 实际轨迹 · {run['案例码']}",fontsize=16,y=.98)
        fig.legend(handles=handles(),loc='lower center',bbox_to_anchor=(.5,.032),ncol=3,fontsize=9,frameon=False)
        fig.text(.5,.012,'数字为清除频道；补漏标记不包含一般智能补测点；所有线段均来自实际HTTP动作',ha='center',fontsize=9)
        fig.subplots_adjust(left=.10,right=.94,top=.87,bottom=.22)
        names.append(f"S31实际演练轨迹/{run['编号']}_实际轨迹"); save(fig,names[-1])
    fig,axes=plt.subplots(1,2,figsize=(15,6.3)); old=data['历史S11统计']; keys=['移动','检测','切换','清除成功','清除失败']
    values=[old['平均时间分项_秒']]+[{k:float(np.mean([r['诊断']['时间分项_秒'][k] for r in runs])) for k in keys}]
    bottom=np.zeros(2)
    for k,color in zip(keys,['#327DA2','#55AAB0','#DAAA51','#25957E','#BB5858']):
        v=np.array([x[k] for x in values]); axes[0].bar([0,1],v,bottom=bottom,label=k,color=color,width=.58); bottom+=v
    for i,v in enumerate(bottom): axes[0].text(i,v+60,f'{v:.1f}秒',ha='center',fontsize=12)
    axes[0].set(xticks=[0,1],xticklabels=['此前S11（6局）','当前S31（5局）'],ylabel='每局平均虚拟时间（秒）',title='实际演练费用分解（不同案例，非配对）',ylim=(0,max(bottom)*1.15)); axes[0].legend(ncol=3,fontsize=9,loc='upper left')
    kinds=['固定检测','追加测向','顺路检测','扇形追加测向','智能站复用扫描','按需覆盖补漏']
    bottom=np.zeros(5)
    for k,color in zip(kinds,['#327DA2','#C2AF8F','#A7C9C9','#D18330','#55AAB0','#9863B1']):
        v=np.array([r['诊断']['检测用途计数'].get(k,0) for r in runs]); axes[1].bar(np.arange(5),v,bottom=bottom,label=k,color=color); bottom+=v
    for i,v in enumerate(bottom):axes[1].text(i,v+3,str(int(v)),ha='center')
    axes[1].set(xticks=np.arange(5),xticklabels=[r['编号'] for r in runs],ylabel='实际检测次数',title='新增站点少，也可能扫描很多次',ylim=(0,280));axes[1].legend(ncol=2,fontsize=9,loc='upper left')
    fig.text(.5,.02,'移动按距离÷5计时；每次检测5秒，切换频道另加1秒；复用位置不等于免除检测费用',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.98)); names.append('S31_费用与检测构成');save(fig,names[-1])
    run=runs[-1]; fig,axes=plt.subplots(1,2,figsize=(14,7))
    draw(axes[0],run,True,True); axes[0].set_title('S31-5：处理第一扇形时的实际路径',fontsize=13)
    axes[0].add_patch(Wedge((0,0),1800,0,60,facecolor='#E7D690',alpha=.22))
    axes[0].annotate('频道13补测点\n却位于西南侧',(-306.02,-791.60),xytext=(-1800,-1500),fontsize=11,arrowprops={'arrowstyle':'->','color':'#9A4444'},color='#9A4444')
    # 取第一扇形开始前频道13的完整外包：画真实外包的局部，不夸大微米级交集面积。
    start=next(e['步骤'] for e in run['完整脱敏事件'] if e['动作']=='开始扇形')
    e=next(e for e in reversed(run['完整脱敏事件']) if e['步骤']<start and e.get('频道')==13 and '外包顶点' in e)
    ax=axes[1];ax.add_patch(Wedge((0,0),12,0,60,facecolor='#E7D690',alpha=.6,label='东北第一扇形'))
    ax.add_patch(Polygon(e['外包顶点'],facecolor='#5A9BBE',alpha=.55,label='频道13的西南方向外包'))
    ax.add_patch(Circle((0,0),5,fill=False,ls='--',lw=1.8,color='#A04444',label='direction已排除的5米近区'))
    ax.scatter(0,0,s=36,c='#A04444',zorder=6)
    ax.annotate('扇尖交集约微米量级\n红点仅为位置标记',xy=(0,0),xytext=(-7,7),fontsize=11,arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',xlabel='东向 x（米）',ylabel='北向 y（米）',title='圆心局部放大：微小交集落在已排除近区内');ax.grid(alpha=.25);ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('耗时诊断：扇尖虚交集触发了跨区补测',fontsize=18)
    fig.text(.5,.02,'这是对已有观测的几何复核；左图为实际路径，右图为局部信息约束；本轮没有修改策略或重新运行演练',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.94));names.append('S31_扇尖判定与跨区绕行');save(fig,names[-1])
    return names


def report(data):
    runs=data['逐局']; old=data['历史S11统计']; n=len(runs)
    avg=lambda k:float(np.mean([r['诊断'][k] for r in runs]))
    part={k:float(np.mean([r['诊断']['时间分项_秒'][k] for r in runs])) for k in old['平均时间分项_秒']}
    differences={k:part[k]-old['平均时间分项_秒'][k] for k in part}
    delta=sum(differences.values()); tn=float(np.mean([r['诊断']['总时间_秒']/r['目标数'] for r in runs]))
    newcounts=Counter(r['目标数'] for r in runs); common=sorted(set(newcounts)&{r['官方结果文件目标总数'] for r in data['历史S11逐局']})
    strata=[]
    for c in common:
        a=np.mean([r['诊断']['总时间_秒']/c for r in runs if r['目标数']==c]);b=np.mean([r['每目标虚拟时间_秒'] for r in data['历史S11逐局'] if r['官方结果文件目标总数']==c])
        strata.append((c,float(a),float(b)))
    equal_change=np.mean([x[1] for x in strata])/np.mean([x[2] for x in strata])-1
    affected=sum(len(r['诊断']['可排除扇尖仍触发的主测向']) for r in runs)
    first=sum(len(r['诊断']['首轮规划诊断'][0]['扇尖可排除目标']) for r in runs)
    measurements=sum(sum(r['诊断']['检测用途计数'].values()) for r in runs)
    reuse=sum(r['扇形调度统计']['智能站复用检测次数'] for r in runs)
    fill=sum(r['扇形调度统计']['补漏检测次数'] for r in runs)
    scan_fees=sum(sum(r['诊断']['检测用途直接费用_秒'].get(k,0) for k in ['智能站复用扫描','按需覆盖补漏']) for r in runs)
    ready_count=sum(len(r['诊断']['已定位待清除仍检测']) for r in runs)
    lines=['# S31五局实际轨迹与耗时诊断','', '状态：用户已要求撤回扇形策略，当前恢复此前路线版。本报告和图表保留作为撤回前的历史演练分析，不是继续优化或启用扇形策略的指令。分析只读取已有日志，没有启动模拟器。', '',
           '## 1. 已核实的结果','',
           f"五局共清除{sum(r['目标数'] for r in runs)}个目标，全部与官方结果文件总数一致；各完成七固定站扫描和六扇形验收。运行源码均为`4e4d7f1`、运行时工作区干净，逐文件SHA256与历史实现匹配；已撤回入口使用逐字节归档核验。没有通信重试、通信异常或清除失败。",'',
           '[五局轨迹总览](../../results/figures/第三问/S31_五局轨迹总览.png) / [SVG](../../results/figures/第三问/S31_五局轨迹总览.svg)。下面按结束时间编号，清除位置是机器狗位置，不是目标精确真值。','',
           '| 编号 | 案例码 | 目标数 | 总时间（秒） | 路程（公里） | 检测次数 | 补漏新增移动 | 轨迹PNG / SVG |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for r in runs:
        d=r['诊断'];base=f"../../results/figures/第三问/S31实际演练轨迹/{r['编号']}_实际轨迹"
        lines.append(f"| {r['编号']} | {r['案例码']} | {r['目标数']} | {d['总时间_秒']:.3f} | {d['总路程_米']/1000:.3f} | {sum(d['检测用途计数'].values())} | {r['扇形调度统计']['补漏新增移动次数']} | [PNG]({base}.png) / [SVG]({base}.svg) |")
    lines+=['','## 2. 时间究竟多在哪里','',
            f"S31平均{avg('总时间_秒'):.3f}秒，S11历史六局平均{old['平均总虚拟时间_秒']:.3f}秒，描述性增加{delta:.3f}秒（{delta/old['平均总虚拟时间_秒']:.2%}）。两批案例码没有重复，目标数均值从{old['平均官方结果文件目标总数']:.3f}变为{np.mean([r['目标数'] for r in runs]):.3f}，因此不能把全部差额归因于算法。每局T/N均值{old['平均每目标虚拟时间_秒']:.3f}→{tn:.3f}秒；共同13/14目标层等权后变化{equal_change:+.2%}，仍非相同场景比较。",'',
            '| 时间组成 | S11均值（秒） | S31均值（秒） | 差额（秒） |','|---|---:|---:|---:|']
    for k in part:lines.append(f"| {k} | {old['平均时间分项_秒'][k]:.3f} | {part[k]:.3f} | {differences[k]:+.3f} |")
    lines+=['',f"增加的检测与切换合计{differences['检测']+differences['切换']:.3f}秒，占两批平均时间差的{(differences['检测']+differences['切换'])/delta:.2%}；移动增加{differences['移动']:.3f}秒，对应平均多走{(avg('总路程_米')-old['平均总路程_米'])/1000:.3f}公里。这是账面差额分解，不是因果贡献率。",'',
            '[费用与检测构成图](../../results/figures/第三问/S31_费用与检测构成.png) / [SVG](../../results/figures/第三问/S31_费用与检测构成.svg)。','',
            '## 3. 已定位到的实现问题：扇尖导致跨区任务','',
            '收到direction意味着目标距离该次观测站大于5米；5米内会返回near。当前凸外包为了保守包含目标，保留了观测站附近的扇尖。扇形归属检查只用了该凸外包与扇形相交、以及no_signal圆排除，没有利用direction已经排除的5米近区。圆心是六个扇形的共同顶点，因此西南、西侧方向区域也可能因为圆心附近微米量级交集，被当成第一扇形的待处理目标。东方固定站等共享边界上的观测站也会发生同类问题。','',
            f"逐事件前缀复核发现：五局第一扇形首次路线规划中共有{first}个这样的频道任务；其中及后续规划共{affected}次实际扇形主测向在执行前即可用已有direction近区约束排除当前扇形责任。这是调度判定过度保守造成的额外任务，不是清除安全证书错误。不能通过直接丢弃小面积交集修复，否则可能漏掉真实边界目标。",'',
            '| 局号 | 第一扇形首轮可排除任务频道 | 全程此类主测向次数 | 第一扇形实际路程（公里） |','|---|---|---:|---:|']
    for r in runs:
        d=r['诊断'];cs='、'.join(str(x['频道']) for x in d['首轮规划诊断'][0]['扇尖可排除目标'])
        lines.append(f"| {r['编号']} | {cs} | {len(d['可排除扇尖仍触发的主测向'])} | {d['分阶段']['扇形1']['路程_米']/1000:.3f} |")
    lines+=['','S31-5的具体路径：处理东北第一扇形时，程序先在约(922,421)清除频道12，随后前往西南(-306,-792)补测频道13，再回到东北(1004,1253)补测频道5。两段路合计比首尾直接连接长3317.727米，按速度折算663.545秒。这仅是局部几何绕行量：中途还取得其他频道观测，删除该点后的完整策略必须重新计算，不能直接承诺净节省这么多时间。','',
            '[扇尖判定与跨区绕行图](../../results/figures/第三问/S31_扇尖判定与跨区绕行.png) / [SVG](../../results/figures/第三问/S31_扇尖判定与跨区绕行.svg)。','',
            '## 4. 智能站少，但扫描费用并不少','',
            f"五局共{measurements}次检测，其中智能站复用{reuse}次、按需补漏{fill}次，这两类直接检测及切换费用合计{scan_fees:.0f}秒，平均{scan_fees/n:.1f}秒/局。这不是可全部删除的费用：未知频道需要覆盖证据，部分复用方向观测有定位价值；它说明单纯减少新增站点并不等于减少检测次数。",'',
            f'S31-2没有新增补漏移动，仍进行了89次智能站复用扫描、204次总检测，时间4444.019秒。当前复用规则对未知频道检查覆盖收益，对已发现B类主要检查离旧站距离和接收可能性，没有为每个频道预测本次角度能带来多少有效收缩；本批还出现{ready_count}次在完整外包已满足19.5米覆盖判据、尚未清除时继续检测该频道。这些是下一轮需要量化筛选的候选优化，不能将所有no_signal视为无用。', '',
            f"平均最后固定检测后的路程由S11的{old['平均按阶段路程_米']['末次固定检测之后']/1000:.3f}公里降至S31的{avg('末段路程_米')/1000:.3f}公里。但部分跨区行走已提前发生在第一扇形，所以末段缩短不足以证明全程不折返。",'',
            '## 5. 撤回前记录的候选改进（不执行）','',
            '1. 先修复扇形责任：把每次direction的5米近区排除纳入判断；只有证明整个扇形交集已被合法排除才放行，继续使用完整外包作清除证书。加入圆心、共享径向边界和真实近边界目标回归。',
            '2. 再筛选扫描：保留未知频道的逐频道覆盖责任；对B类比较交会角与预计区域收缩，对已足够定位的待清除频道减少重复检测。复用与固定扫描都计算检测和切换成本。',
            '3. 在责任判定正确后重新联合安排补测点、补漏点及下一站，避免让不属于当前扇形的预测圆心牵引整条路线。先作固定场景成对构造，再与新演练区分报告；本次没有修改运行算法。','',
            '## 6. 核验与复现','',
            '逐局核对result.json、jlog包SHA256、jlog/psum公开头中的问题号、演练编号及案例码，同队号退出反馈与包生成时间相差1—2毫秒且唯一关联。未解密正文、未验证数字签名；轨迹来自唯一匹配的本地HTTP日志，逐动作与程序记录核对。每一步重算移动、检测、切换及清除计时，并重建完整观测外包、逐频道无信号站和30份扇形证书。', '',
            '[脱敏事件、轨迹、逐项诊断及来源哈希](../../results/tables/第三问/S31_实际轨迹与耗时分析.json)；[图表清单](../../results/tables/第三问/S31_图表清单.json)；[分析脚本](../../scripts/analyze_q3_s31.py)。','',
            '```powershell',"Set-Location 'D:\\mywork\\code\\cumcm2026-b'",'.\\.venv\\Scripts\\python.exe -X utf8 scripts/analyze_q3_s31.py --extract',
            '.\\.venv\\Scripts\\python.exe -X utf8 scripts/analyze_q3_s31.py',
            '.\\.venv\\Scripts\\python.exe -X utf8 scripts/analyze_q3_s31.py --verify','```','',
            '--extract需要本机五局三件套及HTTP记录；默认模式只读取已归档脱敏数据重绘；--verify独立重算保存的费用、证据和诊断并检查图表来源。均不启动模拟器。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return {'平均时间_秒':avg('总时间_秒'),'历史差额_秒':delta,'扫描切换差额_秒':differences['检测']+differences['切换'],
            '平均T除N_秒':tn,'共同层变化':float(equal_change),'可排除任务实际补测次数':affected,'首轮扇尖任务数':first}


def verify(data, with_figures=False):
    for path,value in data['来源SHA256'].items():
        if not path.startswith(('materials/','local-only/')): assert digest(ROOT/path)==value,path
    for run in data['逐局']:
        assert analyze(run['完整脱敏事件'])==run['诊断']
        assert run['诊断']['清除数']==run['目标数']
        for path,value in run['运行源码SHA256'].items(): assert digest(historical_source(path))==value,path
    if with_figures:
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
        assert manifest['数据SHA256']==digest(DATA)
        for path,value in manifest['图表SHA256'].items():
            p=ROOT/path;assert digest(p)==value,path
            if p.suffix=='.png':
                with Image.open(p) as im:im.verify()
            else:ET.parse(p)
        for target in re.findall(r'\]\(([^)]+)\)',DOC.read_text(encoding='utf-8')):
            assert (DOC.parent/target).exists(),target
    print(f"S31_VERIFY_OK：{len(data['逐局'])}局，{sum(len(r['诊断']['实际动作']) for r in data['逐局'])}个实际动作，30份扇形证书；图表核查={with_figures}")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.extract:write_json(DATA,gather())
    data=json.loads(DATA.read_text(encoding='utf-8'));verify(data,args.verify)
    if args.verify:return
    names=plot(data);summary=report(data)
    outputs={}
    for name in names:
        for suffix in ['png','svg']:
            p=FIGDIR/f'{name}.{suffix}';outputs[p.relative_to(ROOT).as_posix()]=digest(p)
    write_json(MANIFEST,{'说明':'5张单局轨迹、1张总览、1张费用图、1张根因诊断图，各PNG/SVG',
                         '数据SHA256':digest(DATA),'图表SHA256':outputs})
    verify(data,True);print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
