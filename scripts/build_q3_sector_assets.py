"""扇形版与此前路线版的成对构造实验、独立逐步验收及中文图表。"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from build_q3_route_assets import audit as audit_common
from build_q3_scheme1_assets import style, save
from cumcm2026_b.q3_sector_strategy import SchemeOneSector, SectorConfig
from cumcm2026_b.q3_sector_geometry import sector_polygon, intersect_sector, cover_polygon
from cumcm2026_b.q3_routed_strategies import SchemeOneRouted, RoutedOneConfig
from cumcm2026_b.q3_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_geometry import TargetRegion, coverage_certificate, search_stations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

ARCHIVE = ROOT/'experiments/第三问/扇形清空/完整成对记录.jsonl'
REPORT = ROOT/'results/tables/第三问/扇形版_成对验证汇总.json'
MANIFEST = ROOT/'results/tables/第三问/扇形版_图表清单.json'
FIGURES = ['扇形版_费用与逐场差值', '扇形版_均匀与边界轨迹', '扇形版_中心与聚集轨迹', '扇形版_执行流程']
NAMES = {'route': '此前路线版', 'sector': '扇形清空版'}


def digest(path): return sha256(path.read_bytes()).hexdigest()


def audit_sector(record):
    counts = audit_common(record)
    result = record['结果']
    regions = {c: TargetRegion() for c in range(1, 21)}
    negatives = {c: [] for c in regions}
    statuses = {c: 'unknown' for c in regions}
    cleared, gates = set(), set()
    targets = {t['频道']: t['位置'] for t in record['真值_仅评估器读取']['目标']}
    checks, rows = 0, 0
    cfg = result['配置']
    for e in result['动作记录']:
        channel = e.get('频道')
        if e['动作'] == '检测':
            regions[channel].observe(e['位置'], e['结果'], e.get('示向度'))
            if e['结果'] == 'no_signal': negatives[channel].append(e['位置'])
            else: statuses[channel] = 'found'
            if record['设置'] == 'sector' and e['用途'].startswith('搜索站'):
                index = int(e['用途'][3:])
                if index >= 2: assert index-2 in gates, ('缺少前扇形证书', index)
        if e['动作'] == '清除':
            if e['结果'] == 'success': cleared.add(channel); statuses[channel] = 'cleared'
            else: regions[channel].failed_clear(e['位置'])
        if e['动作'] != '扇形清空证书': continue
        sector = e['扇形']; gates.add(sector)
        assert set(map(int, e['逐频道证据'])) == set(range(1, 21))
        for c, proof in e['逐频道证据'].items():
            c = int(c); method = proof['方式']; rows += 1
            if method == 'cleared': assert c in cleared
            elif method == 'absent':
                assert coverage_certificate(negatives[c]) is not None or gates == set(range(6))
                assert c not in targets
            elif method == '全局已达16目标上界': assert len(cleared) == 16
            elif method == '完整可能区域与扇形无交集':
                assert statuses[c] == 'found'
                assert not len(intersect_sector(regions[c].vertices, sector))
            else:
                assert method in {'无信号圆联合覆盖', '扇形交集被该频道无信号圆排除'}
                vertices = sector_polygon(sector) if method == '无信号圆联合覆盖' else intersect_sector(regions[c].vertices, sector)
                independent = cover_polygon(vertices, negatives[c], cfg['coverage_radius_m'], cfg['coverage_cell_m'], cfg['coverage_max_nodes'])
                assert independent['covered'], (sector, c, method)
                checks += independent['checked']
            if c in targets and len(intersect_sector([targets[c]], sector)):
                assert c in cleared, ('扇形内有真实目标未清除', sector, c)
    if record['设置'] == 'sector':
        assert result['已验收扇形'] == sorted(gates)
        assert len(gates) == 6 or len(cleared) == 16
    return {**counts, '扇形证据行数': rows, '重新细分核查单元数': checks}


def metrics(record):
    r = record['结果']
    last_fixed = max(e['步骤'] for e in r['动作记录'] if e.get('用途', '').startswith('搜索站'))
    current, tail, fixed, late = [0., 0.], 0., 0, 0
    true_positions = {t['频道']: t['位置'] for t in record['真值_仅评估器读取']['目标']}
    for e in r['动作记录']:
        if e.get('用途', '').startswith('搜索站'): fixed = int(e['用途'][3:])
        if e['动作'] in {'检测', '清除'}:
            length = math.dist(current, e['位置']); current = e['位置']
            if e['步骤'] > last_fixed: tail += length
        if e['动作'] == '清除' and e['结果'] == 'success':
            p = true_positions[e['频道']]
            angle = math.degrees(math.atan2(p[1], p[0]))%360
            if math.hypot(*p) > 1e-6 and int(angle//60) < max(0, fixed-1): late += 1
    return {'总时间_秒': r['虚拟总时间_秒'], '总路程_米': r['总路程_米'], '检测次数': r['检测次数'],
            '末次固定检测后路程_米': tail, '前序扇形目标延后清除数': late,
            '补漏新增移动次数': r.get('扇形调度统计', {}).get('补漏新增移动次数', 0),
            '智能站复用检测次数': r.get('扇形调度统计', {}).get('智能站复用检测次数', 0),
            '规划与执行现实时间_秒': r['程序运行时间_秒']}


def summarize(records):
    groups = {}
    for group in ['开发集', '独立验证集']:
        subset = [r for r in records if r['数据组'] == group]
        groups[group] = {}
        for variant in NAMES:
            selected = [r for r in subset if r['设置'] == variant]
            groups[group][variant] = {'场数': len(selected), '全清数': sum(r['结果']['运行成功'] for r in selected),
                '平均指标': {k: float(np.mean([r['指标'][k] for r in selected])) for k in selected[0]['指标']},
                '平均费用分项_秒': {k: float(np.mean([r['结果']['时间分项_秒'][k] for r in selected])) for k in selected[0]['结果']['时间分项_秒']},
                '零补漏新增移动场数': sum(r['指标']['补漏新增移动次数'] == 0 for r in selected),
                '清除失败次数': sum(r['结果']['清除失败次数'] for r in selected)}
        a = [r for r in subset if r['设置'] == 'route']; b = [r for r in subset if r['设置'] == 'sector']
        differences = [y['指标']['总时间_秒']-x['指标']['总时间_秒'] for x, y in zip(a, b)]
        groups[group]['成对比较'] = {'扇形版更快场数': sum(d < -1e-6 for d in differences),
            '扇形版更慢场数': sum(d > 1e-6 for d in differences), '时间差_秒': differences,
            '平均时间变化比例': float(np.mean(differences)/np.mean([r['指标']['总时间_秒'] for r in a]))}
        groups[group]['分布局'] = {}
        for layout in ['uniform', 'boundary', 'cluster', 'center']:
            ra = [r for r in a if r['构造参数']['layout'] == layout]
            rb = [r for r in b if r['构造参数']['layout'] == layout]
            groups[group]['分布局'][layout] = {'场数': len(ra), '此前路线版平均时间_秒': float(np.mean([r['指标']['总时间_秒'] for r in ra])),
                '扇形版平均时间_秒': float(np.mean([r['指标']['总时间_秒'] for r in rb]))}
    return groups


def run_suite(count):
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ARCHIVE.open('w', encoding='utf-8', newline='\n') as output:
        for group, start in [('开发集', 20260911), ('独立验证集', 20264911)]:
            for i in range(count):
                case = {'seed': start+i, 'count': [10,13,16][i%3], 'layout': ['uniform','boundary','cluster','center'][i%4],
                        'radius_mode': ['min','max','mixed'][(i//3)%3], 'error_mode': ['hash','zero','plus','minus','alternating'][i%5]}
                pair = []
                for name, cls, cfg in [('route', SchemeOneRouted, RoutedOneConfig()), ('sector', SchemeOneSector, SectorConfig())]:
                    env = make_case(**case); client = RobotClient(env, 'local-robot')
                    try:
                        r = {'数据组': group, '场景序号': i+1, '设置': name, '设置说明': NAMES[name], '构造参数': case,
                             '结果': cls(client, cfg).run(), '真值_仅评估器读取': env.truth_for_evaluation()}
                        r['独立验收'] = audit_sector(r); r['指标'] = metrics(r)
                        if name == 'sector': assert r['指标']['前序扇形目标延后清除数'] == 0
                        pair.append(r); records.append(r)
                        output.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); output.flush()
                    finally: client.close()
                assert pair[0]['真值_仅评估器读取'] == pair[1]['真值_仅评估器读取']
                print(f"{group} {i+1}/{count}：路线版 {pair[0]['指标']['总时间_秒']:.1f} 秒，扇形版 {pair[1]['指标']['总时间_秒']:.1f} 秒；两场均全清", flush=True)
    return records


def draw_track(ax, record):
    events = record['结果']['动作记录']
    actual = [e for e in events if e['动作'] in {'检测', '清除'}]
    points = np.array([[0, 0]]+[e['位置'] for e in actual])
    ax.plot(points[:,0], points[:,1], color='#5A8BA6', lw=1., alpha=.75)
    for a,b in zip(points,points[1:]):
        if np.linalg.norm(b-a)>160:
            ax.annotate('', a+.6*(b-a), a+.4*(b-a), arrowprops={'arrowstyle':'-|>','color':'#5A8BA6','lw':.8})
    ax.add_patch(Circle((0,0),1800,fill=False,ls='--',color='#816899'))
    for i in range(6):
        q=1800*np.array([math.cos(i*math.pi/3),math.sin(i*math.pi/3)])
        ax.plot([0,q[0]],[0,q[1]],color='#C5CDD2',lw=.7)
    station=search_stations(); ax.scatter(station[:,0],station[:,1],marker='s',s=20,color='#2B4554',label='原七固定站位置')
    for use, marker, color, label in [('clear','o','#21937E','成功清除'),('fill','D','#D68C36','按需补漏站')]:
        selected = [e['位置'] for e in events if (e['动作']=='清除' and e['结果']=='success') if use=='clear'] if use=='clear' else [e['位置'] for e in events if e['动作']=='按需补漏站']
        if selected:
            xy=np.asarray(selected); ax.scatter(xy[:,0],xy[:,1],marker=marker,s=28,facecolors='none',edgecolors=color,label=label,zorder=4)
    ax.scatter(0,0,marker='*',color='#243D4A',s=65); ax.scatter(*points[-1],marker='X',color='#844B7A',s=35)
    ax.set(xlim=(-2000,2000),ylim=(-2000,2000),aspect='equal',xlabel='东向 x（米）',ylabel='北向 y（米）')
    ax.grid(alpha=.18)
    m=record['指标']; ax.set_title(f"{NAMES[record['设置']]}：{m['总路程_米']/1000:.2f}公里 / {m['总时间_秒']:.1f}秒",fontsize=11)


def plots(records, stats):
    style()
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    group=stats['独立验证集']; bottoms=np.zeros(2)
    labels={'移动':'移动','检测':'检测','切换':'切换','清除成功':'成功清除','清除失败':'失败清除'}
    for part,color in zip(labels,['#5B8BA6','#DB9B50','#8C73A2','#369681','#C65F57']):
        values=[group[v]['平均费用分项_秒'][part] for v in NAMES]
        axes[0].bar(list(NAMES.values()),values,bottom=bottoms,label=labels[part],color=color); bottoms+=values
    axes[0].set(ylabel='平均虚拟时间（秒）',title='独立验证集：完整费用分解'); axes[0].legend(fontsize=9)
    for i,v in enumerate(bottoms): axes[0].text(i,v+35,f'{v:.1f}',ha='center')
    delta=group['成对比较']['时间差_秒']
    axes[1].bar(np.arange(1,len(delta)+1),delta,color=['#C56558' if d>0 else '#2A9780' for d in delta])
    axes[1].axhline(0,color='#333333',lw=.8)
    axes[1].set(xlabel='同场景配对序号',ylabel='扇形版－路线版（秒）',title='正值表示扇形版更慢，负值表示更快')
    fig.suptitle('严格扇形清空的收益与成本：同场景完整对照',fontsize=16)
    fig.text(.5,.01,'本地构造实验，非官方成绩；包含全部验证场景，不按结果筛选。',ha='center')
    fig.tight_layout(rect=[0,.05,1,.93]); save(fig,FIGURES[0])
    for number,layouts in [(1,['uniform','boundary']),(2,['center','cluster'])]:
        fig,axes=plt.subplots(2,2,figsize=(12,12))
        captions=[]
        for row,layout in enumerate(layouts):
            candidates=[r for r in records if r['数据组']=='独立验证集' and r['构造参数']['layout']==layout]
            first=candidates[0]['场景序号']; pair=[r for r in candidates if r['场景序号']==first]
            captions.append(f"第{row+1}行：{ {'uniform':'均匀','boundary':'边界','center':'中心','cluster':'聚集'}[layout]}，验证场景{first}")
            for ax,r in zip(axes[row],pair): draw_track(ax,r)
        axes[0,1].legend(fontsize=8,loc='lower left')
        fig.suptitle('同场景实际动作轨迹；'+ '；'.join(captions),fontsize=13,y=.98)
        fig.text(.5,.015,'按布局取验证集首个场景；方块仅标原固定站位置，橙色菱形为实际补漏决策位置；本地构造。',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,.95]); save(fig,FIGURES[number])
    fig,ax=plt.subplots(figsize=(11,8)); ax.set(xlim=(0,10),ylim=(0,10)); ax.axis('off')
    boxes=[(5,9,'中心初扫与有界初始服务'),(5,7.5,'当前固定站检测；确定60°完整扇形'),
           (5,6,'按完整区域确定责任目标；联合排列补测与清除'),(5,4.5,'复用实际站点扫描必要频道；每次反馈后更新'),
           (5,3,'逐频道检查：目标已清除、区域在外或无信号圆覆盖'),(5,1.5,'证据齐全后推进；最后扇形完成后直接结束')]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x-3.8,y-.4),7.6,.8,boxstyle='round,pad=.08',facecolor='#E5F0F4',edgecolor='#6893A6'))
        ax.text(x,y,label,ha='center',va='center',fontsize=12)
    for (_,y,_),(_,ny,_) in zip(boxes,boxes[1:]): ax.add_patch(FancyArrowPatch((5,y-.5),(5,ny+.5),arrowstyle='-|>',mutation_scale=15,color='#527B8C'))
    ax.text(.25,3.75,'有盲区：\n按需补漏\n再更新',ha='center',fontsize=10,color='#A26C24')
    ax.annotate('',xy=(1.1,4.5),xytext=(1.1,3),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.6','color':'#A26C24'})
    ax.text(9.72,5.2,'B类未完成：\n继续补测\n或有限覆盖',ha='center',fontsize=9,color='#A26C24')
    ax.annotate('',xy=(8.9,6),xytext=(8.9,4.5),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=.6','color':'#A26C24'})
    ax.set_title('扇形版执行流程：预测用于规划，实际证据决定推进',fontsize=15)
    fig.text(.5,.015,'不预设六个新增站；保留完整区域19.5米清除证书；预算不足如实停止。',ha='center',fontsize=10)
    save(fig,FIGURES[3])


def sources():
    paths=list((ROOT/'src/cumcm2026_b').glob('*.py'))+[Path(__file__),ROOT/'scripts/build_q3_route_assets.py',
        ROOT/'scripts/build_q3_scheme1_assets.py',ROOT/'configs/q3_scheme1_sector.json',ROOT/'configs/q3_scheme1_routed.json']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=30)
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.count<4: parser.error('每组至少4个场景以覆盖四种布局')
    if args.plots_only or args.verify:
        report=json.loads(REPORT.read_text(encoding='utf-8'))
        assert report['归档SHA256']==digest(ARCHIVE)
        assert report['来源SHA256']==sources()
        records=[json.loads(line) for line in ARCHIVE.read_text(encoding='utf-8').splitlines()]
        for record in records:
            assert audit_sector(record)==record['独立验收']
            assert metrics(record)==record['指标']
        assert summarize(records)==report['统计']
    else:
        records=run_suite(args.count)
        report={'性质':'本地构造成对对照，非官方演练或正式成绩','版本':'方案一此前路线版与严格扇形清空版',
            '数据说明':'开发集含调整期间查看过的场景；独立验证集种子20264911起，参数固定后运行。每个场景两设置使用相同源与误差函数，动作不同导致反馈不同。',
            '环境':{'Python':platform.python_version(),'NumPy':np.__version__},'统计':summarize(records),
            '验收合计':{k:sum(r['独立验收'][k] for r in records) for k in records[0]['独立验收']},
            '逐场指标':[{k:r[k] for k in ['数据组','场景序号','设置','构造参数','指标']} for r in records],
            '归档SHA256':digest(ARCHIVE),'来源SHA256':sources()}
        REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    if not args.verify:
        plots(records,report['统计'])
        manifest={'说明':'图表可从完整脱敏构造记录重绘','输入SHA256':{REPORT.relative_to(ROOT).as_posix():digest(REPORT),ARCHIVE.relative_to(ROOT).as_posix():digest(ARCHIVE)},
            '图表SHA256':{f'results/figures/第三问/{name}{ext}':digest(ROOT/f'results/figures/第三问/{name}{ext}') for name in FIGURES for ext in ['.png','.svg']}}
        MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
        for path,value in {**manifest['输入SHA256'],**manifest['图表SHA256']}.items(): assert digest(ROOT/path)==value,path
    print(f"SECTOR_VERIFY_OK：{len(records)}次完整运行，全部动作、费用、扇形证据、来源与图表核查完成。",flush=True)


if __name__=='__main__': main()
