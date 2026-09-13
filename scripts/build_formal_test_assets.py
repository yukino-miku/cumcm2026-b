"""由六场正式测试的已核验记录生成中文轨迹、费用和流程图；不补造目标真值。"""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from PIL import Image

from analyze_formal_tests import ROOT, ARCHIVE, TABLES, MODELS, FIG, REPORT, PAPER, MANIFEST, SUMMARY, read, write

COLORS = {'前往固定站':'#438cb0', '前往补测点':'#d88932',
          '前往清除点':'#229e86', '固定扫描后收尾':'#9b5b9d'}


def setup():
    path = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(path)).get_name(),
                         'font.size':10, 'axes.unicode_minus':False,
                         'svg.fonttype':'path', 'svg.hashsalt':'formal-tests-20260913'})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        path = FIG / (name+suffix)
        buffer = BytesIO()
        fig.savefig(buffer, format=suffix[1:], dpi=160, metadata={'Date':None} if suffix == '.svg' else None)
        raw = buffer.getvalue()
        if suffix == '.svg':
            raw = b'\n'.join(line.rstrip() for line in raw.splitlines())+b'\n'
        if not path.exists() or path.read_bytes() != raw:
            path.write_bytes(raw)
    plt.close(fig)


def legend():
    return [Line2D([], [], c=c, label=name) for name,c in COLORS.items()] + [
        Line2D([], [], ls='', marker='s', color='#294653', label='完成检测的固定站'),
        Line2D([], [], ls='', marker='^', color=COLORS['前往补测点'], label='主动补测点'),
        Line2D([], [], ls='', marker='o', mfc='none', mec=COLORS['前往清除点'], label='成功清除时机器狗位置'),
        Line2D([], [], ls='', marker='x', color='#c44b41', label='清除未命中位置'),
        Line2D([], [], ls='', marker='*', color='#294653', label='起点'),
        Line2D([], [], ls='', marker='D', color='#8a548e', label='终点')]


def draw(ax, run, annotate=False):
    r, d = run['结果'], run['诊断']
    fixed = np.asarray(d['固定站'])
    points = [x['终点'] for x in d['移动线段']] + fixed.tolist()
    extent = max(2050., max(abs(v) for p in points for v in p)+130)
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec='#8864a2', ls='--', lw=1.6))
    for leg in d['移动线段']:
        a, b = np.asarray(leg['起点']), np.asarray(leg['终点'])
        color = COLORS[leg['类别']]
        ax.plot(*np.array([a,b]).T, c=color, lw=1.3, alpha=.85)
        if leg['距离_米'] >= 100:
            ax.annotate('', a+.62*(b-a), a+.43*(b-a),
                        arrowprops={'arrowstyle':'-|>', 'color':color, 'lw':1.0})
    for i,p in enumerate(fixed):
        done = i in r['已完成搜索站']
        ax.scatter(*p, marker='s', s=36, facecolors='#294653' if done else 'none',
                   edgecolors='#294653', alpha=1 if done else .35, zorder=4)
        if annotate:
            ax.annotate(f'站{i+1}', p, xytext=(5,6), textcoords='offset points', fontsize=8, color='#294653')
    labels = 0
    for event in r['动作记录']:
        if event.get('用途') == '追加测向':
            ax.scatter(*event['位置'], marker='^', s=30, c=COLORS['前往补测点'], ec='white', lw=.4, zorder=5)
        elif event['动作'] == '清除':
            p = event['位置']
            if event['结果'] == 'success':
                ax.scatter(*p, marker='o', s=65, facecolors='none', edgecolors=COLORS['前往清除点'], lw=1.7, zorder=6)
                if annotate:
                    ax.annotate(f'频{event["频道"]}', p, xytext=(6, 8 if labels%2 else -12),
                                textcoords='offset points', fontsize=8, color='#177360')
                    labels += 1
            else:
                ax.scatter(*p, marker='x', s=30, c='#c44b41', lw=1.2, zorder=5)
    ax.scatter(0, 0, marker='*', s=115, color='#294653', ec='white', zorder=7)
    ax.scatter(*points[len(d['移动线段'])-1], marker='D', s=48, color='#8a548e', ec='white', zorder=7)
    proof = '模型全清证据齐全' if run['完成证据']['模型全清证据'] else '全清待核实'
    ax.set_title(f'{run["编号"]} · 清除{r["清除数"]}个 · {proof}\n'
                 f'{r["总路程_米"]/1000:.2f}公里｜{r["虚拟总时间_秒"]:.1f}秒｜固定站{d["固定站完成数"]}/{len(fixed)}', fontsize=13)
    ax.set(aspect='equal', xlim=(-extent,extent), ylim=(-extent,extent), xlabel='东向 x（米）', ylabel='北向 y（米）')
    ax.grid(alpha=.15)


def trajectories(runs):
    for i,run in enumerate(runs,1):
        fig,ax = plt.subplots(figsize=(9.4, 10.3))
        draw(ax, run, annotate=True)
        fig.legend(handles=legend(), ncol=3, loc='lower center', bbox_to_anchor=(.5,.045), fontsize=9)
        fig.text(.5,.02,'虚线圈为1800米源区域；圆点为成功清除时机器人位置。未解密正式包，不绘制未知源真值。',ha='center',fontsize=9)
        fig.subplots_adjust(left=.12,right=.96,top=.89,bottom=.21)
        save(fig, f'{i:02d}_{run["编号"]}_正式测试轨迹')
    for number in (3,4):
        fig,axs = plt.subplots(1,3,figsize=(18,7.4))
        for ax,run in zip(axs,[r for r in runs if r['题号']==number]):
            draw(ax,run)
        fig.suptitle(f'第{number}问 · 三场正式测试实际轨迹',fontsize=18,y=.97)
        fig.legend(handles=legend(),ncol=5,loc='lower center',bbox_to_anchor=(.5,.045),fontsize=10)
        fig.text(.5,.018,'轨迹来自唯一匹配的逐条HTTP指令；正式文件没有公开目标总数。空心固定站表示未执行。',ha='center',fontsize=10)
        fig.subplots_adjust(left=.045,right=.99,top=.84,bottom=.21,wspace=.20)
        save(fig,f'{number+4:02d}_第{number}问三场轨迹总览')


def costs(runs):
    names = [r['编号'] for r in runs]
    x = np.arange(len(runs))
    fig,axs = plt.subplots(2,1,figsize=(12,8.5),sharex=True)
    bottom = np.zeros(len(runs))
    for name,color in [('移动','#438cb0'),('检测','#6bb8bc'),('切换','#d6b153'),('清除成功','#229e86'),('清除失败','#c44b41')]:
        values = np.array([r['结果']['时间分项_秒'][name] for r in runs])
        axs[0].bar(x,values,bottom=bottom,color=color,label=name,width=.6)
        bottom += values
    for i,(t,run) in enumerate(zip(bottom,runs)):
        axs[0].text(i,t+60,f'{t:.1f}秒\n清除{run["结果"]["清除数"]}个',ha='center',fontsize=9)
    axs[0].set(ylabel='虚拟时间（秒）',ylim=(0,bottom.max()*1.18))
    whole = np.array([r['结果']['总路程_米']/1000 for r in runs])
    tail = np.array([r['诊断']['尾程路程_米']/1000 for r in runs])
    axs[1].bar(x,whole-tail,color='#438cb0',label='末次固定检测及之前',width=.6)
    axs[1].bar(x,tail,bottom=whole-tail,color='#9b5b9d',label='固定扫描后尾程',width=.6)
    for i,(total,last) in enumerate(zip(whole,tail)):
        axs[1].text(i,total+.25,f'{total:.2f}公里\n尾程{last:.2f}',ha='center',fontsize=9)
    axs[1].set(ylabel='实际路程（公里）',ylim=(0,whole.max()*1.2),xticks=x,xticklabels=names)
    axs[0].legend(ncol=5,loc='upper left',fontsize=9)
    axs[1].legend(ncol=2,loc='upper left',fontsize=9)
    for ax in axs:
        ax.axvline(2.5,c='#666666',ls=':',lw=1)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('六场正式测试：费用分解与尾程',fontsize=17)
    fig.text(.5,.025,'第三问与第四问任务及地图不同，时间差不能直接归因于算法优劣；清除数不是官方总目标数。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.10,hspace=.23)
    save(fig,'09_正式测试费用与尾程')


def timeline(runs):
    fig,axs = plt.subplots(2,3,figsize=(15,8.4))
    for ax,run in zip(axs.flat,runs):
        r,d = run['结果'],run['诊断']
        discovered,cleared = set(),set()
        times,found,done = [0.],[0],[0]
        for event in r['动作记录']:
            if event['动作']=='检测' and event['结果']!='no_signal':
                discovered.add(event['频道'])
            elif event['动作']=='清除' and event['结果']=='success':
                cleared.add(event['频道'])
            else:
                continue
            times.append(event['虚拟时间_秒']);found.append(len(discovered));done.append(len(cleared))
        times.append(r['虚拟总时间_秒']);found.append(len(discovered));done.append(len(cleared))
        ax.step(times,found,where='post',c='#d88932',label='累计已发现')
        ax.step(times,done,where='post',c='#229e86',label='累计已清除')
        ax.axvline(d['固定扫描截止时间_秒'],c='#9b5b9d',ls='--',label='固定扫描截止')
        ax.set(title=run['编号'],xlabel='虚拟时间（秒）',ylabel='频道数',ylim=(0,17),yticks=[0,4,8,12,16])
        ax.grid(alpha=.15)
    fig.suptitle('发现与清除进度：何时发现、何时处理完毕',fontsize=17)
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.045))
    fig.text(.5,.02,'两条曲线最终相等仅说明已发现源全部处理完；第四问F4-1、F4-3仍缺少未发现源排除证据。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.06,right=.98,top=.91,bottom=.16,hspace=.35,wspace=.25)
    save(fig,'10_正式测试发现清除进度')


def recording_flow():
    fig,ax = plt.subplots(figsize=(11,7.6))
    ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    rows = [('正式包、启动记录、结束回执','校验题号/正式次数/案例码/队号一致，回执SHA256与正式包相符'),
            ('同次运行的原始HTTP记录','按队号及退出时间唯一关联，保留每条请求、响应、策略事件和序号'),
            ('逐指令复核','核对坐标、频道、返回结果、示向度、串行顺序、移动及动作费用'),
            ('独立重建策略证据','检查位置外包与清除证书；全向负观测覆盖可排除频道，定向不能照搬'),
            ('归档、分析、轨迹与论文资料','原始文件原样ZIP；导出完整JSONL和中文指令表；保留来源SHA256')]
    for i,(title,detail) in enumerate(rows):
        y=5.8-i*1.08
        ax.add_patch(FancyBboxPatch((.5,y-.32),9,.73,boxstyle='round,pad=.04',fc='#eef6f8',ec='#438cb0'))
        ax.text(5,y+.16,title,ha='center',va='center',fontsize=13)
        ax.text(5,y-.16,detail,ha='center',va='center',fontsize=9)
        if i<4:
            ax.annotate('',(5,y-.65),(5,y-.39),arrowprops={'arrowstyle':'->','color':'#438cb0'})
    fig.suptitle('正式测试记录与可复核分析流程',fontsize=18,y=.95)
    fig.text(.5,.045,'正式包正文未解密，未验证官方签名，不把本地完成标志、覆盖推论或未知频道数冒充官方成绩。',ha='center',fontsize=10)
    save(fig,'11_正式测试记录与分析流程')


def provenance():
    helpers = ['scripts/analyze_formal_tests.py', 'scripts/build_formal_test_assets.py',
               'tests/test_formal_test_audit.py',
               'scripts/analyze_q4_practice_4.py','scripts/analyze_q3_practice_log.py',
               'scripts/evaluate_q3_routed_practice.py','src/cumcm2026_b/q3_geometry.py',
               'src/cumcm2026_b/q4_strategy.py']
    paths = [ROOT/x for x in helpers]+[REPORT,PAPER]+sorted(ARCHIVE.glob('*'))+sorted(MODELS.glob('*'))+sorted(FIG.glob('*'))
    paths += [p for p in sorted(TABLES.glob('*')) if p != MANIFEST]
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if not args.verify:
        setup()
        runs=[read(p) for p in sorted(MODELS.glob('F*.json.gz'))]
        assert len(runs)==6
        trajectories(runs);costs(runs);timeline(runs);recording_flow()
        write(MANIFEST,provenance())
    assert len(list(FIG.glob('*.png')))==len(list(FIG.glob('*.svg')))==11
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:
            assert min(im.size)>=1100
            im.verify()
    for p in FIG.glob('*.svg'):
        assert ET.parse(p).getroot().tag.endswith('svg')
        assert b'\r\n' not in p.read_bytes()
    assert read(MANIFEST)==provenance()
    print('11张正式测试中文图各PNG/SVG、完整指令表和图文来源核验通过。')


if __name__=='__main__':
    main()
