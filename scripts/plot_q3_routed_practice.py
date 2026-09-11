#!/usr/bin/env python
"""从脱敏数据绘制s11/s22十二局轨迹与评估图；不读取私有输入或连接模拟器。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from evaluate_q3_routed_practice import REPORT, TRACKS, write_json, digest
from plot_q3_practice_trajectories import draw, legend
from build_q3_scheme1_assets import style, save
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

FIGDIR = ROOT/'results/figures/第三问'
GUIDE = ROOT/'docs/第三问/新版十二局实际轨迹_s11_s22.md'
MANIFEST = ROOT/'results/tables/第三问/新版演练图表清单_s11_s22.json'
OVERVIEWS = {'s11': '新版演练_方案一六局轨迹', 's22': '新版演练_方案二六局轨迹'}
COMPARE = ['新版演练_费用与逐局结果', '新版演练_分层及历史对照', '新版演练_顺路清除与剩余路程']


def verify_tracks(data, report):
    assert digest(TRACKS) == report['轨迹数据SHA256']
    assert len(data['逐局轨迹']) == 12
    for run in data['逐局轨迹']:
        row = next(r for r in report['逐局记录'] if r['显示编号'] == run['编号'])
        assert run['案例码'] == row['案例码'] and run['全清'] == row['全清核对通过']
        assert run['全清'], '本图标题面向本批全部清除记录，不应用于未完成运行'
        actions = run['实际动作']
        points = np.asarray([[0., 0.]] + [e['位置'] for e in actions])
        assert np.isfinite(points).all()
        assert abs(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()-run['总路程_米']) < 1e-5
        assert sum(e['动作'] == '检测' for e in actions) == run['检测次数'] == row['检测次数']
        assert sum(e['动作'] == '清除' and e['结果'] == 'success' for e in actions) == run['清除成功次数'] == row['成功清除数']
        assert sum(e['动作'] == '清除' and e['结果'] != 'success' for e in actions) == run['清除失败次数']
        assert sum(e['用途'] == '顺路证书清除' for e in actions) == run['路线调度统计']['顺路证书清除次数']
        assert run['已完成固定扫描站序号'] == row['已完成固定站']


def draw_current(ax, run, limit, detail=False):
    draw(ax, run, detail)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    for e in run['实际动作']:
        if e['用途'] == '顺路证书清除' and e['结果'] == 'success':
            ax.scatter(*e['位置'], s=13 if detail else 8, color='#25957E', zorder=7)
    if detail:
        for e in run['实际动作']:
            if e['动作'] == '清除' and e['结果'] != 'success':
                ax.annotate(f"一次清除失败：频道{e['频道']}", e['位置'], xytext=(12, -24),
                            textcoords='offset points', fontsize=8, color='#B84444',
                            arrowprops={'arrowstyle': '-', 'color': '#B84444', 'lw': .6})


def plot_trajectories(data):
    runs = data['逐局轨迹']
    max_coordinate = max(abs(v) for r in runs for e in r['实际动作'] for v in e['位置'])
    limit = max(2050, int(np.ceil((max_coordinate+100)/250)*250))
    handles = legend() + [Line2D([], [], marker='o', color='#25957E', ls='', markersize=4, label='实心绿点：顺路证书清除')]
    notice = '箭头表示较长路段行进方向；同地点多次检测不重复连线；清除位置是机器狗位置，非目标精确真值'
    (FIGDIR/'新版演练轨迹').mkdir(parents=True, exist_ok=True)
    for group, title in [('s11', '方案一加路线规划：六局实际轨迹'), ('s22', '方案二两项改进：六局实际轨迹')]:
        fig, axes = plt.subplots(2, 3, figsize=(17.6, 13.0))
        for ax, run in zip(axes.flat, [r for r in runs if r['组'] == group]): draw_current(ax, run, limit)
        fig.suptitle(title, fontsize=19, y=.985)
        fig.text(.5, .953, '各组按结束时间编号；所有轨迹同尺度；两组案例不同，不构成同场景配对', ha='center', fontsize=10)
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .030), ncol=4,
                   fontsize=9, frameon=False, columnspacing=1.6)
        fig.text(.5, .012, notice, ha='center', fontsize=9, color='#586A73')
        fig.subplots_adjust(left=.065, right=.97, bottom=.17, top=.895, wspace=.25, hspace=.31)
        save(fig, OVERVIEWS[group])
    for run in runs:
        fig, ax = plt.subplots(figsize=(9.5, 11.2))
        draw_current(ax, run, limit, True)
        fig.suptitle(f"{run['编号']} 实际行动轨迹", fontsize=18, y=.98)
        fig.text(.5, .934, f"案例码：{run['案例码']}  |  检测{run['检测次数']}次  |  顺路证书清除{run['路线调度统计']['顺路证书清除次数']}个",
                 ha='center', fontsize=10, color='#586A73')
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .03), ncol=3,
                   fontsize=8.5, frameon=False, columnspacing=1.3)
        fig.text(.5, .012, '1800米圆为目标区域边界；路线按实际动作连接，未添加回原点的路径', ha='center', fontsize=8)
        fig.subplots_adjust(left=.12, right=.95, bottom=.225, top=.86)
        save(fig, f"新版演练轨迹/{run['编号']}_实际轨迹")
    return limit


def plot_comparisons(report):
    rows, stats = report['逐局记录'], report['统计']
    groups = ['s11', 's22']; colors = {'s11': '#347CA0', 's22': '#C77B39'}
    note = '用户提供的实际演练，各组6局、均已全清；案例未配对，目标数量及场景难度不同'
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={'width_ratios': [1, 1.15]})
    bottom = np.zeros(2)
    for key, color in zip(['移动', '检测', '切换', '清除成功', '清除失败'], ['#347CA0', '#62A9AE', '#D1AE59', '#57976F', '#B55748']):
        values = [stats[g]['平均时间分项_秒'][key] for g in groups]
        axes[0].bar([0, 1], values, bottom=bottom, color=color, label=key, width=.5); bottom += values
    for i, value in enumerate(bottom): axes[0].text(i, value+70, f'{value:.2f}', ha='center')
    axes[0].set(xticks=[0, 1], xticklabels=['方案一改进版\ns11', '方案二改进版\ns22'],
                ylabel='平均每局虚拟时间（秒）', title='全部费用均计入，包括失败清除', ylim=(0, bottom.max()*1.2))
    axes[0].legend(ncol=3, fontsize=8, loc='upper left')
    axes[1].barh(range(12), [r['总虚拟时间_秒'] for r in rows], color=[colors[r['组']] for r in rows])
    axes[1].set(yticks=range(12), yticklabels=[f"{r['显示编号']}（{r['官方结果文件目标总数']}目标）" for r in rows],
                xlabel='总虚拟时间（秒）', title='12局全部列出，按组内结束时间编号', xlim=(0, max(r['总虚拟时间_秒'] for r in rows)*1.15))
    axes[1].invert_yaxis()
    for i, r in enumerate(rows): axes[1].text(r['总虚拟时间_秒']+30, i, f"{r['总虚拟时间_秒']:.1f}", va='center', fontsize=8)
    fig.suptitle('新版两组演练：费用分解与全部逐局结果', fontsize=16)
    fig.text(.5, .017, note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .935)); save(fig, COMPARE[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    strata = report['两方案比较']['共同目标数分层']
    for i, (label, group) in enumerate([('第一组', 's11'), ('第二组', 's22')]):
        x = np.arange(len(strata)) + (i-.5)*.36
        values = [r[label]['平均每目标时间_秒'] for r in strata]
        axes[0].bar(x, values, width=.34, color=colors[group], label=group)
        for xx, v, r in zip(x, values, strata): axes[0].text(xx, v+6, f"{v:.1f}\nn={r[label]['样本数']}", ha='center', fontsize=9)
    axes[0].set(xticks=range(len(strata)), xticklabels=[f"{r['目标数']}个目标" for r in strata],
                ylabel='平均每局 T/N（秒）', title='共同目标数分层：同数量不等于同难度', ylim=(0, 470))
    axes[0].legend(loc='upper right')
    old = report['原版历史统计']
    values = [old['s1']['平均每目标虚拟时间_秒'], stats['s11']['平均每目标虚拟时间_秒'],
              old['s2']['平均每目标虚拟时间_秒'], stats['s22']['平均每目标虚拟时间_秒']]
    axes[1].bar(range(4), values, color=['#A1BDCC', '#347CA0', '#DFC5AE', '#C77B39'])
    for i, v in enumerate(values): axes[1].text(i, v+8, f'{v:.2f}', ha='center')
    axes[1].set(xticks=range(4), xticklabels=['原方案一\ns1', '新方案一\ns11', '原方案二\ns2', '新方案二\ns22'],
                ylabel='平均每局 T/N（秒）', title='历史非配对参考：每组6局，场景不同', ylim=(0, max(values)*1.22))
    fig.suptitle('新版比较与历史参考：保留目标数构成和样本边界', fontsize=16)
    fig.text(.5, .018, '历史均值变化不能直接归因于算法修改；分层均值、样本数和不确定性见完整报告', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94)); save(fig, COMPARE[1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    before = np.array([r['末次固定检测前成功清除数'] for r in rows])
    after = np.array([r['成功清除数'] for r in rows]) - before
    axes[0].bar(range(12), before, color='#25957E', label='最后一次固定检测之前已清除')
    axes[0].bar(range(12), after, bottom=before, color='#B7CFD0', label='之后清除')
    for i, (a, b) in enumerate(zip(before, after)): axes[0].text(i, a+b+.3, f'{a}+{b}', ha='center', fontsize=9)
    axes[0].set(ylabel='成功清除目标数', ylim=(0, 19), title='各局均在固定搜索期间处理了部分目标')
    axes[0].legend(ncol=2, loc='upper left', fontsize=9)
    before_l = np.array([r['按阶段路程_米']['末次固定检测及之前']/1000 for r in rows])
    after_l = np.array([r['按阶段路程_米']['末次固定检测之后']/1000 for r in rows])
    axes[1].bar(range(12), before_l, color='#347CA0', label='最后一次固定检测及之前的路程')
    axes[1].bar(range(12), after_l, bottom=before_l, color='#C77B39', label='之后的剩余处理路程')
    for i, (a, b) in enumerate(zip(before_l, after_l)): axes[1].text(i, a+b+.3, f'{a+b:.2f}', ha='center', fontsize=9)
    axes[1].set(xticks=range(12), xticklabels=[r['显示编号'] for r in rows], ylabel='实际行走路程（公里）',
                ylim=(0, (before_l+after_l).max()*1.3), title='清除时机已改变，但最后仍有剩余目标需要处理')
    axes[1].legend(ncol=2, loc='upper left', fontsize=9)
    for ax in axes: ax.axvline(5.5, color='#94A3B8', ls='--', lw=1)
    fig.suptitle('顺路机制的实际执行：提前清除与剩余行程', fontsize=16)
    fig.text(.5, .018, '阶段界线为日志中最后一次实际固定检测，不将“之后的路程”解释成全部可省的回头路', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .94)); save(fig, COMPARE[2])


def guide(data, limit):
    lines = ['# 新版十二局实际演练轨迹：s11 / s22', '',
             '本页对应用户提供的s11、s22各6局，按各组结束时间编号。s11为方案一加路线规划，s22为方案二同时启用路线规划与停止重复扫描。所有图均由实际请求/反馈核对后的动作坐标还原，原12局s1/s2图表保留不变。', '',
             '## 读图方式', '',
             '- 蓝线前往固定站，橙线前往追加测向点，绿线前往清除点。颜色按该段终点动作归类，不表示目标类别。',
             '- 较长线段的箭头表示行进方向，同位置的多次检测不重复连线；不人为闭合路线或添加返回原点的路程。',
             '- 绿色空心圆为成功清除位置，圆内实心绿点表示顺路证书清除；红叉表示失败清除。清除位置是机器狗的位置，不是目标精确真值。',
             '- 方块为固定站，实心表示该站完成固定扫描，空心表示其余预设固定站；三角形为追加测向点。黑星为起点，紫菱形为终点。',
             f'- 虚线圆半径1800米，为目标所在区域；机器狗可以在圆外行动。所有轨迹使用±{limit}米的同一坐标范围与等比例坐标轴。',
             '- 本组方案一六局均完成7站，方案二六局均完成14站；固定站次序保持，目标清除穿插其中。两组案例不同，相同面板位置不是同场景配对。', '',
             '## 方案一：六局总览', '', f"![s11六局轨迹](../../results/figures/第三问/{OVERVIEWS['s11']}.png)", '',
             '## 方案二：六局总览', '', f"![s22六局轨迹](../../results/figures/第三问/{OVERVIEWS['s22']}.png)", '',
             '## 十二张单局高清图', '', '| 编号 | 案例码 | 目标数 | 路程/公里 | 顺路证书清除数 | PNG | SVG |',
             '|---|---|---:|---:|---:|---|---|']
    for run in data['逐局轨迹']:
        base = f"../../results/figures/第三问/新版演练轨迹/{run['编号']}_实际轨迹"
        lines.append(f"| {run['编号']} | {run['案例码']} | {run['目标数']} | {run['总路程_米']/1000:.3f} | "
                     f"{run['路线调度统计']['顺路证书清除次数']} | [查看]({base}.png) | [矢量图]({base}.svg) |")
    lines += ['', 'S22-3保留一次有限方格搜索失败的红叉及后续成功清除位置。S22-5与S22-6均为16个目标，但路程相差约5.59公里，图中按实际记录保留；同目标数并不代表同场景。', '',
              '路线交叉、折返或最后阶段的较长路程，不能单凭轨迹图判定为完全可省。是否可以延后或改换测向位置，还需结合当时已经获得的观测与清除证书。', '',
              '## 数据与复现', '',
              '[完整评估报告](新版演练评估_s11_s22.md)；[脱敏完整轨迹](../../results/tables/第三问/新版十二局实际轨迹_s11_s22.json)；[图表来源与哈希](../../results/tables/第三问/新版演练图表清单_s11_s22.json)。', '',
              '从仓库根目录执行`.venv/Scripts/python.exe -X utf8 scripts/plot_q3_routed_practice.py`即可读取已归档的脱敏数据重绘，不需要本机原始私有日志。', '',
              '如要重新提取数据，先执行`scripts/evaluate_q3_routed_practice.py`，需要本机s11/s22三件套与已关联的HTTP日志，并保留原s1/s2来源用于核对历史对照哈希；该操作只读取现有文件，不启动演练。', '',
              '[绘图脚本](../../scripts/plot_q3_routed_practice.py)；[数据提取及核验脚本](../../scripts/evaluate_q3_routed_practice.py)。']
    GUIDE.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    data = json.loads(TRACKS.read_text(encoding='utf-8'))
    verify_tracks(data, report)
    style(); plt.rcParams['svg.hashsalt'] = 'q3-routed-practice-s11-s22'
    limit = plot_trajectories(data)
    plot_comparisons(report)
    guide(data, limit)
    names = list(OVERVIEWS.values()) + [f"新版演练轨迹/{r['编号']}_实际轨迹" for r in data['逐局轨迹']] + COMPARE
    outputs = {}
    for name in names:
        for ext in ['png', 'svg']:
            path = FIGDIR/f'{name}.{ext}'
            if ext == 'png':
                with Image.open(path) as im: im.verify()
            else: ET.parse(path)
            outputs[path.relative_to(ROOT).as_posix()] = digest(path)
    sources = [REPORT, TRACKS, Path(__file__), ROOT/'scripts/plot_q3_practice_trajectories.py', ROOT/'scripts/build_q3_scheme1_assets.py']
    write_json(MANIFEST, {'说明': '12张实际轨迹、2张总览和3张评估图，均为PNG/SVG；不含构造轨迹。',
                          '统一坐标半宽_米': limit, '图表SHA256': outputs,
                          '绘图来源SHA256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}})
    print(f'ROUTED_PRACTICE_PLOTS_OK：12张单局轨迹、2张总览、3张评估图，共{len(outputs)}个PNG/SVG文件。')


if __name__ == '__main__': main()
