#!/usr/bin/env python
"""从216次归档重建第四问折线对照图文；--verify只复核，不运行策略。"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from evaluate_q4_zigzag import ROOT, MODELS, TABLE, read, dump, aggregate, selection, sources, spec, summary
from build_q4_inner13_assets import setup_font, verify_run

FIG = ROOT/'results/figures/第四问/折线门槛对照'
DETAIL = ROOT/'results/tables/第四问/折线对照_核验与诊断.json'
MANIFEST = ROOT/'results/tables/第四问/折线对照_文件校验.json'
REPORT = ROOT/'docs/第四问/折线补测_对照实验与运行.md'
PAPER = ROOT/'paper/sections/第四问_折线补测实验_论文备用稿.md'
COLORS = {'几何': '#337ca0', '折线': '#dc8838'}


def load_all():
    table = read(TABLE)
    assert len(table['逐局']) == 216
    assert table['选参门槛'] == selection(table) == {'几何': 600, '折线': 1600}
    truth_by_scene, runs = {}, {}
    totals = dict(measure=0, switch=0, success=0, fail=0, envelope=0)
    for row in table['逐局']:
        stage, i, m, t = (row[k] for k in ['阶段', '场景编号', '方法', '门槛_米'])
        result = read(MODELS/stage/f'{m}-{t}-{i:02d}.json.gz')
        assert result['实验来源SHA256'] == sources()
        assert result['构造参数'] == spec(stage, i)
        assert result['配置'] == result['实验配置']
        assert result['配置']['localization_detour_limit_m'] == t
        if m == '折线':
            assert result['配置']['zigzag_forward_m'] == 200
            assert result['配置']['zigzag_lateral_m'] == 80
        assert summary(result, stage, i, m, t) == row
        key = (stage, i)
        truth = result['离线真值核验']['目标']
        if key in truth_by_scene: assert truth == truth_by_scene[key]
        truth_by_scene[key] = truth
        for k, n in verify_run(result).items(): totals[k] += n
        runs[(stage, i, m, t)] = result
    assert len(truth_by_scene) == 24
    grouped = [{'阶段': stage, '方法': m, '门槛_米': t,
                **aggregate([r for r in table['逐局'] if (r['阶段'], r['方法'], r['门槛_米']) == (stage, m, t)])}
               for stage, m, t in sorted({(r['阶段'], r['方法'], r['门槛_米']) for r in table['逐局']})]
    assert grouped == table['分组汇总']
    return table, runs, totals


def diagnostics(table, runs, totals):
    valid = [r for r in table['逐局'] if r['阶段'] == '验证']
    old = {r['场景编号']: r for r in valid if r['方法'] == '几何' and r['门槛_米'] == 400}
    new = {r['场景编号']: r for r in valid if r['方法'] == '折线'}
    common = [i for i in old if old[i]['实际全清'] and new[i]['实际全清']]
    missing = []
    for i in range(8):
        a, b = runs[('验证', i, '几何', 400)], runs[('验证', i, '折线', 1600)]
        extra = set(b['离线真值核验']['遗漏频道'])-set(a['离线真值核验']['遗漏频道'])
        for c in sorted(extra):
            truth = next(s for s in b['离线真值核验']['目标'] if s['频道'] == c)
            p = np.asarray(truth['位置'])
            axis = truth['发射轴_度']
            unit = None if axis is None else np.array([math.cos(math.radians(axis)), math.sin(math.radians(axis))])
            visible = lambda q: np.linalg.norm(np.asarray(q)-p) <= truth['接收半径']+1e-8 and (unit is None or np.dot(np.asarray(q)-p, unit) >= -1e-8)
            first = next(e for e in a['动作记录'] if e.get('频道') == c and e.get('结果') in ('direction', 'near'))
            fixed_visible = int(sum(visible(q) for q in b['固定站坐标']))
            all_measured_visible = int(sum(visible(e['位置']) for e in b['动作记录'] if e.get('频道') == c and e['动作'] == '检测'))
            assert fixed_visible == 0 and all_measured_visible == 0 and first['用途'] == '顺路检测'
            missing.append({'验证场景编号': i, '频道': c, '目标事后真值': truth,
                            '13固定站中能接收站数': fixed_visible, '折线实际检测中能接收次数': all_measured_visible,
                            '原版首次发现': {k: first[k] for k in ['步骤', '阶段', '用途', '位置', '虚拟时间_秒']}})
    return {'运行总数': 216, '相同真值的独立场景数': 24, '逐动作核验计数': totals,
            '共同全清验证场景编号': common,
            '共同全清原版汇总': aggregate([old[i] for i in common]),
            '共同全清折线汇总': aggregate([new[i] for i in common]),
            '验证8场折线比原版更快场数': sum(new[i]['虚拟时间_秒'] < old[i]['虚拟时间_秒'] for i in old),
            '共同全清更快场数': sum(new[i]['虚拟时间_秒'] < old[i]['虚拟时间_秒'] for i in common),
            '折线相对原版新增遗漏核查': missing}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.svg']:
        p = FIG/(name+suffix)
        fig.savefig(p, dpi=155, metadata={'Date': None} if suffix == '.svg' else None)
        if suffix == '.svg': p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
    plt.close(fig)


def threshold_plot(table):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    keys = [('平均虚拟时间_秒', '平均虚拟时间（秒）', 1), ('漏源总数', '16场遗漏源总数（越少越好）', 1),
            ('平均尾程_米', '平均固定扫描后路程（公里）', .001), ('平均补测无信号次数', '平均主补测无信号次数', 1)]
    for method in ['几何', '折线']:
        rows = [g for g in table['分组汇总'] if g['阶段'] == '选参' and g['方法'] == method]
        for ax, (k, title, factor) in zip(axes.flat, keys):
            ax.plot([r['门槛_米'] for r in rows], [r[k]*factor for r in rows], 'o-', label=method+'选点', color=COLORS[method])
            ax.set(xlabel='单次主补测绕行门槛（米）', ylabel=title)
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('16个选参场景 × 2种补测 × 6种门槛 = 192次本地运行', fontsize=16)
    fig.text(.5, .018, '先比较漏源，再比较时间；平均时间包含未全清场景。缩短收尾不代表总体更好，不能只看时间曲线。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .96))
    save(fig, '01_门槛选参曲线')


def validation_plot(table):
    groups = [g for g in table['分组汇总'] if g['阶段'] == '验证']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    labels = [f"{g['方法']}\n{g['门槛_米']}米" for g in groups]
    fields = [('平均虚拟时间_秒', '平均虚拟时间（秒）', 1), ('清除总数', '实际清除 / 106个源', 1),
              ('平均尾程_米', '平均收尾路程（公里）', .001), ('平均补测无信号次数', '平均主补测无信号次数', 1),
              ('平均补测次数', '平均主补测总次数', 1), ('平均清除失败次数', '平均光学清除未命中次数', 1)]
    for ax, (k, title, factor) in zip(axes.flat, fields):
        bars = ax.bar(labels, [g[k]*factor for g in groups], color=['#337ca0', '#8bb6c8', '#dc8838'])
        ax.bar_label(bars, fmt='%.1f', padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(g[k]*factor for g in groups)*1.22)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('8个未参与选参的验证场景：速度改善，同时存在漏源代价', fontsize=16)
    fig.text(.5, .018, '固定16场选参结果后才运行验证；没有使用验证结果重新调门槛。均为本地构造，非官方成绩。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, '02_独立验证指标')


def schematic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    paths = [np.array([[0., 0.], [-80, 200], [80, 230], [-32, 80]]),
             np.array([[0., 0.], [-80, 200], [49.04, 372.48]])]
    for ax, p in zip(axes, paths):
        ax.axvline(0, color='#ca5555', ls='--', lw=1)
        for a, b in zip(p[:-1], p[1:]): ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', color='#d58a2c', lw=2))
        ax.scatter(*p[0], c='#304d5a', marker='s', s=80)
        ax.scatter(p[1:, 0], p[1:, 1], c='#d58a2c', marker='^', s=75)
        for j, q in enumerate(p): ax.annotate('锚点' if j == 0 else f'补测{j}', q, xytext=(6, 5), textcoords='offset points')
        ax.set(aspect='equal', xlim=(-150, 190), ylim=(-40, 470), xlabel='侧向位移（米）', ylabel='初始测向纵向位移（米）')
        ax.grid(alpha=.13)
    axes[0].set_title('连续两次无信号：第3次缩短至40%')
    axes[0].text(-137, 370, '原锚点不变\n第2点稍向前；第3点缩回\n因此不是强制一直向前走', fontsize=10)
    axes[1].set_title('收到新方向：以该点重新锚定')
    axes[1].plot([-80, -80+260*math.cos(math.radians(75))], [200, 200+260*math.sin(math.radians(75))], '--', color='#ca5555')
    axes[1].text(-137, 430, '新方向示例为75°，横向换边', fontsize=10)
    fig.suptitle('折线补测几何示意：保持横向基线并随反馈更新', fontsize=15)
    fig.text(.5, .012, '数值示意采用纵向200米、横向80米；真实运行随位置外包缩步。示向射线有误差，任何新点均无必然接收保证。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    save(fig, '03_折线补测示意')


def trajectory(ax, r, label, compact=False):
    moves = [e for e in r['动作记录'] if e['动作'] in ('检测', '清除')]
    mark = next(e['步骤'] for e in r['动作记录'] if e['动作'] == '固定扫描完成')
    fixed = np.asarray(r['固定站坐标'])
    pos = np.zeros(2)
    for e in moves:
        p = np.asarray(e['位置']); length = np.linalg.norm(p-pos)
        color = '#9d6aac' if e['步骤'] > mark else '#df8b32' if e.get('用途') == '追加测向' else '#249b86' if e['动作'] == '清除' else '#4b94b1'
        if length > .01:
            ax.plot([pos[0], p[0]], [pos[1], p[1]], color=color, lw=1 if compact else 1.4, alpha=.8)
            if length > 150: ax.annotate('', pos+.62*(p-pos), pos+.46*(p-pos), arrowprops=dict(arrowstyle='-|>', color=color, lw=.8))
        if e.get('用途') == '追加测向': ax.scatter(*p, marker='^', s=13 if compact else 25, c='#df8b32', zorder=4)
        if e['动作'] == '清除' and e['结果'] == 'success': ax.scatter(*p, marker='o', s=18 if compact else 30, c='#249b86', zorder=5)
        pos = p
    ax.add_patch(Circle((0, 0), 1800, fill=False, color='#8968a1', ls='--', lw=1.3))
    ax.scatter(fixed[:, 0], fixed[:, 1], marker='s', s=18 if compact else 29, color='#304d5a', zorder=4)
    if not compact:
        for j, q in enumerate(fixed, 1): ax.annotate(str(j), q, xytext=(5, 4), textcoords='offset points', fontsize=8)
    truth = r['离线真值核验']
    for s in truth['目标']:
        p = np.asarray(s['位置']); miss = s['频道'] in truth['遗漏频道']
        ax.scatter(*p, marker='x', s=25 if compact else 38, c='#cc3636' if miss else '#657078', zorder=6)
        if s['发射轴_度'] is not None:
            angle = math.radians(s['发射轴_度']); v = 115*np.array([math.cos(angle), math.sin(angle)])
            ax.annotate('', p+v, p, arrowprops=dict(arrowstyle='->', color='#b49b63', lw=1))
        if miss: ax.annotate('漏'+str(s['频道']), p, xytext=(4, 4), textcoords='offset points', fontsize=8, color='#b42929')
    ax.scatter(0, 0, marker='*', s=85, c='#293d51', zorder=7)
    ax.scatter(*pos, marker='D', s=30, c='#85497f', zorder=7)
    extent = max([1800]+[abs(v) for e in moves for v in e['位置']])+180
    ax.set(aspect='equal', xlim=(-extent, extent), ylim=(-extent, extent), xlabel='东向 x（米）', ylabel='北向 y（米）')
    ax.grid(alpha=.15)
    ax.set_title(f"{label} · 清除{r['清除数']}/{truth['目标数']}\n路程{r['总路程_米']/1000:.2f}公里｜虚拟时间{r['虚拟总时间_秒']:.1f}秒", fontsize=10 if compact else 12)


def trajectories(table, runs):
    for group in range(2):
        fig, axes = plt.subplots(4, 2, figsize=(12, 23))
        for row in range(4):
            i = 4*group+row
            for ax, method, t in zip(axes[row], ['几何', '折线'], [400, 1600]):
                trajectory(ax, runs[('验证', i, method, t)], f'验证{i:02d} {method}{t}米', True)
        fig.suptitle(f'第四问相同地图配对轨迹（{group+1}/2）', fontsize=18)
        fig.text(.5, .012, '紫线：固定扫描后收尾；橙三角：主补测；绿点：清除成功；红叉：遗漏源；小箭头：事后发射轴。\n固定站/源位置相同，接收反馈随站点变化；本地构造，真值不参与决策。', ha='center', fontsize=11)
        fig.tight_layout(rect=(0, .04, 1, .965))
        save(fig, f'04_验证轨迹总览{group+1}')
    for i in range(8):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
        for ax, method, t in zip(axes, ['几何', '折线'], [400, 1600]):
            trajectory(ax, runs[('验证', i, method, t)], f'验证{i:02d} {method}{t}米')
        handles = [Line2D([], [], color=c, marker=m, linestyle=ls, label=name) for name,c,m,ls in
                   [('扫描及其他移动','#4b94b1',None,'-'),('主补测','#df8b32','^','-'),('清除','#249b86','o','-'),
                    ('扫描后收尾','#9d6aac',None,'-'),('固定站','#304d5a','s','None'),('遗漏源真值','#cc3636','x','None'),('终点','#85497f','D','None')]]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10)
        fig.suptitle(f'相同地图验证{i:02d}：原选点400米 vs 折线1600米（本地构造）', fontsize=15)
        fig.tight_layout(rect=(0, .10, 1, .95))
        save(fig, f'轨迹对照_验证{i:02d}')


def markdown_tables(table):
    train = '| 补测方式 | 门槛/米 | 清除/212 | 全清/16 | 平均时间/秒 | 平均尾程/公里 | 平均补测/无信号 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for g in table['分组汇总']:
        if g['阶段'] == '选参': train += f"| {g['方法']} | {g['门槛_米']} | {g['清除总数']} | {g['全清局数']} | {g['平均虚拟时间_秒']:.1f} | {g['平均尾程_米']/1000:.3f} | {g['平均补测次数']:.2f} / {g['平均补测无信号次数']:.2f} |\n"
    valid = '| 设置 | 清除/106 | 全清/8 | 平均时间/秒 | 平均路程/公里 | 平均尾程/公里 | 平均补测/无信号 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for g in table['分组汇总']:
        if g['阶段'] == '验证': valid += f"| {g['方法']}{g['门槛_米']}米 | {g['清除总数']} | {g['全清局数']} | {g['平均虚拟时间_秒']:.1f} | {g['平均路程_米']/1000:.3f} | {g['平均尾程_米']/1000:.3f} | {g['平均补测次数']:.3f} / {g['平均补测无信号次数']:.3f} |\n"
    return train, valid


def documents(table, detail):
    train, valid = markdown_tables(table)
    common_a, common_b = detail['共同全清原版汇总'], detail['共同全清折线汇总']
    common_percent = (1-common_b['平均虚拟时间_秒']/common_a['平均虚拟时间_秒'])*100
    text = r'''# 第四问折线补测：实现、对照实验与运行

> 状态更新：用户已撤回折线方案。本文及图表仅作历史试验记录，旧`run_q4_zigzag.py`现已转入当前几何方案，默认试用100米补测间距，见[撤回后的当前说明](撤回折线_几何补测间距.md)；下文折线运行命令不再运行历史折线策略。原运行入口已逐字节归档于`docs/第四问/历史代码/run_q4_zigzag_ccb0ad5.py.txt`。

本轮按用户示意实现“沿有效测向前进、左右交替设点”，并测试增大400米绕行门槛。**折线显著减少无信号补测与尾程，但独立验证多漏3个源，不能认定全面优于原版。** 新版作为独立试验入口；原入口与配置保留。

## 1. 当前版本与试验结论

- 原版：`scripts/run_q4.py`，仍使用原几何选点和400米默认门槛。
- 折线试验版：`scripts/run_q4_zigzag.py`，读取`configs/q4_zigzag.json`，配置文件暂取选参所得1600米。此值是本轮6个候选中的选参结果，不是全局最优或已通过覆盖验收的推荐值。
- 两版均保持13站、顺路扫描、定位后清除、路线插入、目标延后和末尾评估；没有重新加入扇形清空或自动未知源外围搜索，前三问代码不变。
- 新模块直接以类默认配置实例化时继承400米；运行入口通过JSON显式加载1600米。归档保存每次实际配置，复现时不要混用。

## 2. 如何生成折线补测点

最近一次真正收到`direction`的检测站作为锚点S，示向度给出单位向量d，n是d逆时针旋转90°的法向量。该正观测可来自固定站、主补测或顺路检测。初始发射轴未知，测得的是“站点指向源”的方向，不是干扰源的发射轴。

将当前保守位置外包投影到d上，投影区间中点相对S的距离记为估计量l（最低5米）。它只用于选择步长，不是真实距离。基本纵向步长与横向宽度为：

$$s=\min(200,\max(20,0.4l)),\qquad w=\min(80,\max(15,0.25l)).$$

最近正观测之后已经执行的该频道主补测次数记为k；该频道全程已经执行的主补测次数记为N。下一点为：

$$q=0.4^{\lfloor k/2\rfloor},\qquad
P=S+q\,s\,[1+0.15(k\bmod2)]d+(-1)^Nq\,w\,n.$$

1. 第一个补测点通常向前不超过200米、横移不超过80米，形成侧向基线。
2. 收到新方向，就把这个有效站点设为新锚点，使用新的方向与位置外包规划下一段；横向按已执行主补测次数交替。
3. 没收到信号，则保留最后有效锚点，尝试另一侧。两次都没有新方向时，下一对步长缩至40%，避免越过近处源后继续向前走。因此允许必要的局部缩回，并非所有段都强制向前。
4. 候选被路线判为暂不顺路时，不消耗折线序号；只有实际执行才计数。遇到已测重复点则转原有限覆盖分支。达到每频道8次主补测预算后也使用原回退。

200米、80米、1.15倍和40%都是本轮固定的启发式构造参数，**没有进行联合最优搜索**。新点不再使用原来的候选角评分，也没有原候选的999.5米全外包距离筛选；可能收到方向、near或无信号。新策略没有估计发射朝向或接收概率，不承诺左右测两次就一定成功。无信号仍不删除1000米圆内位置，不破坏保守定位外包。

![折线选点示意](../../results/figures/第四问/折线门槛对照/03_折线补测示意.png)

```mermaid
flowchart TD
 A[固定站扫描或顺路检测] --> B[更新各频道状态与位置外包]
 B --> C[保持固定站顺序的路线插入]
 C --> D{下一任务}
 D -->|固定站| A
 D -->|已经定位| E[沿路线清除并顺路检测]
 D -->|已发现未定位| F[最近有效方向锚定折线候选]
 F --> G{实际单次绕行不超过门槛}
 G -->|否| H[暂缓该目标]
 H --> C
 G -->|是| I[执行补测与顺路检测]
 I -->|收到方向| J[新锚点 新方向 重新估计步长]
 I -->|无信号| K[保留锚点 换侧 每两次缩步]
 J --> B
 K --> B
 E --> B
```

图为扫描中核心流程；每轮连续两次无信号延后、24服务动作检查、每频道8次主补测及收尾有限光学覆盖仍按原规则执行。固定扫描完成后的处理没有“下一固定站”门槛分支，见原版[完整说明](当前策略完整说明.md)。

## 3. 门槛到底控制什么

当前位置X、候选补测点P、下一个固定站F对应额外路程：

$$\Delta=|XP|+|PF|-|XF|.$$

默认旧门槛为400米，新试验配置为1600米。只有当路线规划先选中该目标、且候选通过此门槛时才去补测；调大门槛不会强制把所有已发现源立即处理。门槛是单个补测动作的筛选，不是整频道或整局的绕行总预算，也不能保证不回头。以5米/秒计，1600米相当于该局部比较最多多320秒移动，须靠减少之后的回访等收益补偿。

路线仍使用未定位外包中心作为位置预测，未加入全局时间最优求解；已定位清除仍由原路线插入安排，不因更改补测门槛改变清除半径。

## 4. 配对试验设计

在读取本轮结果前固定：16个选参场景，种子96000—96015；8个独立验证场景，种子97000—97007。每8场依次覆盖均匀混合、均匀定向变半径、边界切向定向、边界混合变半径、聚集定向、中心混合、均匀定向最小半径、边界朝内混合。目标10—16个，定向比例50%或100%，半径最小1000米或混合1000—1500米。

每个选参场景交叉比较两种补测和400、600、800、1000、1200、1600米，共192次；同一场景的源位置、频道、发射轴、半径及误差生成规则一致，检测地点变化时反馈允许随之改变。逐份核对真值相同，策略只读公开反馈，结束后才添加真值用于评价。来源SHA256覆盖12个运行模块。

评价顺序预先固定为：先最少异常、最少漏源、最多全清局，再最低平均虚拟时间；并列取小门槛。两种选点分别选参，结果为几何600米、折线1600米。固定结果后，8个新场景只比较原几何400米、选参几何600米、选参折线1600米，共24次。**验证集不用于重新选择门槛。**

均为本地构造，非用户六局官方演练的重跑，也不是正式成绩。现有演练日志没有完整真值，无法用其直接模拟更改后的同地图反馈。全部未全清案例保留。

## 5. 选参结果

__TRAIN__

几何选点单纯加大门槛并不稳定改善时间或清除率；折线版800—1600米的选参平均时间相近，1600米位于搜索区间上端，不能外推“越大越好”。折线400到1600米，平均时间只降低约69.6秒，平均尾程从3.559降至1.363公里；大量工作提前发生，尾程减少远大于总时间收益。

![门槛曲线](../../results/figures/第四问/折线门槛对照/01_门槛选参曲线.png)

## 6. 独立验证：收益与代价同时报告

__VALID__

折线1600米相对原版400米，8场平均时间降低11.35%，平均路程减少约2.380公里，平均尾程从7.412降至1.145公里。主补测无信号总数从134降至12，但主补测总数从219增至251；更容易收到信号不等于一次就达到定位精度。新折线的小横移有时需要多次交会收缩，验证05中心混合场景反而从4752.6增至5552.4秒。

**平均时间包含未全清场景，不能直接解释为同等任务效率提高11.35%。** 同时报告双方都全清的__NCOMMON__场（编号__COMMON__）：原版平均__OLDTIME__秒，折线平均__NEWTIME__秒，减少__PERCENT__%；其中__FASTER__场更快。这是事后描述子集，不能代替完整验证集评价或当成总体保证。

![独立验证指标](../../results/figures/第四问/折线门槛对照/02_独立验证指标.png)

## 7. 为什么短路线反而多漏了源

折线版已发现的源均完成清除，新增遗漏发生在发现阶段。逐源用事后真值核查：验证02频道9、验证03频道2与8，在全部13个固定站均不满足接收条件；折线版实际对其执行的检测点也全部无接收。原版分别在顺路检测时首次发现这些源，其中频道9和8是在固定扫描完成后的绕行收尾阶段发现。

因此原版某些“绕路”同时提供了对未知定向源的额外扫描。减少这些轨迹，虽然降低已知源补测及回访成本，却也改变了未知源的发现机会。这不是折线定位把已发现目标丢失，也不是证明原版13站完全覆盖。详见[逐源核验JSON](../../results/tables/第四问/折线对照_核验与诊断.json)。

当前结论：保留折线作为可运行实验版；1600米只作本轮选参候选，原默认入口仍保留。后续若要把它作为最终方案，需要同时验证未知源发现能力；本轮未自动加入用户此前排除的外围搜索，也未用独立验证集反复调参来掩盖退化。

## 8. 同地图轨迹图

固定站顺序相同，方块为13个固定站；橙三角为主补测，绿圆为成功清除，紫线为固定扫描后收尾；红叉为遗漏源，细箭头为事后发射轴，不提供给策略。轨迹按真实动作顺序绘制，不人为闭合、不删除回头段。

![配对轨迹前四场](../../results/figures/第四问/折线门槛对照/04_验证轨迹总览1.png)

![配对轨迹后四场](../../results/figures/第四问/折线门槛对照/04_验证轨迹总览2.png)

__LINKS__

## 9. 运行与复现

PowerShell先进入项目目录：

```powershell
Set-Location 'D:\mywork\code\cumcm2026-b'
```

仅在本地构造运行新版本：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_q4_zigzag.py --mode local --seed 97000 --count 10
```

用户自行选择模拟器第四问演练后，可运行（本轮没有连接模拟器）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_q4_zigzag.py --mode http --robot-id '你的队号'
```

测试其他门槛，例如800米：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_q4_zigzag.py --mode http --robot-id '你的队号' --detour-limit 800
```

日志保存于新建的`local-only/第四问/{模式-唯一编号}/`，包含源码来源、实际配置、请求响应和运行结果。回到旧版仍运行`scripts/run_q4.py`；不必删除新版。

重建并核验本轮归档与图文：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_q4_zigzag.py --stage 选参
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_q4_zigzag.py --stage 验证
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_q4_zigzag.py --stage 核验
.\.venv\Scripts\python.exe -X utf8 scripts/build_q4_zigzag_assets.py
.\.venv\Scripts\python.exe -X utf8 scripts/build_q4_zigzag_assets.py --verify
```

选参和验证命令默认读取已存在归档，核对源码版本，补跑缺失局；已有结果不会被悄悄重跑覆盖。若运行源码改变会拒绝混用旧实验。192+24份完整运行压缩JSON在`results/models/第四问/折线门槛对照/`，汇总在`results/tables/第四问/折线门槛对照.json`。

原几何方法仍含0.8秒候选规划上限；不同硬件重新执行可能改变实际处理的候选数量。跨机器精确复核应重读原始动作并重算物理反馈、费用和图表，不能仅以随机种子承诺逐动作完全一致。
'''
    substitutions = {'__TRAIN__': train, '__VALID__': valid, '__NCOMMON__': str(len(detail['共同全清验证场景编号'])),
                     '__COMMON__': '、'.join(f'{i:02d}' for i in detail['共同全清验证场景编号']),
                     '__OLDTIME__': f"{common_a['平均虚拟时间_秒']:.1f}", '__NEWTIME__': f"{common_b['平均虚拟时间_秒']:.1f}",
                     '__PERCENT__': f'{common_percent:.2f}', '__FASTER__': str(detail['共同全清更快场数']),
                     '__LINKS__': '\n'.join(f'- [验证{i:02d}高清配对轨迹](../../results/figures/第四问/折线门槛对照/轨迹对照_验证{i:02d}.png)' for i in range(8))}
    for k, value in substitutions.items(): text = text.replace(k, value)
    REPORT.write_text(text, encoding='utf8', newline='\n')
    paper = '''# 第四问折线补测实验：论文备用材料

> 状态：折线方案已按用户要求撤回。以下为历史试验材料，不作为当前执行方案。

以下结果来自本地构造对照，不能写成官方测试成绩或完整区域覆盖证明。实现与公式见[实验完整说明](../../docs/第四问/折线补测_对照实验与运行.md)。

针对定向干扰导致的重复无信号补测，构造由最近有效测向锚定的交替折线路径。每次有效观测后更新锚点与方向，在纵向推进的同时保留侧向基线；一对试探均未获得新方向时按0.4比例缩短步长。位置外包随正向观测更新，负向反馈保留未知距离与发射朝向的歧义。补测仍受单次路线增量门槛约束，定位清除与固定站顺序使用原调度机制。

实验采用16场选参和8场独立验证。选参部分交叉比较两种选点方式及六个绕行门槛，共192次；以漏源优先、时间次优的规则分别选出原几何600米、折线1600米，再与原几何400米共同在8个新场景验证。全部216次运行均保存动作与事后真值，独立核对接收物理条件、位置外包、清除条件和虚拟费用。

'''+valid+f'''
折线1600米的平均主动补测无信号次数由16.75下降至1.50，平均收尾路程由7.412公里下降至1.145公里；平均总时间下降11.35%，但清除数同时从103/106下降至100/106。双方共同全清的{len(detail['共同全清验证场景编号'])}场中，平均时间由{common_a['平均虚拟时间_秒']:.1f}秒下降至{common_b['平均虚拟时间_秒']:.1f}秒，降幅{common_percent:.2f}%；此子集为事后描述，不能替代完整验证结果。

对新增遗漏的三个源进行接收几何复核，发现13个固定站均无法接收，原版依靠额外轨迹上的顺路检测发现。缩短已知源处理路线的同时减少了对未知源的额外空间扫描，显示定位效率与发现完整性之间存在耦合。因此本实验支持折线改善有效反馈率，但不支持“折线1600米全面更优”或“13站必然发现全部定向源”的结论。门槛1600米为本轮有限候选集的选参结果，尚不能作为普适最优值。

建议配图：折线补测示意、门槛选参曲线、独立验证指标、验证02与03的相同地图轨迹。全部PNG/SVG位于`results/figures/第四问/折线门槛对照/`，不能只选时间下降而隐去新增漏源图。
'''
    PAPER.write_text(paper, encoding='utf8', newline='\n')


def manifest():
    paths = list(MODELS.rglob('*.json.gz'))+list(FIG.glob('*'))+[TABLE, DETAIL, REPORT, PAPER,
        ROOT/'scripts/evaluate_q4_zigzag.py', Path(__file__), ROOT/'docs/第四问/历史代码/run_q4_zigzag_ccb0ad5.py.txt',
        ROOT/'configs/q4_zigzag.json', ROOT/'tests/test_q4_zigzag.py']+[ROOT/p for p in sources()]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--verify', action='store_true'); args = p.parse_args()
    table, runs, totals = load_all()
    detail = diagnostics(table, runs, totals)
    if args.verify:
        assert read(DETAIL) == detail
        assert read(MANIFEST) == manifest()
        for path in FIG.glob('*'):
            if path.suffix == '.png':
                with Image.open(path) as im: im.verify()
            elif path.suffix == '.svg': ET.parse(path)
        print(json.dumps({'核验': '通过', '统计': detail}, ensure_ascii=False, indent=2))
        return
    setup_font(); plt.rcParams['svg.hashsalt'] = 'q4-zigzag-20260912'
    threshold_plot(table); validation_plot(table); schematic(); trajectories(table, runs)
    dump(DETAIL, detail); documents(table, detail); dump(MANIFEST, manifest())
    print(f'已生成{len(list(FIG.glob("*.png")))}张中文图及SVG、报告、论文备用稿和文件校验。')


if __name__ == '__main__': main()
