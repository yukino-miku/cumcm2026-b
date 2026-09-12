"""清除优先版中文覆盖图、流程、20组配对轨迹、费用分析及论文备用说明。"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from evaluate_q4_reliable_selection import (ROOT, TABLE, RUNS, read, dump, path_for, sources,
                                           previous_path, config)
from evaluate_q4_reliable_final import TABLE as COMPARISON, RUNS as COMPARISON_RUNS
from evaluate_q4_reliable import TABLE as PROTOTYPE, RUNS as PROTOTYPE_RUNS
from build_q4_inner13_assets import setup_font
from cumcm2026_b.q4_coverage import discovery_mesh

FIG = ROOT / 'results/figures/第四问/清除优先版'
REPORT = ROOT / 'docs/第四问/清除优先版_方法与验证.md'
PAPER = ROOT / 'paper/sections/第四问_连续发现保障与路线修复_论文备用说明.md'
MANIFEST = ROOT / 'results/tables/第四问/清除优先版图文_来源校验.json'


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ['.png', '.svg']:
        fig.savefig(FIG / (name + ext), dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def trajectory(ax, r, title):
    fixed = np.asarray(r['固定站坐标'])
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec='#755791', ls='--', lw=1.5))
    ax.scatter(*fixed.T, marker='s', s=25, color='#d1d9df', zorder=2)
    current = np.zeros(2)
    all_points = [current]
    for e in r['动作记录']:
        if e['动作'] not in {'检测', '清除'}:
            continue
        point = np.asarray(e['位置'])
        all_points.append(point)
        safeguard = e['阶段'] == '连续发现保障与尾部清除'
        color = '#8964a6' if safeguard else '#249c89' if e['动作'] == '清除' else '#df923b' if e.get('用途') == '追加测向' else '#448bb0'
        if math.dist(current, point) > 1e-6:
            ax.plot([current[0], point[0]], [current[1], point[1]], color=color, lw=1.05, alpha=.82)
            a, b = current + .42 * (point - current), current + .59 * (point - current)
            ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'lw': .75, 'color': color, 'mutation_scale': 7})
        if e.get('用途', '').startswith('固定搜索站'):
            ax.scatter(*point, marker='s', s=22, color='#263f50', zorder=3)
        if e.get('用途') == '追加测向':
            ax.scatter(*point, marker='^', s=15, color='#df923b', zorder=4)
        if e['动作'] == '清除' and e['结果'] == 'success':
            ax.scatter(*point, marker='o', s=25, facecolor='white' if safeguard else '#249c89', edgecolor='#249c89', zorder=5)
        current = point
    extras = np.asarray(r.get('发现保障站坐标', [])).reshape(-1, 2)
    if len(extras):
        ax.scatter(*extras.T, marker='D', s=22, color='#8964a6', zorder=4)
    for s in r['离线真值核验']['目标']:
        missing = s['频道'] in r['离线真值核验']['遗漏频道']
        ax.scatter(*s['位置'], marker='x', s=35 if missing else 10, color='#bf3b3b' if missing else '#555555', zorder=4)
    ax.scatter(0, 0, marker='*', s=80, color='#172f42', edgecolor='white', lw=.4, zorder=6)
    ax.scatter(*current, marker='P', s=35, color='#3f294a', zorder=7)
    bound = max(2050, np.max(np.abs(all_points)) + 120)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel='东向 x（米）', ylabel='北向 y（米）', aspect='equal')
    ax.grid(alpha=.2)
    ax.set_title(f'{title}\n清除 {r["清除数"]}/{r["离线真值核验"]["目标数"]} · {r["虚拟总时间_秒"]:.1f} 秒 · {r["总路程_米"]/1000:.2f} 公里', fontsize=10)


def pairs(data):
    selected = data['选择']
    handles = [Line2D([], [], marker=m, color=c, ls='none', label=t) for m, c, t in
               [('s', '#263f50', '实际固定检测站'), ('^', '#df923b', '主动补测'), ('o', '#249c89', '成功清除位置'),
                ('D', '#8964a6', '发现保障站'), ('x', '#555555', '离线源真值'), ('x', '#bf3b3b', '遗漏源真值')]]
    for stage, count in [('配置比较', 12), ('独立复核', 8)]:
        for i in range(count):
            old = read(previous_path(i, '上一版') if stage == '配置比较' else path_for(stage, i, '仅发现保障'))
            new = read(path_for(stage, i, selected))
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
            trajectory(axes[0], old, '原13站反馈版' if stage == '配置比较' else '仅发现保障对照')
            trajectory(axes[1], new, '清除优先默认版')
            fig.suptitle(f'{stage} {i + 1:02d} · 同地图本地构造，非官方测试', fontsize=13)
            fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False)
            fig.subplots_adjust(bottom=.19, wspace=.27, top=.84)
            save(fig, f'{stage}_{i + 1:02d}_配对轨迹')


def diagrams(data):
    stations, cells = discovery_mesh('compact')
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for triangle in stations[cells]:
        p = np.vstack((triangle, triangle[0]))
        ax.plot(*p.T, color='#bcc8cc', lw=.8)
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec='#755791', ls='--', lw=2, label='源区域边界：1800米'))
    ax.scatter(*stations[:13].T, color='#285568', s=40, label='主路线：13个圆内站', zorder=3)
    ax.scatter(*stations[13:].T, marker='D', color='#a56c37', s=40, label='保障池：12个外环站', zorder=3)
    ax.set(xlim=(-2200, 2200), ylim=(-2200, 2200), aspect='equal', xlabel='东向 x（米）', ylabel='北向 y（米）')
    ax.set_title('25站备选构造的连续发现保障\n36个三角形，最长边983.512米，凸包内切半径1835.259米', fontsize=13)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, fontsize=9, frameon=False)
    ax.grid(alpha=.15)
    fig.text(.5, .02, '外环半径1900米；只有缺少全清证据时补扫，实测站凸包可替代单元顶点。', ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.2)
    save(fig, '01_连续发现覆盖证明')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis('off')
    boxes = [(5, 9, '圆内13站主路线：检测未知与未定位频道'),
             (5, 7.45, '每次停点：顺路清除、补测；不合格候选再找可行替代'),
             (5, 5.9, '是否已发现16个不同频道的源？'),
             (5, 4.35, '其余情况：按真实负观测核查36个连续覆盖单元'),
             (5, 2.8, '缺覆盖：选择必要保障站，开放路线重排并顺路处理目标'),
             (5, 1.25, '终止：16源已清除，或已清除源＋有不存在证书频道＝20')]
    for k, (x, y, label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x - 4.1, y - .43), 8.2, .86, boxstyle='round,pad=.13', fc='#e8f3f1' if k == 5 else '#edf2f7', ec='#40616b'))
        ax.text(x, y, label, ha='center', va='center', fontsize=11)
        if k:
            ax.annotate('', xy=(5, y + .58), xytext=(5, boxes[k - 1][1] - .57), arrowprops={'arrowstyle': '->', 'color': '#40616b'})
    ax.text(5.2, 5.1, '否', fontsize=9)
    ax.annotate('', xy=(9.25, 1.25), xytext=(9.25, 5.9),
                arrowprops={'arrowstyle': '->', 'connectionstyle': 'bar,fraction=-.16', 'color': '#249c89'})
    ax.text(9.5, 3.7, '是：先完成已知源清除', rotation=90, va='center', fontsize=9, color='#249c89')
    ax.annotate('', xy=(.7, 4.35), xytext=(.7, 2.8), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3,rad=-.55', 'color': '#8964a6'})
    ax.text(.05, 3.55, '重查', fontsize=9, color='#8964a6')
    ax.set_title('第四问清除优先流程\n未知频道不因采样无增益、13站结束或连续无信号而被直接判为空', fontsize=13)
    save(fig, '02_策略流程')

    rows = data['逐场']; names = ['仅发现保障', data['选择']]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, stage, count in zip(axes, ['配置比较', '独立复核'], [12, 8]):
        for j, name in enumerate(names):
            values = sorted((r for r in rows if r['阶段'] == stage and r['方法'] == name), key=lambda r: r['组内编号'])
            ax.bar(np.arange(count) + j * .35, [r['时间_秒'] for r in values], .35, label=name, color=['#8ba4b5', '#269d8c'][j])
        ax.set_xticks(np.arange(count) + .175, [str(i + 1) for i in range(count)])
        ax.set(ylabel='总虚拟时间（秒）', xlabel='场景编号', title=f'{stage}：两种方法均全清，所有检测与切换费用计入')
        ax.grid(axis='y', alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(h_pad=2.5)
    save(fig, '03_清除保证相同的时间对比')


def documents(data):
    selected = data['选择']
    groups = {(r['阶段'], r['方法']): r for r in data['分组']}
    old, new = [groups['独立复核', k] for k in ['仅发现保障', selected]]
    change = (new['平均时间_秒'] / old['平均时间_秒'] - 1) * 100
    compare = read(COMPARISON)
    previous_rows = [r for r in compare['逐场'] if r['方法'] == '上一版']
    prev_cleared, prev_total = sum(r['清除数'] for r in previous_rows), sum(r['目标数'] for r in previous_rows)
    rows = sorted((r for r in data['逐场'] if r['阶段'] == '独立复核' and r['方法'] == selected), key=lambda r: r['组内编号'])
    baseline = {r['组内编号']: r for r in data['逐场'] if r['阶段'] == '独立复核' and r['方法'] == '仅发现保障'}
    table = '\n'.join(f'|{r["组内编号"]+1}|{r["场景"]}|{r["清除数"]}/{r["目标数"]}|{r["时间_秒"]:.1f}|{r["时间_秒"]-baseline[r["组内编号"]]["时间_秒"]:+.1f}|{r["保障站数"]}|' for r in rows)
    report = f'''# 第四问清除优先版：方法、证书、验证与运行

用户最新要求优先避免未清除，现将默认入口切换为`reliable`，读取`configs/q4_reliable.json`。保留圆内13站、100米优先补测间距、800米绕行门槛、小范围顺路清除和方向反馈。固定扫描结束时尚无全清证据，就执行必要的连续发现补扫，包括外侧站；不再将13站流程结束当作任务结束。

## 1. 实际采用的优化

1. **发现保障。** 保留原13站主路线，另建半径1900米上的12个外环备选站。按当前频道的实际负观测逐个核查覆盖单元，已经覆盖的单元不补；已清除频道不扫。补充站路线按当前机器人位置做最近邻与开放2-opt，不要求返回圆心。每次处理已知源后重新规划，允许顺路清除，不先机械跑完外环再统一清除。
2. **合格旧候选保留，拒绝后查找替代。** 对旧方案本来会接受的补测点保持其选择；若因800米绕行门槛或完整任务预测而被延后，则将这些限制放入候选筛选，再在可行集合里评分。上一轮F4-3和F4-4的当时历史均通过回归：存在可执行替代，不再因为最佳点不合格就直接放弃整个频道。替代点不保证收到信号，仍按实测反馈处理。
3. **确定超距免测。** 只有已发现频道的连续位置外包到当前站的最短距离大于1500米，才免去必定无信号的检测。没有真实检测就不添加观测记录；未知频道不能用这条规则排除。
4. **16源发现上界。** 已发现与已清除频道合计16个时，不再规划未知源保障站，优先完成这16源的清除。仅仅“已发现16个”尚不等于清除完成。
5. **密集补扫仅保留为可选。** 实现100米停点间距加离散位置/朝向样本增益判断；该组合在对照中检测费用明显增加，因此默认继续600米顺路间隔，完整发现责任由连续保障承担。样本没有增益只允许暂缓此次可选扫描，绝不判定频道不存在。

默认选择为“{selected}”。未恢复已撤回的折线或扇形策略；第三问代码未改。

## 2. 为什么保障构造能避免纯方向漏源

令目标区域为半径1800米闭圆盘。新增备选站为

`Q_k = 1900 (cos(kπ/6), sin(kπ/6)), k=0,...,11`。

与原13站合并得到25个点，其凸包为外侧正十二边形，内切半径为`1900 cos(π/12)=1835.259米`，严格包含目标圆盘。将这些点作Delaunay三角剖分，得到36个三角形；逐边检查最大长度为983.512米，小于最低接收半径1000米。这里连续覆盖的证明来自凸包包含与完整三角剖分，边界采样只是复核。

任意源G落在某三角形内，可写成三个顶点的凸组合，且到每个顶点的距离均不超过该三角形直径。对任意发射方向单位向量u，如果三个顶点都在严格背侧，则三个`u·(P_i-G)`均为负，其凸组合也为负，却又必须等于零，矛盾。因此至少一个顶点位于接收前半面且在1000米内。全向源当然也满足发现条件；源恰在顶点时，原地检测按近距离反馈处理。

该结论保证至少一次发现，不表示每个源自动有两次有效测向，也不保证一次补测定位。后续仍需已有的自适应测向或20米方格有限清除。

### 按频道建立连续不存在证书

对一个尚未发现频道，只能引用真实返回`no_signal`的站。每个三角形单元采用以下任一证据：原三个顶点均实际测过该频道；或者从实测站中选出距该单元三个顶点均不超过999.5米的站，其凸包严格包含单元三个顶点。后者使用1微米内部裕量，近边界数值不确定时保守地不接受替代。原顶点分支要求坐标逐项相等，避免把近似到站当作几何顶点。

所有36单元均通过证书后，假如该频道有源，则至少有一次应收到信号，与全部负反馈矛盾，才可标记不存在。**不能凭“连续两次无信号”“样本没找到可行源”或“13站全跑完”宣称不存在。** 每个证书保存实际负观测坐标、各单元见证站索引及所用网格。

最终全清证据只有两种：清除16个不同频道源；或者所有20频道都已清除或有连续不存在证书。预算、协议或几何异常时记录未完成并退出，不能强行报告全清。

![连续覆盖](../../results/figures/第四问/清除优先版/01_连续发现覆盖证明.png)
![流程](../../results/figures/第四问/清除优先版/02_策略流程.png)

## 3. 试验过程与选择依据

- 首轮107000—107011，共12场×4版本=48次。31站原型的发现保障有效，但成本偏高；原型及完整动作保存在步骤048提交`59237b0`和“清除优先对照”目录，未丢弃慢结果。
- 改用25站构造后，在108000—108011做36次新场景对照。密集未知补扫组合仍变慢，随后该组用于配置比较，因此不能再称其为未参与决策的独立验证。
- 在同一108000组追加“路线修复加免测”及“仅免测”两组，共24次。按异常、漏源、全清证据、总时间依次排序，选定当前配置。三种清除保障方案均157/157；旧13站版只清除{prev_cleared}/{prev_total}，四场全部朝外压力构造均未发现源。旧版提前漏源退出的短时间不能与全清耗时直接排名。
- 冻结选择后，109000—109007独立复核8场，两种有发现保障的配置各运行一次，共16次。全流程累计124次本地运行，源真值只在退出后供审计使用；没有连接官方模拟器，也没有把先前官方六局当作可重复同地图场景。

## 4. 独立复核结果

当前默认和“仅发现保障”对照均107/107清除，8/8场全清且都有连续证据。平均总时间由{old['平均时间_秒']:.1f}秒变为{new['平均时间_秒']:.1f}秒，变化{change:.2f}%；平均路程由{old['平均路程_米']/1000:.3f}公里变为{new['平均路程_米']/1000:.3f}公里，反而略增加。节省部分来自少检测，不能宣称路径普遍更短。

|编号|本地布局|清除|默认总时间/秒|相对仅保障/秒|追加保障站数|
|---|---|---|---|---|---|
{table}

第3、5、6场变慢，第6场增加约556秒；第1、2、4、7、8场变快。8场平均改善很小，不足以证明对任意地图都更快。当前选择优先保证发现和清除，再减少能够确定无收益的操作；路线评分仍属启发式。

发现保障会额外花时间。尤其实际源数少于16时，即使源已经全清，在线程序也不知道真实总数，还需排除其余频道。独立复核平均使用{new['平均保障站数']:.2f}个补充站；这一阶段平均{new['平均保障阶段时间_秒']:.1f}秒，包含补充搜索、所发现源的定位和清除，不能全部归为纯扫描费用。

保证依赖题设的静止源、独立频道、半径至少1000米且不超过1500米、全向或固定180度发射、规定测向误差及20米光学清除范围，也依赖协议正常及预算足够。它是模型内连续几何保证；本地全部通过不能替代官方实测结果。

![费用](../../results/figures/第四问/清除优先版/03_清除保证相同的时间对比.png)

## 5. 运行与复核

PowerShell先进入项目，模拟器里选择第四问演练，再执行：

```powershell
Set-Location 'D:\\mywork\\code\\cumcm2026-b'
.\\.venv\\Scripts\\python.exe -X utf8 scripts/run_q4.py --mode http --robot-id '202623003006'
```

默认`--strategy reliable`，不需要额外指定。日志输出到`local-only/第四问/http-随机编号`，保留实际配置、源码SHA256、Git状态、每次动作及证书。运行成功现在要求有全清证据。`--strategy feedback`恢复此前13站800米反馈版；`--strategy spacing`恢复100米/400米几何版；单独`--probe-spacing 0`保留更早的原版恢复语义。这些恢复版本仍可能漏源，不作为本次默认。

`--dense-unknown-scans`可重试密集增益补扫，因本次对照较慢而默认关闭。`--probe-spacing`控制主动补测间距，`--detour-limit`控制绕行阈值，与未知频道顺路检测间隔是三件不同的事。

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts/evaluate_q4_reliable.py --verify
.\\.venv\\Scripts\\python.exe -X utf8 scripts/evaluate_q4_reliable_final.py --verify
.\\.venv\\Scripts\\python.exe -X utf8 scripts/evaluate_q4_reliable_selection.py --verify
.\\.venv\\Scripts\\python.exe -X utf8 scripts/build_q4_reliable_assets.py --verify
.\\.venv\\Scripts\\python.exe -X utf8 -m pytest -q
```

核验从归档动作复算物理反馈、角度误差、路程、时间、切换、清除证书、实际负观测、连续覆盖、未知/已发现/已清除/不存在状态及同图性，不重新消耗演练次数。现实规划有时限，重新运行可能改变候选评估数量；精确复核用归档动作。

## 6. 图表与队友阅读

图目录为`results/figures/第四问/清除优先版`，共23张，各有PNG和SVG：覆盖构造、策略流程、费用图、12组配置比较配对轨迹和8组独立复核配对轨迹。轨迹图保留整个1800米边界、站点、运动方向、追加补测、清除位置及外侧保障站；源真值只来自退出后的本地环境，未与官方未知真值混淆。灰色固定站是计划站，深色为实际发生检测的站；紫色路径为保障及尾部处理。

核心代码：`q4_reliable_strategy.py`、`q4_coverage.py`。默认参数：`q4_reliable.json`。旧运行代码、原始材料、已有轨迹与历史结果全部保留；改变的历史入口和核验器留有逐字节档案，仍要求精确哈希匹配。
'''
    for stage, count in [('配置比较', 12), ('独立复核', 8)]:
        report += '\n### ' + stage + '全部配对轨迹\n\n'
        report += '\n'.join(f'- [{stage}{i:02d}](../../results/figures/第四问/清除优先版/{stage}_{i:02d}_配对轨迹.png)' for i in range(1, count + 1)) + '\n'
    REPORT.write_text(report, encoding='utf-8', newline='\n')
    PAPER.write_text(f'''# 第四问连续发现保障与路线修复：论文备用说明

## 研究动机

定向发射使圆内测站的距离覆盖不等于信号覆盖。此前六局官方演练清除75/77且两次遗漏均在未发现阶段；本次采用条件补充发现和连续不存在证书，并保持在线算法只能读取观测反馈。改进后的官方实测尚未进行。

## 模型与方法

圆内13站承担主扫描，12个外侧备选站位于半径1900米的正十二边形顶点。全部25站的凸包内切半径1835.259米包含1800米搜索圆盘，其36个三角剖分单元最大直径983.512米小于最低接收半径。任意源是所处单元顶点的凸组合；若所有顶点均位于其发射背面，则沿发射方向投影的凸组合为负，与相对位移凸组合等于零矛盾，从而证明至少一次有效发现。

未知频道按实际负观测单独核验各单元；原三个网格顶点或满足可靠距离且凸包严格包含单元的替代站构成证书。所有单元均通过且全部观测为无信号才能确认该频道不存在。已发现16源时优先完成已知源清除，避免搜索不可能新增的第17源；未达到上界时，无法从已清除数推断真实总数。

已知源的正测向连续外包、清除安全半径以及有限20米格点覆盖保持不变。路线修复仅在原候选会因绕行或完整任务规则被拒绝时，在可行候选中重新评分；已知位置外包完全超出1500米时跳过必无信号检测。离散位置/朝向采样仅参与可选顺路扫描评分，不能删减连续外包或替代发现证书。

## 实验及适用范围

累计124次本地运行，完整保留48次31站原型、36次紧凑构造与密集扫描对照、24次配置比较及16次冻结后的独立复核。配置比较集与独立复核集分离，按遗漏优先、证据优先、时间其次选择。最终独立8场均107/107清除，含两场边界全朝外压力；对照亦全部清除。平均时间{old['平均时间_秒']:.1f}降至{new['平均时间_秒']:.1f}秒（{change:.2f}%），平均路程略增，且三场变慢。因此结果支持模型内发现保障及局部操作优化，不支持对任意场景显著提速的结论。

模型保证依赖题设接收半径、固定180度发射、误差界、静止独立频道及预算和协议条件。公式、完整表、复现命令与23张PNG/SVG图见`docs/第四问/清除优先版_方法与验证.md`。这些构造成绩不能写成官方成绩。
''', encoding='utf-8', newline='\n')


def manifest():
    paths = list(FIG.glob('*')) + [REPORT, PAPER, TABLE, COMPARISON, PROTOTYPE, Path(__file__),
            ROOT / 'configs/q4_reliable.json', ROOT / 'scripts/run_q4.py', ROOT / 'tests/test_q4_reliable.py']
    paths += list(RUNS.rglob('*.gz')) + list(COMPARISON_RUNS.rglob('*.gz')) + list(PROTOTYPE_RUNS.rglob('*.gz'))
    paths += [ROOT / p for p in sources()]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    data = read(TABLE)
    assert read(ROOT / 'configs/q4_reliable.json') == data['默认配置']
    if args.verify:
        assert read(MANIFEST) == manifest()
        assert len(list(FIG.glob('*.png'))) == len(list(FIG.glob('*.svg'))) == 23
        for path in FIG.glob('*'):
            if path.suffix == '.png':
                with Image.open(path) as im:
                    im.verify()
            else:
                ET.parse(path)
        print('核验通过：124次本地运行来源、23张PNG/SVG、方法和论文备用说明。')
        return
    setup_font()
    diagrams(data); pairs(data); documents(data)
    dump(MANIFEST, manifest())
    print('生成23张中文图各PNG/SVG、方法说明、论文备用说明和来源清单。')


if __name__ == '__main__':
    main()
