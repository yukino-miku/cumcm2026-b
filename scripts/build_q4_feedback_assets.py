"""生成第四问改进版的中文对照图、流程、方法与论文备用资料。"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from evaluate_q4_discovery import (ROOT, TABLE, RUNS, BASELINE, IMPROVED, read, dump, run_path, sources)
from evaluate_q4_feedback import TABLE as FIRST_TABLE, RUNS as FIRST_RUNS
from build_q4_inner13_assets import setup_font
from build_q4_zigzag_assets import trajectory
from cumcm2026_b.q4_feedback_strategy import FeedbackFour, FeedbackConfig, joint_hypotheses, reception_support
from cumcm2026_b.q4_local_env import LocalEnvironment
from cumcm2026_b.q3_protocol import RobotClient

FIG = ROOT / 'results/figures/第四问/改进版_方向反馈与顺路覆盖'
REPORT = ROOT / 'docs/第四问/改进版_方向反馈与顺路覆盖.md'
PAPER = ROOT / 'paper/sections/第四问_方向反馈与顺路覆盖_论文备用说明.md'
MANIFEST = ROOT / 'results/tables/第四问/改进版图文_来源校验.json'


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        path = FIG / (name + suffix)
        fig.savefig(path, dpi=160, metadata={'Date': None} if suffix == '.svg' else None)
        if suffix == '.svg':
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    plt.close(fig)


def figures(first, final):
    setup_font()
    plt.rcParams['svg.hashsalt'] = 'q4-feedback-discovery-20260912'
    train = [g for g in first['分组汇总'] if g['阶段'] == '选参']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    labels = [g['方法'].replace('完整任务', '任务\n').replace('仅方向反馈', '方向\n').replace('两项组合', '组合\n').replace('仅小范围', '小范围\n') for g in train]
    for ax, key, title in zip(axes, ('平均虚拟时间_秒', '清除总数'), ('平均虚拟时间（秒）', '清除数量 / 212')):
        bars = ax.bar(labels, [g[key] for g in train], color=['#dfa050' if g['方法'] == first['选参设置'] else '#5294aa' for g in train])
        ax.bar_label(bars, fmt='%.0f', padding=4)
        ax.set(title=title, ylim=(0, max(g[key] for g in train) * 1.18))
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('16场选参：7种设置、112次运行；先比较发现数量，再比较总时间', fontsize=16)
    fig.text(.5, .018, '800米在此阶段被选中；后续验证仍暴露未知频道漏扫，完整过程另见报告。均为本地构造。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .92))
    save(fig, '01_消融与门槛选参')
    fresh = [r for r in final['逐局'] if r['阶段'] == '新验证']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(8)
    for name, offset, color in ((BASELINE, -.18, '#5294aa'), (IMPROVED, .18, '#dfa050')):
        rows = [next(r for r in fresh if r['场景编号'] == i and r['方法'] == name) for i in range(8)]
        axes[0].bar(x + offset, [r['虚拟时间_秒'] for r in rows], width=.36, color=color, label='基线100米' if name == BASELINE else '最终改进版')
        bars = axes[1].bar(x + offset, [r['清除数'] for r in rows], width=.36, color=color)
    for i in range(8):
        a, b = [next(r for r in fresh if r['场景编号'] == i and r['方法'] == name) for name in (BASELINE, IMPROVED)]
        axes[1].text(i, max(a['清除数'], b['清除数']) + .25,
                     f"{a['清除数']}→{b['清除数']}\n共{a['目标数']}个", ha='center', va='bottom', fontsize=8)
    axes[0].set(title='逐场总时间：包含补齐未知频道的测量开销', ylabel='虚拟时间（秒）')
    axes[0].legend(fontsize=9)
    axes[1].set(title='实际清除数：基线104/106，改进106/106', ylabel='目标数', ylim=(0, 19))
    for ax in axes:
        ax.set_xticks(x, [f'{i:02d}' for i in range(8)])
        ax.set_xlabel('新验证场景编号（103000—103007）')
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    save(fig, '02_新验证逐场费用与清除')
    for i in range(8):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
        for ax, name, label in zip(axes, (BASELINE, IMPROVED), ('基线100米', '最终改进版800米')):
            trajectory(ax, read(run_path('新验证', i, name)), f'新验证{i:02d} {label}')
        fig.suptitle(f'新验证{i:02d}：同地图实际动作轨迹对照', fontsize=16)
        fig.text(.5, .02, '紫线：固定扫描后收尾；橙三角：主补测；绿点：清除成功；方块：固定站。\n叉号及细箭头是退出后才读取的源位置与发射轴，红叉表示遗漏；均为本地构造。', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .13, 1, .94))
        save(fig, f'新验证{i:02d}_轨迹对照')
    solver = FeedbackFour(RobotClient(LocalEnvironment([]), 'local-robot'), FeedbackConfig())
    state = solver.channels[1]
    state.status = 'found'
    state.region.observe([0, 0], 'direction', 0.)
    state.region.observe([700, 200], 'no_signal')
    state.measured_sites = [[0, 0], [700, 200]]
    hypotheses = joint_hypotheses(state)
    plan = solver._observation_plan(state, np.array([700., 200.]))
    xs, ys = np.linspace(-200, 1550, 71), np.linspace(-600, 600, 45)
    values = np.array([[reception_support(hypotheses, np.array([x, y])) for x in xs] for y in ys])
    fig, ax = plt.subplots(figsize=(11, 7))
    mesh = ax.pcolormesh(xs, ys, values, vmin=0, vmax=1, cmap='YlGnBu', shading='auto')
    fig.colorbar(mesh, ax=ax, label='启发式接收支持度（不是实测概率）')
    v = np.vstack((state.region.vertices, state.region.vertices[0]))
    ax.plot(v[:, 0], v[:, 1], color='#8b4f85', lw=1.5, label='保留的连续位置外包')
    for p, marker, color, label in [([0, 0], 's', '#283e50', '有效接收站'), ([700, 200], 'x', '#d34b45', '无信号站'),
                                   (plan['位置'], '^', '#f2a33a', '重新评分后的补测候选'), ([300, 0], 'o', '#d8e870', '构造反例的源位置')]:
        ax.scatter(*p, marker=marker, c=color, s=75, label=label, zorder=6)
    ax.annotate('', plan['位置'], (700, 200), arrowprops=dict(arrowstyle='->', color='#f2a33a', lw=2))
    ax.set(aspect='equal', xlabel='东向 x（米）', ylabel='北向 y（米）', title='近源背侧反例：保留完整外包，用正负方向反馈重排候选')
    ax.legend(loc='lower right', fontsize=9)
    fig.text(.5, .02, '本图为已知几何构造示意；源真值未参与候选计算，朝向弧与位置样本只用于排序。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, 1))
    save(fig, '03_方向反馈选点示意')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    def box(x, y, text, width=.25, height=.13, color='#e5f0f5'):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                   boxstyle='round,pad=.008', fc=color, ec='#547785', lw=1.4))
        ax.text(x, y, text, ha='center', va='center', fontsize=12)
    def arrow(a, b):
        ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', color='#547785', lw=1.6))
    box(.5, .87, '固定站扫描并更新频道状态\n保留连续位置外包', .48)
    for x, txt in ((.18, '已有单点安全证书\n按路线安排清除'), (.5, '少量格点联合覆盖\n比较完整插入与延后费用'), (.82, '已发现但未定位\n利用方向反馈选择补测站')):
        box(x, .60, txt)
        arrow((.5, .79), (x, .68))
        arrow((x, .52), (.5, .36))
    box(.5, .29, '在实际停点顺路检测\n补齐满足600米间隔的未知频道', .56, .14, '#f8eddc')
    arrow((.5, .21), (.5, .14))
    box(.5, .075, '重新规划；固定扫描后处理已知目标并退出\n13站不等于任意定向源全清证明', .72, .11)
    ax.set_title('第四问当前流程：方向反馈、顺路清除与未知频道扫描', fontsize=17, pad=12)
    fig.tight_layout()
    save(fig, '04_当前策略流程')


def table(groups):
    text = '| 阶段/设置 | 清除/目标 | 全清/局数 | 平均时间/秒 | 平均路程/公里 | 平均主补测无信号 |\n|---|---:|---:|---:|---:|---:|\n'
    for g in groups:
        text += f"| {g['阶段']}·{g['方法']} | {g['清除总数']}/{g['目标总数']} | {g['全清局数']}/{g['局数']} | {g['平均虚拟时间_秒']:.1f} | {g['平均路程_米']/1000:.3f} | {g['平均补测无信号次数']:.3f} |\n"
    return text


def documents(first, final):
    fresh = [g for g in final['分组汇总'] if g['阶段'] == '新验证']
    a, b = [next(g for g in fresh if g['方法'] == name) for name in (BASELINE, IMPROVED)]
    change = 100 * (1 - b['平均虚拟时间_秒'] / a['平均虚拟时间_秒'])
    pairs = '| 新验证场景 | 基线清除 | 改进清除 | 基线时间/秒 | 改进时间/秒 |\n|---|---:|---:|---:|---:|\n'
    common = []
    for i in range(8):
        x, y = [next(r for r in final['逐局'] if r['阶段'] == '新验证' and r['场景编号'] == i and r['方法'] == name) for name in (BASELINE, IMPROVED)]
        if x['清除数'] == y['清除数'] == x['目标数'] == y['目标数']:
            common.append((x['虚拟时间_秒'], y['虚拟时间_秒']))
        pairs += f"| [{i:02d} {x['场景名称']}](../../results/figures/第四问/改进版_方向反馈与顺路覆盖/新验证{i:02d}_轨迹对照.png) | {x['清除数']}/{x['目标数']} | {y['清除数']}/{y['目标数']} | {x['虚拟时间_秒']:.1f} | {y['虚拟时间_秒']:.1f} |\n"
    common_a, common_b = np.mean(common, axis=0)
    common_note = f'双方共同全清的{len(common)}场，平均时间由{common_a:.1f}降至{common_b:.1f}秒，减少{100 * (1 - common_b / common_a):.2f}%；其中{sum(y < x for x, y in common)}场更快。'
    text = f'''# 第四问改进版：方向反馈、顺路覆盖与未知频道补全

当前入口仍为`scripts/run_q4.py`，默认读取`configs/q4_feedback.json`。独立新8场验证：改进106/106、8/8全清；基线104/106、6/8全清。平均时间减少{change:.2f}%，从{a['平均虚拟时间_秒']:.1f}降至{b['平均虚拟时间_秒']:.1f}秒。这是本地构造证据，不是官方成绩，也不是任意定向源全清保证。

## 当前配置与运行

默认开启小范围顺路覆盖、方向反馈选点、完整任务预测及合格未知频道扫描补全；优先补测间距100米、绕行门槛800米、最多4个小范围覆盖点。同频道顺路检测的600米距离间隔、13固定站顺序、8次主补测预算、20米清除物理半径及末尾不自动外围搜索保持。

```powershell
Set-Location 'D:\\mywork\\code\\cumcm2026-b'
.\\.venv\\Scripts\\python.exe -X utf8 scripts/run_q4.py --mode http --robot-id '你的队号'
```

恢复本次改进前的100米几何版本：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts/run_q4.py --mode http --robot-id '你的队号' --strategy spacing
```

单独传`--probe-spacing 0`保留历史语义，恢复未增加间距的原几何版本。显式`--strategy feedback`优先；给旧配置文件且不显式指定策略时采用旧spacing类。`--detour-limit 600`可覆盖门槛，但本轮新验证没有据结果回调门槛。旧折线入口也转入当前策略，不恢复折线。

无官方接口的本地复核：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts/run_q4.py --mode local --seed 103003 --count 16 --layout boundary --radius mixed
.\\.venv\\Scripts\\python.exe -X utf8 scripts/evaluate_q4_feedback.py --stage 核验
.\\.venv\\Scripts\\python.exe -X utf8 scripts/evaluate_q4_discovery.py --stage 核验
.\\.venv\\Scripts\\python.exe -X utf8 scripts/build_q4_feedback_assets.py --verify
```

命令须从项目根目录执行。HTTP题号在模拟器中选择；Python不会替用户切换题号。较早文档中的“当前”指其生成时的版本，运行参数以本文及实际日志配置为准。

## 1. 小范围覆盖也可以顺路清除

位置外包半径超过19.5米时，不能冒充单点安全证书。新版对旋转包围盒最多4个20米网格点求完整访问顺序，以当前位置X、下一固定站F计算：

`额外路程 = |XQ1| + Σ|QiQi+1| + |QkF| − |XF|`

`最坏增加费用 = 额外路程/5 + 3(k−1) + 5`

只有整组绕行不超过门槛、预测不比延后更贵、且本路段剩余动作额度能容纳整组，才执行；一旦清除成功即停止。格点联合覆盖整个外包，成功不依赖方向接收。实际清除成功后保留顺路检测，其费用另计。较大区域继续用有限回退，不在中途大面积扫网格。

## 2. 用正负方向反馈重新评价候选

此前几何候选要求对整个位置外包都处在999.5米内；近处源可能已被越过，剩余候选仍在背侧。新版在已发现频道出现可排除超距的无信号后激活方向反馈：位置G与发射朝向u同时考虑，有效站P给`u·(P−G)≥0`；距离已保证的无信号站Q给`u·(Q−G)<0`。

在完整正观测外包中取最多41个代表位置，每个位置计算允许朝向的圆弧交集；收到正反馈的最远距离与1000米下界共同给半径下界，含距离歧义的负反馈按各假设分别判断。无可用样本时回原几何方法。

候选包含原几何方向附近的位置及最后有效接收站附近的位置，没有左右交替的固定次序。候选不再一律要求离整个外包都小于1000米；可容忍部分假设超距，但会用接收支持度降低其评分，并明确记录距离保证是否成立。仍优先保持100米实际站间距；不足时记录放宽。

评分为移动与检测成本，加“接收到方向后的采样后续费用”与“无信号后保守覆盖费用”的加权值；权重是位置样本和朝向弧的启发式支持度，未被校准为真实概率。连续位置外包不因该采样删减，所有单点清除仍检查完整外包；重复无信号后样本重算，8次预算和有限回退保证已发现任务仍有终止路径。

![方向反馈示意](../../results/figures/第四问/改进版_方向反馈与顺路覆盖/03_方向反馈选点示意.png)

## 3. 完整任务预测与路线

路线仍保持13固定站的先后顺序。未定位任务同时估计“去补测点→到可能目标代表位置清除→回下一站”的完整路程，以及放到后续固定路段的插入代价；只先执行一个动作，再按新反馈重规划。单次补测绕行仍受800米门槛约束；当完整任务超过门槛且预测延后更省路程时，延后处理。代表位置和后续费用均为预测，不能充当定位证书或全局最优证明。

## 4. 补齐真实停点的未知频道

第一轮800米版在旧验证场景02比基线多漏频道2、8。事后核查两源在13固定站均不可见，但新版已到达可接收它们的位置：对应频道距最近旧测量站分别827.4、816.9米，符合600米间隔，却被“最多扫描4频道”排在外面。

最终版先执行原最多4个候选的顺路检测，再把同一停点仍未发现、且距离所有该频道旧测量站至少600米的其他频道补齐。已清除频道不重测，已发现但未定位频道仍保留原限额规则。新增检测只发生在已经到达的补测或清除停点，没有加外围站或插入移动路径；检测与切换费用全部计入结果。

![当前流程](../../results/figures/第四问/改进版_方向反馈与顺路覆盖/04_当前策略流程.png)

## 5. 实验阶段与边界

第一阶段16个预设场景101000—101015，7种设置共112次。排名先少异常、少漏源、多全清，再比总时间，选中800米。第二阶段8个新场景102000—102007，共24次，原800米设置的速度改善伴随新增遗漏，故未直接接受为最终版。

随后根据可见停点漏扫的具体机制补齐未知频道，800米及其余参数固定。原8场再跑8次只作诊断复查；第三阶段8个全新场景103000—103007与基线配对，共16次，不据其结果重新调参。合计160次完整运行。每阶段保留全部结果，包括更慢、漏源及被修正的版本。

{table(first['分组汇总'])}

{table(final['分组汇总'])}

小范围清除单独启用并非总能改善：选参场景11改变沿途扫描后有新增漏源，因此不能凭局部50秒测算就宣称整局必然更优。最终采用组合机制并补齐合格未知频道，已将新增扫描开销计入新验证结果。

![逐场新验证](../../results/figures/第四问/改进版_方向反馈与顺路覆盖/02_新验证逐场费用与清除.png)

{pairs}

新验证场景02改进版仍更慢；场景06多清除1源但也更慢，均原样列出。{common_note}不能把未全清时的较短时间直接当作效率优势。本轮8/8不能推出任意发射朝向的连续发现保证，特别是固定站不可见且实际路径没有经过接收区的目标。

## 6. 复现、验证和保留内容

所有运行保存完整压缩JSON、场景、配置、源码SHA256及退出后真值；逐动作核验物理反馈、示向误差、移动/检测/切换/清除费用、外包包含真值、清除证书、固定站次序及同图性。安全外包、前三问和旧Q4核心未改；新模块分别为`q4_feedback_strategy.py`与`q4_discovery_strategy.py`。

全库270项自动测试通过（86.91秒），含新方向反馈、联合覆盖、完整任务费用及未知频道补全回归；当前CLI本地种子103003的16源边界混合场景清除16/16，6084.278秒，与新验证03归档一致。未连接官方模拟器。

图表共12张，各有PNG/SVG：消融选参、新验证费用、方向反馈示意、策略流程，以及全部8组新验证配对轨迹。原始官方日志和之前图表未覆盖；变更入口及核验器逐字节留档，历史哈希只接受当前文件或指定的精确档案。运行有现实规划时限，不同机器重跑可能改变实际评估数，历史动作重算才是精确审计。
'''
    REPORT.write_text(text, encoding='utf8', newline='\n')
    PAPER.write_text('# 第四问方向反馈与顺路覆盖：论文备用说明\n\n'
        '在13固定站和原连续位置外包基础上，加入少量格点联合覆盖的完整路线插入、位置/朝向联合假设评分，以及满足600米间隔的未知频道顺路扫描补全。方向假设仅用于规划排序；单点清除仍使用完整外包证书，多点清除则保证整组覆盖，未用离散样本冒充连续定位保证。\n\n'
        f'最终8场独立本地验证清除106/106、8场全清；基线104/106、6场全清。平均虚拟时间{a["平均虚拟时间_秒"]:.1f}降至{b["平均虚拟时间_秒"]:.1f}秒，降低{change:.2f}%；平均路程{a["平均路程_米"]/1000:.3f}降至{b["平均路程_米"]/1000:.3f}公里。新增未知频道扫描的费用已经计入。\n\n'
        + common_note + '\n\n' +
        '最初800米设置在旧验证集出现新增遗漏，查明实际可接收停点被四频道上限漏扫后修正。旧集重跑属于诊断复查，最终结论来自另一个新验证集；全部160次实验均保存。较少回头或更低平均时间不能替代发现数量评价，8场全清不构成任意场景全清证明；本轮均为本地构造，不是官方成绩。\n\n'
        '[完整方法、流程和全部配对图](../../docs/第四问/改进版_方向反馈与顺路覆盖.md)。\n', encoding='utf8', newline='\n')


def manifest():
    paths = list(FIRST_RUNS.rglob('*.json.gz')) + list(RUNS.rglob('*.json.gz')) + list(FIG.glob('*'))
    paths += [FIRST_TABLE, TABLE, REPORT, PAPER, Path(__file__), ROOT / 'configs/q4_feedback.json',
              ROOT / 'scripts/run_q4.py', ROOT / 'scripts/run_q4_zigzag.py', ROOT / 'scripts/build_q4_zigzag_assets.py',
              ROOT / 'scripts/build_q4_inner13_assets.py', ROOT / 'tests/test_q4_feedback.py', ROOT / 'tests/test_q4_discovery.py']
    paths += [ROOT / path for path in sources()]
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    first, final = read(FIRST_TABLE), read(TABLE)
    assert len(first['逐局']) == 136 and len(final['逐局']) == 24
    if args.verify:
        assert read(MANIFEST) == manifest()
        for path in FIG.glob('*'):
            if path.suffix == '.png':
                with Image.open(path) as im:
                    im.verify()
            elif path.suffix == '.svg':
                ET.parse(path)
        print('改进版160次运行及12张中文图文的来源与文件结构核验通过。')
        return
    figures(first, final)
    documents(first, final)
    dump(MANIFEST, manifest())
    print('已生成12张中文PNG/SVG、完整方法、运行说明和论文备用资料。')


if __name__ == '__main__':
    main()
