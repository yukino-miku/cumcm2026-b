"""几何补测间距试验的中文图文归档；仅重读本地结果。"""
import argparse
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from evaluate_q4_spacing import ROOT, TABLE, RUNS, read, dump, sources
from build_q4_inner13_assets import setup_font
from build_q4_zigzag_assets import trajectory

FIG=ROOT/'results/figures/第四问/撤回折线_几何间距'
REPORT=ROOT/'docs/第四问/撤回折线_几何补测间距.md'
PAPER=ROOT/'paper/sections/第四问_几何补测间距_试验备用说明.md'
MANIFEST=ROOT/'results/tables/第四问/几何间距_文件校验.json'


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ['.png','.svg']:
        p=FIG/(name+ext)
        fig.savefig(p,dpi=160,metadata={'Date':None} if ext=='.svg' else None)
        if ext=='.svg':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    plt.close(fig)


def figures(data):
    setup_font();plt.rcParams['svg.hashsalt']='q4-geometric-spacing'
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,stage,total in zip(axes,['选参','验证'],[212,106]):
        groups=[g for g in data['分组汇总'] if g['阶段']==stage]
        labels=['原版' if g['间距_米']==0 else str(g['间距_米'])+'米' for g in groups]
        for ax,key,title in zip(row,['平均虚拟时间_秒','清除总数'],['平均虚拟时间（秒）',f'实际清除数 / {total}']):
            values=[g[key] for g in groups]
            bars=ax.bar(labels,values,color=['#397da0']+['#df964b']*(len(groups)-1))
            ax.bar_label(bars,fmt='%.1f',padding=3)
            ax.set_title(stage+'场景：'+title);ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.12)
    fig.suptitle('原几何补测拉开间距：发现收益与时间代价',fontsize=16)
    fig.text(.5,.012,'16场×4设置选参，8场×2设置验证；100米只作温和试用值，并非所测场景总体时间最优。均为本地构造。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.95));save(fig,'01_间距与发现耗时')
    for i in [4,1]:
        fig,axes=plt.subplots(1,2,figsize=(14,7.5))
        for ax,gap in zip(axes,[0,100]):
            result=read(RUNS/'验证'/f'间距{gap}-{i:02d}.json.gz')
            trajectory(ax,result,f'验证{i:02d} '+('原几何' if gap==0 else '几何间距100米'))
        fig.suptitle(f'验证{i:02d}相同地图：'+('解释额外发现的目标' if i==4 else '展示同样全清时的耗时增加'),fontsize=16)
        fig.text(.5,.022,'紫线为扫描后收尾；橙三角为主补测；绿点为成功清除；红叉为遗漏源；细箭头为事后发射轴。\n本地构造，真值未提供给策略；两图用于诊断，全部8场结果另见报告。',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.14,1,.95));save(fig,f'轨迹对照_验证{i:02d}')


def documents(data):
    summary='| 阶段 | 间距/米 | 清除/目标 | 全清/局数 | 平均时间/秒 | 平均路程/公里 | 平均主补测无信号 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for g in data['分组汇总']:
        summary+=f"| {g['阶段']} | {g['间距_米']} | {g['清除总数']}/{g['目标总数']} | {g['全清局数']}/{g['局数']} | {g['平均虚拟时间_秒']:.1f} | {g['平均路程_米']/1000:.3f} | {g['平均补测无信号次数']:.3f} |\n"
    pairs='| 验证场景 | 原版清除 | 100米清除 | 原版时间/秒 | 100米时间/秒 |\n|---|---:|---:|---:|---:|\n'
    for i in range(8):
        a,b=[next(r for r in data['逐局'] if r['阶段']=='验证' and r['场景编号']==i and r['间距_米']==gap) for gap in [0,100]]
        pairs+=f"| {i:02d} {a['场景名称']} | {a['清除数']}/{a['目标数']} | {b['清除数']}/{b['目标数']} | {a['虚拟时间_秒']:.1f} | {b['虚拟时间_秒']:.1f} |\n"
    text=r'''# 撤回折线，恢复原几何补测并试用100米间距

用户已取消折线方案。**当前仍使用原几何候选与角采样费用评分，试用“优先拉开100米”的补测间距。** 原命令`scripts/run_q4.py`继续使用；旧折线入口也转入当前几何方案。折线模块、216次实验及图表仅作历史复核，不重新启用折线。

## 1. 本次改动

用户说的“增大两个观测点距离”，本轮按“把主动补测站放得更分散、让机器狗实际移动到不同位置”处理。不是增大600米同频道顺路扫描间隔，也不是把400米绕行门槛改大。

原方案在候选几何筛选中只避免该频道与旧站相差不足0.01米的重复坐标，并没有100米最小间距。现在从原候选集合中，优先保留同时满足以下条件的点P：

$$|P-X|\ge b,\qquad |P-S_{\mathrm{last}}|\ge b,\qquad b=100\ \mathrm{米}.$$

X为机器狗当前位置，S_last为**该频道上一次检测位置**。约束不是所有频道所有历史站的两两距离，也不对13个固定站强制重排。当前站与该频道上次站有时不同，因此两个条件都要检查。

筛选后仍按原移动成本加角采样后续成本评分，不指定左右交替，不沿单条射线固定步进。保留原正观测外包、候选距离上限999.5米、近区过滤、8次主补测预算及有限光学覆盖。

若当前没有满足间距的候选，放宽间距并回到原候选集合，记录`间距约束已放宽=true`。这是一项优先规则，**不保证每段都至少100米**；不能为了拉开站点而直接放弃已发现目标的可行定位机会。候选仍须通过400米单次绕行门槛，不顺路时继续延后。

```mermaid
flowchart TD
 A[原几何候选集合与有效性检查] --> B[检查距当前机器狗及该频道上次站均不少于100米]
 B --> C{是否还有候选}
 C -->|有| D[对筛选后的原候选作原角采样费用评分]
 C -->|没有| E[记录放宽间距 使用原候选集合]
 E --> D
 D --> F[原400米单次绕行检查]
 F -->|可顺路执行| G[补测并在同一位置顺路扫描其他频道]
 F -->|不顺路| H[延后目标 前往下一固定站]
```

13个固定站和11.4公里纯遍历、停止已定位频道的重复固定检测、每轮连续两次无信号延后、顺路扫描最多4个其他频道及600米间隔、19.5米安全清除证书、结束规则均保持。没有新增未知源外围搜索，也没有改动前三问运行代码。

## 2. 80次本地对照及选值依据

16场种子98000—98015，分别比较原版（间距0）及100、200、300米，共64次；8场新种子99000—99007比较原版与100米，共16次。沿用均匀、边界、聚集、中心以及全向/定向混合等8种预设布局。每个场景的目标位置、频道、半径、发射轴和误差生成规则相同，策略只读反馈，退出后才读取真值用于评价。

16场选参按异常、漏源、全清数、平均时间的顺序，**总体仍选中0，即不增加间距的原版**。100米是非零候选中时间最小的温和值，因此选它进入8场新场景验证；没有用验证结果重新调节100/200/300米参数。

__SUMMARY__

16场训练构造中，各设置均清除207/212个源；100米平均比原版慢86.2秒，没有证明发现率增加。新8场验证中，100米清除103/106，原版102/106，多清除一个；全清从5场变6场，但平均时间从6217.8增至6347.8秒，多约130.0秒，路程多约613米。

按照用户本次希望适度扩大沿途发现机会的偏好，**把100米作为试用默认值，是一次明确的发现与耗时取舍，不是统计证明的最优参数**。这24个构造不足以证明总体发现概率提高，更不能保证任意定向源全清；如果优先保留原时间策略，可直接把间距设为0。

![间距对照](../../results/figures/第四问/撤回折线_几何间距/01_间距与发现耗时.png)

## 3. 全部独立验证结果与轨迹

__PAIRS__

![新增发现示例](../../results/figures/第四问/撤回折线_几何间距/轨迹对照_验证04.png)

![同样全清但更慢示例](../../results/figures/第四问/撤回折线_几何间距/轨迹对照_验证01.png)

两组轨迹按用途选择用于解释收益与代价，没有把它们当作随机代表。完整80份运行压缩JSON保存在`results/models/第四问/几何补测间距对照/`，[逐局汇总](../../results/tables/第四问/几何补测间距对照.json)保留未全清案例。所有已发现源最终均清除；遗漏仍发生在未发现阶段。

## 4. 运行方法

在模拟器选择第四问演练后，由用户自行运行：

```powershell
Set-Location 'D:\mywork\code\cumcm2026-b'
.\.venv\Scripts\python.exe -X utf8 scripts/run_q4.py --mode http --robot-id '你的队号'
```

默认读取`configs/q4_geometric_spacing.json`，优先补测间距100米，绕行门槛400米。原`configs/q4_inner13.json`仍逐字节保留；旧折线配置不应再传入当前入口。

完全恢复未加间距的原几何选点：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_q4.py --mode http --robot-id '你的队号' --probe-spacing 0
```

也可用`--probe-spacing 200`手动比较；本轮没有证据说明200或300米优于100米。无须更改源码，也不需要恢复折线。

本地复核，不连接模拟器：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_q4.py --mode local --seed 99000 --count 10
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_q4_spacing.py --stage 核验
.\.venv\Scripts\python.exe -X utf8 scripts/build_q4_spacing_assets.py --verify
```

已有选参/验证命令读取归档、补跑缺失局，并拒绝混用不同运行源码。原几何候选处理有0.8秒规划上限，不同硬件重新执行可能改变实际评估数量；精确历史复核采用原动作重算，不声称只凭随机种子就能逐动作完全复现。

## 5. 归档和验证

原Q4核心`q4_strategy.py`和第三问核心保持不变，间距扩展单独实现于`q4_spacing_strategy.py`，0米直接委托原几何函数。旧运行与图表来源涉及入口变更的文件，逐字节保存在`docs/第四问/历史代码/`；核验器只允许当前文件或明确映射的历史源码与原SHA256完全匹配，不重写历史运行、原始演练或旧图表。

本轮没有调用官方模拟器。结构测试、临时本机HTTP入口测试、80次动作/外包/清除/费用/间距复核，以及旧图表来源兼容核验均应完成后再使用本试用版；具体通过记录见项目工作日志。
'''
    text=text.replace('__SUMMARY__',summary).replace('__PAIRS__',pairs)
    REPORT.write_text(text,encoding='utf8',newline='\n')
    PAPER.write_text('''# 几何补测间距：中文试验备用说明

本轮撤回折线补测，恢复原有限几何候选和角采样费用评分。在原候选中增加优先间距规则：下一补测站距机器狗当前位置及该频道上一次检测站均至少100米；无满足条件候选时恢复原候选集合。固定扫描、路线门槛和同频道顺路检测间隔保持不变。

'''+summary+'''
16场四设置对照中，100、200、300米均没有增加总清除数，原版平均时间最低。100米是非零设置中耗时最小者，在另外8场对照中多清除一个源，但平均增加约130秒。因此100米仅作为侧重沿途发现机会的温和试用值，不得写为全局最优或总体发现概率已获提升。全部80次均为本地构造，非官方成绩，未全清场景均保留。

公式、全部验证结果、费用及轨迹见[撤回折线后的当前说明](../../docs/第四问/撤回折线_几何补测间距.md)。此前折线试验仅作为弃用方法的历史材料。
''',encoding='utf8',newline='\n')


def manifest():
    paths=list(RUNS.rglob('*.json.gz'))+list(FIG.glob('*'))+[TABLE,REPORT,PAPER,Path(__file__),
        ROOT/'scripts/evaluate_q4_spacing.py',ROOT/'scripts/run_q4.py',ROOT/'scripts/run_q4_zigzag.py',
        ROOT/'configs/q4_geometric_spacing.json',ROOT/'scripts/q4_archive_sources.py',
        ROOT/'scripts/build_q4_inner13_assets.py',ROOT/'scripts/build_q4_zigzag_assets.py',
        ROOT/'tests/test_q4_spacing.py',ROOT/'tests/test_q4_zigzag.py']+[ROOT/p for p in sources()]
    return {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');args=p.parse_args()
    data=read(TABLE);assert len(data['逐局'])==80
    if args.verify:
        assert read(MANIFEST)==manifest()
        for p in FIG.glob('*'):
            if p.suffix=='.png':
                with Image.open(p) as im:im.verify()
            else:ET.parse(p)
        print('几何间距图文及来源核验通过。');return
    figures(data);documents(data);dump(MANIFEST,manifest())
    print('已生成3张中文PNG/SVG、结果说明与论文备用稿。')


if __name__=='__main__':main()
