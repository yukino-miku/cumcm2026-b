#!/usr/bin/env python
"""方案二完整本地对照与中文图表：同场景五设置，不连接官方模拟器。"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from build_q3_scheme1_assets import check_run, style, save, LAYOUTS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import matplotlib
import numpy as np
import scipy
from cumcm2026_b.q3_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_strategy import SchemeOne, StrategyConfig
from cumcm2026_b.q3_scheme2 import SchemeTwo, SchemeTwoConfig, layout_spec
from cumcm2026_b.q3_geometry import search_stations

ARCHIVE = ROOT/"experiments/第三问/方案二/完整对照记录.jsonl"
SUMMARY = ROOT/"results/tables/第三问/方案二_完整闭环对照汇总.json"
VARIANTS = {
    "scheme1": "方案一默认",
    "main14": "方案二14站主布局",
    "main14_defer": "方案二14站延后清除",
    "margin14": "方案二14站10米余量",
    "compact15": "方案二15站小余量",
}
SHORT = ["方案一\n默认", "方案二\n14站主布局", "方案二\n14站延后清除", "方案二\n14站10米余量", "方案二\n15站小余量"]
FIGURES = ["方案二_完整轨迹与阶段", "方案二_两方案及布局对照", "方案二_扫描精度与后续费用", "方案二_执行流程图"]


def source_paths():
    names = ["q1_geometry.py", "q2_physical_geometry.py", "q3_geometry.py", "q3_local_env.py",
             "q3_protocol.py", "q3_strategy.py", "q3_scheme2.py"]
    return [ROOT/"src/cumcm2026_b"/name for name in names]+[
        ROOT/"configs/q3_scheme1.json",ROOT/"configs/q3_scheme2.json",Path(__file__),
        ROOT/"scripts/build_q3_scheme1_assets.py"]


def run_suite(count):
    one = json.loads((ROOT/"configs/q3_scheme1.json").read_text(encoding="utf-8"))
    two = json.loads((ROOT/"configs/q3_scheme2.json").read_text(encoding="utf-8"))
    records = []
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    with ARCHIVE.open("w",encoding="utf-8",newline="\n") as stream:
        for i in range(count):
            case = {"seed":20260911+i, "count":[10,13,16][i%3], "layout":list(LAYOUTS)[i%4],
                    "radius_mode":["min","max","mixed"][(i//3)%3],
                    "error_mode":["hash","zero","plus","minus","alternating"][i%5]}
            for variant in VARIANTS:
                env = make_case(**case)
                client = RobotClient(env,"local-robot")
                if variant == "scheme1":
                    strategy = SchemeOne(client,StrategyConfig(**one))
                else:
                    settings = {**two, "layout":"main14" if variant == "main14_defer" else variant,
                                "scan_clear": variant != "main14_defer"}
                    strategy = SchemeTwo(client,SchemeTwoConfig(**settings))
                result = strategy.run()
                truth = env.truth_for_evaluation()
                checked = check_run(result,truth)
                record = {"场景序号":i,"构造参数":case,"设置":variant,"设置说明":VARIANTS[variant],
                          "真值_仅评估器读取":truth,"结果":result,"外包真值包含检查次数":checked}
                records.append(record)
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush()
                client.close()
            if (i+1)%5 == 0: print(f"已完成{i+1}/{count}个同场景五设置对照，均已全清",flush=True)
    return records


def summarize(records):
    stats = {}
    for variant in VARIANTS:
        results = [r["结果"] for r in records if r["设置"] == variant]
        stats[variant] = {"名称":VARIANTS[variant],"运行数":len(results),"完整清除次数":sum(r["运行成功"] for r in results),
            **{f"平均{k}":float(np.mean([r[k] for r in results])) for k in
               ["虚拟总时间_秒","总路程_米","检测次数","频道切换次数","程序运行时间_秒"]},
            "逐场平均每目标时间的均值_秒":float(np.mean([r["平均定位清除时间_秒"] for r in results])),
            "平均追加测向次数":float(np.mean([sum(c["追加测向次数"] for c in r["频道记录"]) for r in results])),
            "最大虚拟总时间_秒":max(r["虚拟总时间_秒"] for r in results),
            "清除失败总次数":sum(r["清除失败次数"] for r in results)}
        if variant != "scheme1":
            scans = [r["扫描阶段"] for r in results]
            stats[variant].update({"平均扫描阶段时间_秒":float(np.mean([s["实际虚拟时间_秒"] for s in scans])),
                "平均扫描阶段清除数":float(np.mean([s["清除数"] for s in scans])),
                "扫描结束待清除总数":sum(s.get("待清除数",0) for s in scans),
                "扫描结束需继续缩小外包总数":sum(s.get("需继续缩小外包数",0) for s in scans)})
    pairs = {}
    baseline = [r["结果"]["虚拟总时间_秒"] for r in records if r["设置"] == "scheme1"]
    for variant in list(VARIANTS)[1:]:
        times = [r["结果"]["虚拟总时间_秒"] for r in records if r["设置"] == variant]
        delta = np.array(times)-baseline
        pairs[variant] = {"比方案一更快场景数":int(np.sum(delta < -1e-6)),"更慢场景数":int(np.sum(delta > 1e-6)),
                         "持平场景数":int(np.sum(np.abs(delta) <= 1e-6)), "平均差值_秒":float(delta.mean()),
                         "平均总时间变化百分比":float((np.mean(times)/np.mean(baseline)-1)*100)}
    return stats,pairs


def plot_assets(records):
    style()
    plt.rcParams["svg.hashsalt"] = "q3-scheme2"
    count = len(records)//len(VARIANTS)
    notice = f"本地构造，非官方演练或正式成绩；统计仅描述{count}个指定场景"
    def finish(fig,name,rect=(0,.08,1,.92)):
        fig.text(.5,.015,notice,ha="center",fontsize=9,color="#586A73")
        fig.tight_layout(rect=rect)
        save(fig,name)
    pair = [next(r for r in records if r["场景序号"] == 0 and r["设置"] == v) for v in ("scheme1","main14")]
    fig,axes = plt.subplots(1,2,figsize=(14,7.8))
    for ax,record in zip(axes,pair):
        result = record["结果"]
        events = result["动作记录"]
        end = next((e["步骤"] for e in events if e["动作"] == "固定扫描完成"),None)
        points = [[0.,0.]]
        for e in events:
            if e["动作"] in {"检测","清除"}:
                old = np.asarray(points[-1]); new = np.asarray(e["位置"])
                ax.plot([old[0],new[0]],[old[1],new[1]],lw=1,alpha=.65,
                        c="#C77B39" if end is not None and e["步骤"] > end else "#347CA0")
                points.append(e["位置"])
        ax.plot([],[],c="#347CA0",label="方案一完整路线 / 方案二固定扫描")
        if end is not None: ax.plot([],[],c="#C77B39",label="方案二后续定位清除")
        fixed = search_stations() if end is None else np.array(result["布局"]["站点"])
        ax.scatter(fixed[:,0],fixed[:,1],c="#173E50",marker="s",s=25,label="固定扫描站",zorder=5)
        for i,p in enumerate(fixed): ax.annotate(str(i),p,xytext=(5,7),textcoords="offset points",fontsize=8)
        truth = record["真值_仅评估器读取"]["目标"]
        xy = np.array([s["位置"] for s in truth])
        ax.scatter(xy[:,0],xy[:,1],marker="*",c="#AF4936",s=90,label="真实目标（离线核验）",zorder=5)
        ax.add_patch(Circle((0,0),1800,fill=False,lw=1.5,color="#79528B",label="1800米目标边界"))
        ax.set(xlim=(-2050,2050),ylim=(-2050,2050),aspect="equal",xlabel="东向坐标 x（米）",ylabel="北向坐标 y（米）")
        ax.grid(alpha=.15)
        ax.set_title(f"{record['设置说明']}\n{result['清除数']}个全清；虚拟时间{result['虚拟总时间_秒']:.1f}秒")
    handles,labels = axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.045),ncol=3,fontsize=9)
    fig.suptitle("同一场景的完整行动轨迹（固定使用场景0）",fontsize=15)
    finish(fig,FIGURES[0],rect=(0,.20,1,.92))

    stats,pairs = summarize(records)
    fig,axes = plt.subplots(1,2,figsize=(14,6.8))
    means = [stats[v]["平均虚拟总时间_秒"] for v in VARIANTS]
    axes[0].bar(range(5),means,color=["#347CA0"]+["#C77B39","#B49873","#A97947","#668F77"])
    for i,value in enumerate(means): axes[0].text(i,value+40,f"{value:.1f}",ha="center",fontsize=9)
    axes[0].set_xticks(range(5),SHORT,fontsize=9)
    axes[0].set(ylabel="平均虚拟总时间（秒）",ylim=(0,max(means)*1.15),title="完整任务费用，包含扫描、精定位与清除")
    off = np.array([r["结果"]["虚拟总时间_秒"] for r in records if r["设置"] == "scheme1"])
    on = np.array([r["结果"]["虚拟总时间_秒"] for r in records if r["设置"] == "main14"])
    delta = on-off
    axes[1].bar(range(count),delta,color=np.where(delta < 0,"#348D7F","#C77B39"))
    axes[1].axhline(0,color="#293D48",lw=1)
    axes[1].set(xlabel="场景序号",ylabel="方案二主布局－方案一（秒）",title="逐场成对差值：负值表示方案二更快")
    fig.suptitle("两方案及方案二参数对照",fontsize=15)
    finish(fig,FIGURES[1])

    fig,axes = plt.subplots(1,2,figsize=(14,6.8))
    main = [r for r in records if r["设置"] == "main14"]
    for r in main:
        values = [s["外包覆盖圆半径_米"] for s in r["结果"]["扫描阶段"].get("频道快照",[]) if s["状态"] == "found"]
        axes[0].scatter([r["场景序号"]]*len(values),values,s=18,alpha=.55,c=["#C77B39" if v > 19.5 else "#348D7F" for v in values])
    axes[0].axhline(19.5,ls="--",c="#A44942",label="实现清除判据19.5米（预留0.5米）")
    axes[0].set(yscale="log",xlabel="场景序号",ylabel="保守外包最小覆盖圆半径（米，对数轴）",
                title="14站主布局扫描后，尚未清除目标的精度")
    axes[0].legend(fontsize=8,loc="upper left")
    axes[0].grid(alpha=.15)
    variants = list(VARIANTS)[1:]
    scans = [stats[v]["平均扫描阶段时间_秒"] for v in variants]
    tails = [stats[v]["平均虚拟总时间_秒"]-s for v,s in zip(variants,scans)]
    axes[1].bar(range(4),scans,color="#347CA0",label="固定扫描阶段（含原地清除）")
    axes[1].bar(range(4),tails,bottom=scans,color="#C77B39",label="扫描后精定位与清除")
    axes[1].set_xticks(range(4),SHORT[1:],fontsize=9)
    axes[1].set(ylabel="平均虚拟时间（秒）",title="实际阶段费用，包含省略检测与清除动作",
                ylim=(0,max(stats[v]["平均虚拟总时间_秒"] for v in variants)*1.22))
    axes[1].legend(fontsize=8,loc="upper left")
    fig.suptitle("双重接收之后，还需要多少定位与清除工作",fontsize=15)
    finish(fig,FIGURES[2])

    fig,ax = plt.subplots(figsize=(13,7))
    ax.set(xlim=(0,13),ylim=(0,7)); ax.axis("off")
    boxes = [(0.4,4.6,"进入场景\n核对剩余时间与协议"),(4.65,4.6,"按顺序访问固定站\n检测未完成频道"),(8.9,4.6,"累计完整历史\nnear或原地证书清除"),
             (8.9,1.3,"固定扫描结束验收\n未清除目标至少两处接收"),(4.65,1.3,"逐源安全清除或追加测向\n无进展则有限光学覆盖"),(0.4,1.3,"全部清除或覆盖排除\n退出并保存完整日志")]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),3.7,1.4,boxstyle="round,pad=0.12",fc="#E6F0F2",ec="#34748C",lw=1.3))
        ax.text(x+1.85,y+.7,label,ha="center",va="center",fontsize=12)
    for a,b in [((4.1,5.3),(4.52,5.3)),((8.35,5.3),(8.77,5.3)),((10.75,4.42),(10.75,2.86)),
                ((8.78,2),(8.48,2)),((4.5,2),(4.24,2))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,color="#34748C",lw=1.5))
    ax.text(10.5,3.6,"全部固定站\n扫描完毕",ha="right",va="center",fontsize=10)
    ax.add_patch(FancyArrowPatch((10.75,6.15),(6.5,6.15),connectionstyle="arc3,rad=.2",
                                arrowstyle="-|>",mutation_scale=16,color="#34748C",lw=1.3))
    ax.text(8.6,6.72,"还有固定站：继续扫描",ha="center",fontsize=10)
    ax.text(6.5,.35,"所有动作串行计费；预算不足或证书异常如实记录未完成；目标真值只供运行后评估",ha="center",fontsize=10,color="#586A73")
    fig.suptitle("方案二执行流程：固定双覆盖扫描 → 后续精定位清除",fontsize=15)
    finish(fig,FIGURES[3],rect=(0,.05,1,.95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only",action="store_true")
    parser.add_argument("--cases",type=int,default=30)
    args = parser.parse_args()
    if args.cases < 4: parser.error("至少4个场景")
    if args.plots_only:
        previous = json.loads(SUMMARY.read_text(encoding="utf-8"))
        for relative,value in previous["来源"].items():
            path = ROOT/relative
            if path.resolve() != Path(__file__).resolve() and sha256(path.read_bytes()).hexdigest() != value:
                raise RuntimeError("计算来源已变化，必须完整重算")
        if sha256(ARCHIVE.read_bytes()).hexdigest() != previous["完整记录SHA256"]:
            raise RuntimeError("归档哈希不符")
        records = [json.loads(line) for line in ARCHIVE.read_text(encoding="utf-8").splitlines()]
    else:
        records = run_suite(args.cases)
    plot_assets(records)
    stats,pairs = summarize(records)
    report = {"说明":"本地构造同场景五设置比较，非官方成绩，非全局最优证明", "统计":stats,"与方案一成对比较":pairs,
        "布局证明参数":[layout_spec(name) for name in ("main14","margin14","compact15")],
        "完整记录":ARCHIVE.relative_to(ROOT).as_posix(),"完整记录SHA256":sha256(ARCHIVE.read_bytes()).hexdigest(),
        "来源":{p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in source_paths()},
        "外包真值包含检查总数":sum(r["外包真值包含检查次数"] for r in records),
        "环境":{"Python":platform.python_version(),"NumPy":np.__version__,"SciPy":scipy.__version__,"Matplotlib":matplotlib.__version__},
        "逐场摘要":[{"场景序号":r["场景序号"],"构造参数":r["构造参数"],"设置":r["设置"],
            **{k:r["结果"][k] for k in ["运行成功","清除数","虚拟总时间_秒","检测次数","总路程_米","清除失败次数"]}} for r in records]}
    SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"统计":stats,"与方案一成对比较":pairs},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__": main()
