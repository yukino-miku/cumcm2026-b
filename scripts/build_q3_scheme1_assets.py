#!/usr/bin/env python
"""运行方案一本地成对构造比较，生成可追溯记录与中文图；不连接官方接口。"""
from __future__ import annotations
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
import numpy as np
import scipy
from cumcm2026_b.q3_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_strategy import SchemeOne, StrategyConfig
from cumcm2026_b.q3_geometry import min_distance, minimum_circle, search_stations

ARCHIVE = ROOT/"experiments/第三问/方案一/本地对照完整记录.jsonl"
SUMMARY = ROOT/"results/tables/第三问/方案一_本地对照汇总.json"
FIGURES = ROOT/"results/figures/第三问"
NOTICE = "本地构造验证，非官方演练或正式成绩；统计仅描述这30个指定场景"
LAYOUTS = {"uniform":"均匀散布", "boundary":"圆周边缘", "cluster":"外围聚集", "center":"中心聚集"}


def check_run(result, truth):
    if not result["运行成功"] or not truth["全部清除"]:
        raise RuntimeError(f"本地构造未全部完成：{result['异常']}")
    if result["清除数"] != truth["目标数"]: raise RuntimeError("清除统计不一致")
    positions = {x["频道"]:x["位置"] for x in truth["目标"]}
    if set(result["已证明不存在频道"]) & set(positions): raise RuntimeError("错误排除存在频道")
    checked = 0
    for e in result["动作记录"]:
        if "外包顶点" in e:
            if min_distance(e["外包顶点"], positions[e["频道"]]) > 1e-5:
                raise RuntimeError("真实目标被外包错误排除")
            checked += 1
        if e["动作"] == "清除" and e["具有覆盖证书"] and e["结果"] != "success":
            raise RuntimeError("安全清除证书失效")
    if abs(sum(result["时间分项_秒"].values())-result["虚拟总时间_秒"]) > .001:
        raise RuntimeError("费用分项与虚拟时间不一致")
    return checked


def run_suite(count):
    records = []
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    # 构建输出属于可再生本地构造；不涉及真实接口日志。
    with ARCHIVE.open("w",encoding="utf-8",newline="\n") as stream:
        for i in range(count):
            scenario = {"seed":20260911+i,"count":[10,13,16][i%3],"layout":list(LAYOUTS)[i%4],
                        "radius_mode":["min","max","mixed"][(i//3)%3],
                        "error_mode":["hash","zero","plus","minus","alternating"][i%5]}
            for opportunistic in (False,True):
                env = make_case(**scenario)
                client = RobotClient(env,"local-robot")
                config = StrategyConfig(opportunistic=opportunistic)
                result = SchemeOne(client,config).run()
                truth = env.truth_for_evaluation()
                checked = check_run(result,truth)
                record = {"场景序号":i,"构造参数":scenario,"顺路检测":opportunistic,
                          "真值_仅评估器读取":truth,"结果":result,"外包真值包含检查次数":checked}
                records.append(record)
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush()
            if (i+1)%5 == 0: print(f"已完成 {i+1}/{count} 个场景，两种设置均完整清除",flush=True)
    return records


def style():
    font=Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=str(font)).get_name(),
        "font.size":10,"axes.unicode_minus":False,"svg.fonttype":"path","svg.hashsalt":"q3-scheme1",
        "figure.facecolor":"white","savefig.facecolor":"white"})


def save(fig,name):
    FIGURES.mkdir(parents=True,exist_ok=True)
    buffer=BytesIO()
    fig.savefig(buffer,format="png",dpi=190,bbox_inches="tight")
    (FIGURES/f"{name}.png").write_bytes(buffer.getvalue())
    path=FIGURES/f"{name}.svg"
    fig.savefig(path,bbox_inches="tight",metadata={"Date":None})
    path.write_text("\n".join(x.rstrip() for x in path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
    plt.close(fig)


def make_plots(records):
    style()
    pair=records[:2]
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.6))
    for ax,record in zip(axes,pair):
        result=record["结果"]
        events=[e for e in result["动作记录"] if e["动作"] in {"检测","清除"}]
        path=np.array([[0,0]]+[e["位置"] for e in events])
        ax.plot(path[:,0],path[:,1],lw=1.1,alpha=.6,color="#4E7D9B",label="实际动作路线")
        sources=record["真值_仅评估器读取"]["目标"]
        xy=np.array([s["位置"] for s in sources])
        ax.scatter(xy[:,0],xy[:,1],marker="*",s=90,color="#D1742F",label="真实目标（离线核验）",zorder=5)
        for s in sources:ax.annotate(str(s["频道"]),s["位置"],xytext=(5,6),textcoords="offset points",fontsize=8)
        anchors=search_stations()
        ax.scatter(anchors[:,0],anchors[:,1],marker="s",s=30,c="#184758",label="七个搜索站",zorder=4)
        for i,point in enumerate(anchors):ax.annotate(f"S{i}",point,xytext=(6,10) if i==3 else (6,-13),textcoords="offset points",fontsize=8)
        cleared=np.array([e["位置"] for e in events if e["动作"]=="清除" and e["结果"]=="success"])
        ax.scatter(cleared[:,0],cleared[:,1],marker="o",facecolors="none",edgecolors="#098272",s=55,label="成功清除位置",zorder=6)
        ax.add_patch(Circle((0,0),1800,fill=False,color="#70418B",lw=1.8,label="1800米目标边界"))
        ax.set(aspect="equal",xlabel="东向坐标 x（米）",ylabel="北向坐标 y（米）")
        ax.grid(alpha=.15)
        ax.set_title(("开启" if record["顺路检测"] else "关闭")+f"顺路检测\n清除{result['清除数']}个；虚拟时间{result['虚拟总时间_秒']:.1f}秒")
    axes[0].legend(fontsize=8,loc="upper left")
    fig.suptitle("方案一：同一构造场景的完整运动轨迹与清除结果",fontsize=15)
    fig.text(.5,.015,NOTICE,ha="center",fontsize=9,color="#586A73")
    fig.tight_layout(rect=(0,.05,1,.93))
    save(fig,"方案一_运动轨迹对照")

    fig,axes=plt.subplots(1,2,figsize=(13.5,6.4))
    labels=list(LAYOUTS.values())
    parts=["移动","检测","切换","清除成功","清除失败"]
    colors=["#317392","#62A9AE","#D1AE59","#57976F","#B55748"]
    tallest=0.0
    for j,opp in enumerate((False,True)):
        bottom=np.zeros(4)
        for part,color in zip(parts,colors):
            heights=[np.mean([r["结果"]["时间分项_秒"][part] for r in records if r["顺路检测"]==opp and r["构造参数"]["layout"]==layout]) for layout in LAYOUTS]
            axes[0].bar(np.arange(4)+(j-.5)*.36,heights,width=.34,bottom=bottom,color=color,label=part if j==0 else None)
            bottom+=heights
        tallest=max(tallest,float(bottom.max()))
    axes[0].set_xticks(range(4),labels)
    axes[0].set(ylabel="平均虚拟总时间（秒）",title="按布局比较时间分项\n每组左柱关闭、右柱开启顺路检测")
    axes[0].set_ylim(0,tallest*1.16)
    axes[0].legend(fontsize=8,ncol=3,loc="upper left")
    off=np.array([r["结果"]["虚拟总时间_秒"] for r in records if not r["顺路检测"]])
    on=np.array([r["结果"]["虚拟总时间_秒"] for r in records if r["顺路检测"]])
    delta=on-off
    axes[1].bar(np.arange(len(delta)),delta,color=np.where(delta<=0,"#308B7F","#CB7751"))
    axes[1].axhline(0,color="#293D48",lw=1)
    axes[1].set(xlabel="固定场景序号",ylabel="开启－关闭的虚拟时间差（秒）",title="每个场景的成对差值\n负值表示开启顺路检测更快")
    fig.suptitle("方案一：顺路检测的收益与代价均如实保留",fontsize=15)
    fig.text(.5,.015,NOTICE,ha="center",fontsize=9,color="#586A73")
    fig.tight_layout(rect=(0,.05,1,.93))
    save(fig,"方案一_时间分项与成对比较")

    representative=records[1]
    events=representative["结果"]["动作记录"]
    positives=[e for e in events if "外包顶点" in e]
    channel=max({e["频道"] for e in positives},key=lambda c:sum(e["频道"]==c for e in positives))
    selected=[e for e in positives if e["频道"]==channel]
    selected=selected[:2]+selected[-1:] if len(selected)>2 else selected
    source=next(s["位置"] for s in representative["真值_仅评估器读取"]["目标"] if s["频道"]==channel)
    fig,axes=plt.subplots(1,len(selected),figsize=(5*len(selected),5.6),squeeze=False)
    for ax,event in zip(axes[0],selected):
        p=np.array(event["外包顶点"])
        center,radius=minimum_circle(p)
        ax.add_patch(Polygon(p,facecolor="#8CCCD1",edgecolor="#24717C",alpha=.7,label="保守物理外包"))
        ax.add_patch(Circle(center,radius,fill=False,color="#AF6B31",ls="--",label="最小覆盖圆"))
        ax.scatter(*source,marker="*",s=100,c="#B95832",label="真实目标（离线核验）",zorder=5)
        station=np.array(event["位置"])
        angle=np.deg2rad(event["示向度"])
        ray=np.array([station,station+2000*np.array([np.cos(angle),np.sin(angle)])])
        ax.plot(ray[:,0],ray[:,1],ls=":",color="#326391",lw=1,label="本次示向射线")
        span=max(np.ptp(p,axis=0).max(),20)
        ax.set(xlim=(center[0]-.7*span,center[0]+.7*span),ylim=(center[1]-.7*span,center[1]+.7*span),aspect="equal",
               xlabel="东向坐标 x（米）",ylabel="北向坐标 y（米）")
        ax.set_title(f"站位({station[0]:.1f},{station[1]:.1f})\n覆盖圆半径 {radius:.2f} 米")
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,loc="upper right")
    fig.suptitle(f"频道{channel}：累计测向后的区域收缩（各面板独立缩放）",fontsize=15)
    fig.text(.5,.02,NOTICE,ha="center",fontsize=9,color="#586A73")
    fig.tight_layout(rect=(0,.13,1,.9))
    save(fig,"方案一_逐次定位区域")


def main():
    global NOTICE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only",action="store_true")
    parser.add_argument("--cases",type=int,default=30)
    args=parser.parse_args()
    if not args.plots_only and args.cases < 4: parser.error("至少4个场景以覆盖四类布局")
    if args.plots_only:
        previous=json.loads(SUMMARY.read_text(encoding="utf-8"))
        for relative,expected in previous["来源"].items():
            source=ROOT/relative
            if source.resolve()!=Path(__file__).resolve() and sha256(source.read_bytes()).hexdigest()!=expected:
                raise RuntimeError("计算源码或配置已变化，不能只重画图；请完整重算")
        if sha256(ARCHIVE.read_bytes()).hexdigest()!=previous["完整记录SHA256"]:
            raise RuntimeError("归档数据哈希不符，停止重画")
        records=[json.loads(line) for line in ARCHIVE.read_text(encoding="utf-8").splitlines()]
    else: records=run_suite(args.cases)
    NOTICE=f"本地构造验证，非官方演练或正式成绩；统计仅描述这{len(records)//2}个指定场景"
    make_plots(records)
    stats={}
    for opp,name in [(False,"关闭顺路检测"),(True,"开启顺路检测")]:
        rs=[r["结果"] for r in records if r["顺路检测"]==opp]
        stats[name]={"场景数":len(rs),"全部清除次数":sum(r["运行成功"] for r in rs),
            "平均虚拟总时间_秒":float(np.mean([r["虚拟总时间_秒"] for r in rs])),
            "平均每目标时间_秒":float(np.mean([r["平均定位清除时间_秒"] for r in rs])),
            "最小虚拟总时间_秒":min(r["虚拟总时间_秒"] for r in rs),
            "最大虚拟总时间_秒":max(r["虚拟总时间_秒"] for r in rs),
            "平均运行时间_秒":float(np.mean([r["程序运行时间_秒"] for r in rs])),
            "最大运行时间_秒":max(r["程序运行时间_秒"] for r in rs),
            "平均检测次数":float(np.mean([r["检测次数"] for r in rs])),
            "清除失败总次数":sum(r["清除失败次数"] for r in rs)}
    off=[r["结果"]["虚拟总时间_秒"] for r in records if not r["顺路检测"]]
    on=[r["结果"]["虚拟总时间_秒"] for r in records if r["顺路检测"]]
    sources=list((ROOT/"src/cumcm2026_b").glob("q3_*.py"))+[ROOT/"src/cumcm2026_b/q1_geometry.py",ROOT/"src/cumcm2026_b/q2_physical_geometry.py",Path(__file__),ROOT/"configs/q3_scheme1.json"]
    report={"说明":NOTICE,"对照统计":stats,"开启更快场景数":sum(a<b for a,b in zip(on,off)),
            "开启更慢场景数":sum(a>b for a,b in zip(on,off)),"持平场景数":sum(abs(a-b)<1e-6 for a,b in zip(on,off)),
            "总外包真值包含检查次数":sum(r["外包真值包含检查次数"] for r in records),
            "完整记录":str(ARCHIVE.relative_to(ROOT)),"完整记录SHA256":sha256(ARCHIVE.read_bytes()).hexdigest(),
            "来源":{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
            "环境":{"Python":platform.python_version(),"NumPy":np.__version__,"SciPy":scipy.__version__,"Matplotlib":matplotlib.__version__},
            "逐场摘要":[{"场景序号":r["场景序号"],"构造参数":r["构造参数"],"顺路检测":r["顺路检测"],
                **{k:r["结果"][k] for k in ["运行成功","清除数","虚拟总时间_秒","平均定位清除时间_秒","程序运行时间_秒","总路程_米","检测次数","清除失败次数"]}} for r in records]}
    SUMMARY.parent.mkdir(parents=True,exist_ok=True)
    SUMMARY.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"对照统计":stats,"开启更快场景数":report["开启更快场景数"],"开启更慢场景数":report["开启更慢场景数"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
