#!/usr/bin/env python
"""读取演练日志公开文件头，并与本地HTTP明文记录关联；不解密、不改动原日志。"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from build_q3_scheme1_assets import style, save
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def public_header(path):
    raw = path.read_bytes()
    if len(raw) < 14 or raw[:8] != b"JMBPLOG1": raise ValueError("不支持的演练日志格式")
    version = int.from_bytes(raw[8:10],"big")
    length = int.from_bytes(raw[10:14],"big")
    if version != 1 or not 0 < length <= len(raw)-14: raise ValueError("日志文件头长度或版本异常")
    header = json.loads(raw[14:14+length])
    fields = ["package_type","envelope_version","payload_schema_version","problem_no","formal_index",
              "practice_run_no","case_code","client_version","build_id","created_at_utc",
              "compression","compression_level","content_encryption","key_wrap_algorithm"]
    # 仅导出分析所需公开字段，不导出票据、队号或密钥封装内容。
    summary = {k:header.get(k) for k in fields}
    summary.update({"文件名":path.name,"文件字节数":len(raw),"SHA256":sha256(raw).hexdigest(),
                    "魔数":"JMBPLOG1","文件头字节数":length,"头后区域字节数":len(raw)-14-length,
                    "正文已解密":False,"包签名已验证":False})
    return header,summary


def matched_run(header):
    created = datetime.fromisoformat(header["created_at_utc"].replace("Z","+00:00")).timestamp()*1000
    candidates = []
    for folder in (ROOT/"local-only/第三问").glob("*http*"):
        log = folder/"请求响应日志.jsonl"
        result_path = folder/"运行结果.json"
        if not log.exists() or not result_path.exists(): continue
        items = [json.loads(s) for s in log.read_text(encoding="utf-8").splitlines()]
        requests = [x for x in items if x.get("类型") == "请求"]
        exits = {x["payload"]["request_id"] for x in requests if x["path"] == "/exit" and x["payload"].get("robot_id") == header["team_no"]}
        responses = [x for x in items if x.get("类型") == "响应" and x["request_id"] in exits and x["response"].get("accepted") is True]
        if not responses: continue
        delta = created-responses[-1]["response"]["real_timestamp_ms"]
        if abs(delta) <= 2000: candidates.append((folder,items,delta))
    if len(candidates) != 1: raise ValueError(f"找到{len(candidates)}份时间接近的同队号记录，不能唯一关联")
    return candidates[0]


def audit_actions(items,result):
    requests = {}
    responses = {}
    for item in items:
        if item.get("类型") == "请求": requests[item["payload"]["request_id"]] = item
        elif item.get("类型") == "响应": responses[item["request_id"]] = item
    position = (0.,0.); channel = 1; distance = 0.; virtual = 0.
    measurements = switches = clears = failures = 0
    endpoints = Counter(); feedback = Counter()
    for request_id,request in requests.items():
        response_item = responses[request_id]; response = response_item["response"]
        assert response_item["HTTP"] == 200 and response.get("accepted") is True
        path = request["path"]; payload = request["payload"]
        endpoints[path] += 1
        if path in {"/measure","/clear"}:
            point = tuple(payload["position"][k] for k in ("x","y"))
            segment = math.dist(position,point); distance += segment; virtual += segment/5; position = point
            if path == "/measure":
                measurements += 1; switch = int(channel != payload["channel"]); switches += switch
                channel = payload["channel"]; virtual += 5+switch
                feedback[response["measure_result"]] += 1
            else:
                if response["clear_result"] == "success": clears += 1; virtual += 5
                else: failures += 1; virtual += 3
        assert abs(virtual-response["virtual_time_s"]) < .001
    for value,key in [(measurements,"检测次数"),(switches,"频道切换次数"),(clears,"清除数"),
                      (failures,"清除失败次数"),(distance,"总路程_米"),(virtual,"虚拟总时间_秒")]:
        assert abs(value-result[key]) < .001,key
    assert len(requests) == len(responses)
    return {"不同动作数":len(requests),"接口次数":dict(endpoints),"测量反馈":dict(feedback),
            "HTTP状态":dict(Counter(x["HTTP"] for x in items if x.get("类型") == "响应")),
            "重试请求数":sum(x.get("尝试",1)>1 for x in items if x.get("类型") == "请求"),
            "通信异常数":sum(x.get("类型") == "通信异常" for x in items),"逐动作费用核验":"通过"}


def make_figure(result,case_code):
    style()
    plt.rcParams["svg.hashsalt"] = "q3-practice-analysis"
    fig,axes = plt.subplots(1,3,figsize=(17,6.8),gridspec_kw={"width_ratios":[1.15,1,1]})
    events = result["动作记录"]
    end = next(e["步骤"] for e in events if e["动作"] == "固定扫描完成")
    previous = [0.,0.]
    for e in events:
        if e["动作"] not in {"检测","清除"}: continue
        point = e["位置"]
        axes[0].plot([previous[0],point[0]],[previous[1],point[1]],color="#C77B39" if e["步骤"] > end else "#347CA0",lw=1.2,alpha=.8)
        previous = point
        if e["动作"] == "清除" and e["结果"] == "success":
            axes[0].scatter(*point,c="#317B68",s=25,zorder=4)
            axes[0].annotate(str(e["频道"]),point,xytext=(5,6),textcoords="offset points",fontsize=8)
    fixed = np.array(result["布局"]["站点"])
    axes[0].scatter(fixed[:,0],fixed[:,1],marker="s",s=15,c="#173E50",label="固定检测站")
    axes[0].plot([],[],c="#347CA0",label="固定扫描路线")
    axes[0].plot([],[],c="#C77B39",label="后续定位清除路线")
    axes[0].scatter([],[],c="#317B68",label="成功清除位置（非目标真值）")
    axes[0].add_patch(Circle((0,0),1800,fill=False,color="#79528B",lw=1.2))
    axes[0].set(aspect="equal",xlabel="东向坐标（米）",ylabel="北向坐标（米）",title="实际轨迹与清除频道",xlim=(-2050,2050),ylim=(-2050,2050))
    axes[0].legend(loc="lower center",bbox_to_anchor=(.5,-.31),fontsize=8,ncol=2)
    parts = result["时间分项_秒"]
    axes[1].bar(range(len(parts)),list(parts.values()),color=["#347CA0","#62A9AE","#D1AE59","#57976F","#B55748"])
    for i,v in enumerate(parts.values()): axes[1].text(i,v+50,f"{v:.1f}",ha="center",fontsize=9)
    axes[1].set_xticks(range(len(parts)),list(parts),fontsize=9,rotation=20)
    axes[1].set(ylabel="虚拟时间（秒）",title=f"移动占总时间{parts['移动']/result['虚拟总时间_秒']*100:.2f}%",ylim=(0,max(parts.values())*1.2))
    states = [s for s in result["扫描阶段"]["频道快照"] if s["状态"] == "found"]
    channels = [s["频道"] for s in states]; radii = [s["外包覆盖圆半径_米"] for s in states]
    axes[2].bar(range(len(states)),radii,color=["#C77B39" if r>19.5 else "#62A9AE" for r in radii])
    axes[2].axhline(19.5,c="#A44942",ls="--",label="实现安全清除阈值19.5米")
    axes[2].set_xticks(range(len(states)),channels,fontsize=8)
    axes[2].set(xlabel="目标频道",ylabel="外包最小覆盖圆半径（米）",
                title=f"扫描后{sum(r>19.5 for r in radii)}个频道未通过清除阈值",ylim=(0,max([19.5]+radii)*1.32))
    axes[2].legend(fontsize=8,loc="upper left")
    fig.suptitle(f"方案二演练行为分析：{case_code}",fontsize=15)
    fig.text(.5,.015,"行为来自时间吻合的本地HTTP记录；jlog正文未解密，未读取官方目标真值",ha="center",fontsize=10,color="#586A73")
    fig.tight_layout(rect=(0,.12,1,.92))
    name = f"演练日志分析_{case_code}"
    save(fig,name)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jlog",type=Path,required=True)
    args = parser.parse_args()
    header,summary = public_header(args.jlog)
    if header.get("package_type") != "practice_behavior_log" or header.get("problem_no") != 3:
        raise ValueError("此入口仅分析问题3演练日志")
    folder,items,delta = matched_run(header)
    result_path = folder/"运行结果.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "扫描阶段" not in result: raise ValueError("当前绘图仅支持方案二扫描阶段结构")
    audit = audit_actions(items,result)
    case_code = header["case_code"]
    if re.fullmatch(r"[A-Z0-9-]+",case_code) is None: raise ValueError("案例码格式异常")
    figure = make_figure(result,case_code)
    report = {"信息来源边界":"文件头来自指定jlog；行为统计来自同队号且退出时间吻合的本地HTTP记录，未解密正文、未验证包签名或官方目标总数",
        "日志公开头":summary,"关联":{"本地目录":folder.relative_to(ROOT).as_posix(),"队号一致":True,
            "文件生成减退出响应时间_毫秒":delta,"证据性质":"时间与队号高度吻合；非密文内容比对"},
        "本地运行摘要":{k:result[k] for k in ["方案","配置","运行成功","正常退出","异常","终止依据","清除数",
            "已清除频道","已证明不存在频道","虚拟总时间_秒","平均定位清除时间_秒","程序运行时间_秒",
            "总路程_米","检测次数","频道切换次数","清除失败次数","时间分项_秒"]},
        "扫描阶段":{k:v for k,v in result["扫描阶段"].items() if k!="频道快照"},
        "扫描后逐频道":[{k:v for k,v in x.items() if k!="外包顶点"} for x in result["扫描阶段"]["频道快照"]],
        "追加测向":[{k:v for k,v in e.items() if k!="外包顶点"} for e in result["动作记录"] if e.get("用途")=="追加测向"],
        "清除顺序":[{"频道":e["频道"],"位置":e["位置"],"虚拟时间_秒":e["虚拟时间_秒"],"结果":e["结果"]} for e in result["动作记录"] if e["动作"]=="清除"],
        "请求响应核验":audit,"图表":figure,
        "来源SHA256":{p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in
            [result_path,folder/"请求响应日志.jsonl",Path(__file__),ROOT/"scripts/build_q3_scheme1_assets.py"]}}
    output = ROOT/f"results/tables/第三问/演练日志分析_{case_code}.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"公开头":summary,"匹配时间差_毫秒":delta,"请求响应核验":audit},ensure_ascii=False,indent=2))
    print(f"报告数据：{output}")


if __name__ == "__main__": main()
