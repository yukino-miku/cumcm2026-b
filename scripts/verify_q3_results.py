#!/usr/bin/env python
"""独立验收第三问方案一归档、真值包含、计时、中文图和来源。"""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cumcm2026_b.q3_geometry import min_distance


def strict(path):
    return json.loads(path.read_text(encoding="utf-8"),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def main():
    report=strict(ROOT/"results/tables/第三问/方案一_本地对照汇总.json")
    archive=ROOT/report["完整记录"]
    assert sha256(archive.read_bytes()).hexdigest()==report["完整记录SHA256"],"完整记录发生变化"
    for relative,expected in report["来源"].items():
        assert sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    records=[json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert len(records)==len(report["逐场摘要"])
    checked=0
    for record,brief in zip(records,report["逐场摘要"]):
        result=record["结果"]
        truth=record["真值_仅评估器读取"]
        positions={x["频道"]:x["位置"] for x in truth["目标"]}
        assert result["运行成功"] and truth["全部清除"]
        assert result["清除数"]==len(positions)==truth["清除数"]
        assert set(result["已证明不存在频道"]).isdisjoint(positions)
        assert set(result["已清除频道"])==set(positions)
        assert abs(sum(result["时间分项_秒"].values())-result["虚拟总时间_秒"])<.001
        for key in ["清除数","虚拟总时间_秒","检测次数","清除失败次数"]:assert brief[key]==result[key]
        for event in result["动作记录"]:
            if "外包顶点" in event:
                assert min_distance(event["外包顶点"],positions[event["频道"]])<=1e-5
                checked+=1
            if event["动作"]=="清除" and event["具有覆盖证书"]:
                assert event["结果"]=="success" and event["最远可能目标距离_米"]<20
        for state in result["频道记录"]:
            if state["状态"]=="absent":assert state["不存在证书"] is not None
    assert checked==report["总外包真值包含检查次数"]
    for opp,name in [(False,"关闭顺路检测"),(True,"开启顺路检测")]:
        subset=[r["结果"] for r in records if r["顺路检测"]==opp]
        stats=report["对照统计"][name]
        assert len(subset)==stats["场景数"]==stats["全部清除次数"]
        assert abs(np.mean([r["虚拟总时间_秒"] for r in subset])-stats["平均虚拟总时间_秒"])<1e-8
    figures=["方案一_运动轨迹对照","方案一_时间分项与成对比较","方案一_逐次定位区域"]
    for name in figures:
        with Image.open(ROOT/f"results/figures/第三问/{name}.png") as im:im.verify()
        ET.parse(ROOT/f"results/figures/第三问/{name}.svg")
    links=0
    docs=list((ROOT/"docs/第三问").glob("*.md"))+[ROOT/"paper/sections/第三问_方案一_论文备用稿.md"]
    for path in docs:
        text=path.read_text(encoding="utf-8")
        assert '\ufffd' not in text and text.count('```')%2==0,path
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).resolve().exists(),(path,target)
                links+=1
    print(f"Q3_VERIFY_OK：{len(records)}次完整运行、{checked}次外包真值检查、3组PNG/SVG、{links}个本地链接；源码与数据哈希一致。")


if __name__=="__main__":main()
