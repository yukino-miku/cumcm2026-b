#!/usr/bin/env python
"""独立验收方案二同场景归档、实际动作费用、双接收及图表链接。"""
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cumcm2026_b.q3_geometry import min_distance


def strict(text):
    return json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def main():
    report = strict((ROOT/"results/tables/第三问/方案二_完整闭环对照汇总.json").read_text(encoding="utf-8"))
    archive = ROOT/report["完整记录"]
    assert sha256(archive.read_bytes()).hexdigest() == report["完整记录SHA256"]
    for path,value in report["来源"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == value,path
    records = [strict(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(report["逐场摘要"])
    grouped = {}; checked = 0; scan_sources = 0
    for record,brief in zip(records,report["逐场摘要"]):
        result,truth = record["结果"],record["真值_仅评估器读取"]
        targets = {s["频道"]:s for s in truth["目标"]}
        assert result["运行成功"] and result["正常退出"] and truth["全部清除"]
        assert set(result["已清除频道"]) == set(targets)
        assert result["清除数"] == truth["目标数"] == truth["清除数"]
        assert set(result["已证明不存在频道"]).isdisjoint(targets)
        for key in brief:
            if key in result: assert brief[key] == result[key]
            else: assert brief[key] == record[key]
        grouped.setdefault(record["场景序号"],[]).append(record)
        virtual = distance = 0.; measurements = switches = successes = failures = 0
        position = [0.,0.]; channel = 1
        for e in result["动作记录"]:
            if e["动作"] in {"检测","清除"}:
                segment = math.dist(position,e["位置"])
                distance += segment
                virtual += round(segment/5,6)
                position = e["位置"]
                if e["动作"] == "检测":
                    measurements += 1
                    switches += e["频道"] != channel
                    virtual += 5+(e["频道"] != channel)
                    channel = e["频道"]
                    if e["结果"] == "direction":
                        target = targets[channel]
                        assert 5 < math.dist(position,target["位置"]) <= target["接收半径"]+1e-6
                        angle = math.degrees(math.atan2(target["位置"][1]-position[1],target["位置"][0]-position[0]))
                        error = (e["示向度"]-angle+180)%360-180
                        assert abs(error) <= 1.005+1e-8
                elif e["结果"] == "success":
                    successes += 1; virtual += 5
                    assert math.dist(position,targets[e["频道"]]["位置"]) <= 20+1e-6
                else:
                    failures += 1; virtual += 3
                assert abs(virtual-e["虚拟时间_秒"]) < .001
            if "外包顶点" in e:
                assert min_distance(e["外包顶点"],targets[e["频道"]]["位置"]) <= 1e-5
                checked += 1
            if e["动作"] == "清除" and e["具有覆盖证书"]:
                assert e["结果"] == "success" and e["最远可能目标距离_米"] < 20
        for actual,key in [(virtual,"虚拟总时间_秒"),(distance,"总路程_米"),(measurements,"检测次数"),
                           (switches,"频道切换次数"),(successes,"清除成功次数"),(failures,"清除失败次数")]:
            assert abs(actual-result[key]) < .001,(key,record["设置"],record["场景序号"])
        assert abs(sum(result["时间分项_秒"].values())-virtual) < .001
        if record["设置"] != "scheme1":
            end = next(e["步骤"] for e in result["动作记录"] if e["动作"] == "固定扫描完成")
            fixed = np.array(result["布局"]["站点"])
            before = [e for e in result["动作记录"] if e["步骤"] < end]
            for e in before:
                if e["动作"] in {"检测","清除"}:
                    assert np.linalg.norm(fixed-e["位置"],axis=1).min() < 1e-7
                assert e.get("用途") not in {"追加测向","顺路检测","有限方格覆盖"}
            for row in result["扫描阶段"].get("频道快照",[]):
                assert row["状态"] != "unknown"
                if row["状态"] == "found":
                    positives = {tuple(e["位置"]) for e in before if e["动作"] == "检测" and e["频道"] == row["频道"] and e["结果"] == "direction"}
                    assert len(positives) == row["不同方向站数"] >= 2
                    assert min_distance(row["外包顶点"],targets[row["频道"]]["位置"]) <= 1e-5
                    scan_sources += 1
    assert checked == report["外包真值包含检查总数"]
    for group in grouped.values():
        assert {r["设置"] for r in group} == {"scheme1","main14","main14_defer","margin14","compact15"}
        assert len(group) == 5
        assert all(r["构造参数"] == group[0]["构造参数"] for r in group)
        assert all(r["真值_仅评估器读取"] == group[0]["真值_仅评估器读取"] for r in group)
    for variant,stats in report["统计"].items():
        subset = [r["结果"] for r in records if r["设置"] == variant]
        assert len(subset) == stats["运行数"] == stats["完整清除次数"]
        assert abs(np.mean([r["虚拟总时间_秒"] for r in subset])-stats["平均虚拟总时间_秒"]) < 1e-8
        if variant != "scheme1":
            delta = np.array([next(r["结果"]["虚拟总时间_秒"] for r in group if r["设置"] == variant)-
                              next(r["结果"]["虚拟总时间_秒"] for r in group if r["设置"] == "scheme1") for group in grouped.values()])
            pair = report["与方案一成对比较"][variant]
            assert (delta < -1e-6).sum() == pair["比方案一更快场景数"]
            assert (delta > 1e-6).sum() == pair["更慢场景数"]
            assert abs(delta.mean()-pair["平均差值_秒"]) < 1e-8
    figures = ["方案二_完整轨迹与阶段","方案二_两方案及布局对照","方案二_扫描精度与后续费用","方案二_执行流程图"]
    for name in figures:
        with Image.open(ROOT/f"results/figures/第三问/{name}.png") as image: image.verify()
        ET.parse(ROOT/f"results/figures/第三问/{name}.svg")
    links = 0
    for path in list((ROOT/"docs/第三问").glob("*.md"))+list((ROOT/"paper/sections").glob("第三问*.md")):
        text = path.read_text(encoding="utf-8")
        assert '\ufffd' not in text and text.count('```')%2 == 0,path
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).resolve().exists(),(path,target)
                links += 1
    print(f"Q3_SCHEME2_VERIFY_OK：{len(records)}次运行、{checked}次外包核验、{scan_sources}个扫描后待清除目标双接收核验、4组PNG/SVG、{links}个本地链接；逐动作费用与来源哈希一致。")


if __name__ == "__main__": main()
