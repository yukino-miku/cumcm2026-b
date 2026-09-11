#!/usr/bin/env python
"""复核路线调度归档、原始反馈/费用/证书、同场景配对、统计和中文图表。"""
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from PIL import Image
from build_q3_route_assets import ROOT, REPORT, VARIANTS, audit, summarize


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    archive = ROOT / report["完整记录"]
    assert sha256(archive.read_bytes()).hexdigest() == report["完整记录SHA256"]
    for name, value in report["来源SHA256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == value, name
    records = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    totals = {k: 0 for k in report["核验合计"]}
    groups = {}
    for record in records:
        checked = audit(record)
        assert checked == record["独立核验"]
        for key, value in checked.items(): totals[key] += value
        groups.setdefault((record["数据组"], record["场景序号"]), []).append(record)
    for batch in groups.values():
        assert len(batch) == len(VARIANTS) and {r["设置"] for r in batch} == set(VARIANTS)
        assert all(r["构造参数"] == batch[0]["构造参数"] for r in batch)
        assert all(r["真值_仅评估器读取"] == batch[0]["真值_仅评估器读取"] for r in batch)
    old_seeds = {r["构造参数"]["seed"] for r in records if r["数据组"] == "既有构造集"}
    new_seeds = {r["构造参数"]["seed"] for r in records if r["数据组"] == "新增验证集"}
    assert old_seeds.isdisjoint(new_seeds)
    stats, pairs = summarize(records)
    assert stats == report["统计"] and pairs == report["成对比较"] and totals == report["核验合计"]
    assert len(records) == len(report["逐场摘要"])
    for record, brief in zip(records, report["逐场摘要"]):
        for key, value in brief.items(): assert value == (record[key] if key in record else record["结果"][key])
    for name in report["图表"]:
        with Image.open(ROOT/f"results/figures/第三问/{name}.png") as im: im.verify()
        ET.parse(ROOT/f"results/figures/第三问/{name}.svg")
    links = 0
    for path in list((ROOT/"docs/第三问").glob("*.md")) + list((ROOT/"paper/sections").glob("第三问*.md")):
        text = path.read_text(encoding="utf-8")
        assert '\ufffd' not in text and text.count('```') % 2 == 0, path
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).resolve().exists(), (path, target)
                links += 1
    print(f"Q3_ROUTE_VERIFY_OK：{len(groups)}个场景、{len(records)}次完整运行；{totals}；4组PNG/SVG、{links}个中文文档链接及全部来源哈希通过。")


if __name__ == "__main__": main()
