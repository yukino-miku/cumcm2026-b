"""第二问交付验收：严格 JSON、连续检测条件、数值结果与文件来源。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cumcm2026_b.q2_physical_geometry import build_physical_region
from cumcm2026_b.q2_active_localization import robust_objective


def strict_load(path):
    def reject(text):
        raise ValueError(f"JSON 出现非法常量 {text}：{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def main():
    tables = ROOT / "results/tables/第二问"
    all_json = list(tables.glob("*.json")) + list((ROOT / "experiments/第二问").glob("*.json"))
    for path in all_json:
        strict_load(path)
    metadata = strict_load(ROOT / "experiments/第二问/计算配置与来源.json")
    for group in ("计算代码SHA256", "输入SHA256", "第一问受保护文件SHA256"):
        for name, expected in metadata[group].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"来源或受保护文件哈希不一致：{name}")
    archive = strict_load(tables / "鲁棒搜索输入输出.json")
    assert archive["configuration"] == metadata["configuration"], "计算配置与归档配置不一致"
    plot_metadata = strict_load(ROOT / "experiments/第二问/实验配置与环境.json")
    assert plot_metadata["计算来源记录"] == metadata, "绘图引用了不同计算来源"
    assert plot_metadata["归档搜索结果SHA256"] == hashlib.sha256((tables / "鲁棒搜索输入输出.json").read_bytes()).hexdigest()
    for name, expected in plot_metadata["绘图代码SHA256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"绘图源码哈希不同：{name}"
    physics = archive["configuration"]["physical"]
    total_candidates, q1_checks = 0, 0
    assert len(archive["cases"]) == len(archive["configuration"]["cases"]) >= 4
    for case in archive["cases"]:
        inputs, result = case["input"], case["search"]
        region = build_physical_region(inputs["station1"], inputs["bearing1_deg"], physics["half_width_deg"],
            physics["target_center"], physics["target_radius"], physics["max_receive_radius"])
        assert abs(region.area - case["physical_region"]["area_m2"]) < 1e-5
        assert region.contains(np.asarray(case["target_scenarios"]), tol=1e-6).all()
        grid = result["grid"]
        values = np.asarray(grid["diameter_m"], dtype=float)
        assert values.shape == (len(grid["y_local_m"]), len(grid["x_local_m"]))
        assert np.all(np.asarray(grid["reliable_bearing"])[np.isfinite(values)])
        best = result["best"]
        assert best["status"] == "finite" and best["q1_verified"] and best["reliable_bearing"]
        candidates = set()
        for key, record in result["candidate_regions"].items():
            threshold = (1 + record["eta"]) * best["diameter_m"]
            assert record["grid_count"] == int(np.count_nonzero(values <= threshold))
            assert record["estimated_area_m2"] == record["grid_count"] * result["configuration"]["grid_step_m"] ** 2
            for index in record["point_indices"]:
                value = result["evaluations"][index]
                assert value["reliable_bearing"] and value["diameter_m"] <= threshold + 1e-7
                assert value["angle_count"] >= result["configuration"]["final_angle_count"] and value["refine"]
                candidates.add(index)
        for index in sorted(candidates):
            value = result["evaluations"][index]
            point = np.asarray(value["station"])
            maximum, minimum = region.max_distance(point), region.min_distance(point)
            assert maximum <= physics["guaranteed_radius"] - physics["safety_margin"] + 1e-7
            assert minimum >= physics["near_radius"] + physics["safety_margin"] - 1e-7
            assert abs(value["max_distance_m"] - maximum) < 1e-6
            assert abs(value["movement_distance_m"] - np.linalg.norm(point - region.station)) < 1e-6
        # 对每例近优点作确定性分层抽查，再用更密的角网格和第一问原函数核验。
        ordered = sorted(candidates, key=lambda i: result["evaluations"][i]["diameter_m"])
        selected = [result["evaluations"][ordered[i]] for i in sorted(set(np.linspace(0, len(ordered) - 1, min(9, len(ordered)), dtype=int)))]
        selected += result["side_optima"]
        for value in selected:
            checked = robust_objective(region, value["station"], angle_count=513, refine=True, verify_q1=True,
                radius=physics["guaranteed_radius"], safety_margin=physics["safety_margin"], near_radius=physics["near_radius"])
            assert checked["q1_verified"] and checked["reliable_bearing"]
            assert abs(checked["diameter_m"] - value["diameter_m"]) <= max(2e-5, value["diameter_m"] * 1e-7)
            q1_checks += 1
        total_candidates += len(candidates)
    figure_dir = ROOT / "results/figures/第二问"
    png_names = {p.stem for p in figure_dir.glob("*.png")}
    svg_names = {p.stem for p in figure_dir.glob("*.svg")}
    assert png_names == svg_names and len(png_names) >= 9, "图表数量或 PNG/SVG 配对不完整"
    for path in figure_dir.glob("*.svg"):
        assert ET.parse(path).getroot().tag.endswith("svg")
    required = [ROOT / "docs/第二问/阅读导航.md", ROOT / "docs/第二问/思路与数学推导.md",
                ROOT / "docs/第二问/验证与复现记录.md", ROOT / "paper/sections/第二问_论文备用稿.md",
                ROOT / "experiments/第二问/构造实验报告.md"]
    assert all(path.is_file() for path in required), "中文交付文档不完整"
    links = 0
    for path in list((ROOT / "docs/第二问").glob("*.md")) + required[-2:] + list(tables.glob("*.md")):
        for matched in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding="utf-8-sig")):
            target = matched.group(1).strip().strip("<>")
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            assert (path.parent / target).exists(), f"失效链接：{path.name} -> {target}"
            links += 1
    print(f"第二问验收通过：{len(all_json)} 份严格 JSON，{total_candidates} 个公开近优点连续距离核查，{q1_checks} 次更密角网格与第一问原函数抽查，{len(png_names)} 组图，{links} 个本地链接。")
    print(f"计算来源一致；第一问 {len(metadata['第一问受保护文件SHA256'])} 个受保护文件保持不变。")


if __name__ == "__main__":
    main()
