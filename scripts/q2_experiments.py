"""第二问构造实验与结果归档；计算结果供独立绘图入口使用。"""

from __future__ import annotations

import json
import math
import hashlib
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cumcm2026_b.q1_geometry import solve_bearings
from cumcm2026_b.q2_physical_geometry import (
    analytic_candidates, build_physical_region, centerline_interval,
    reliable_boundary, sample_region,
)
from cumcm2026_b.q2_active_localization import (
    robust_objective, sampled_robust_objective, search_second_station, serialize_result,
)

NOTICE = "构造算例，非官方模拟器测试"
TABLES = ROOT / "results/tables/第二问"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_q1_files():
    """明确保护第一问代码、输入、测试、文稿和已有输出。"""
    single = ["src/cumcm2026_b/q1_geometry.py", "tests/test_q1_geometry.py",
              "tests/test_q1_reference_polygons.py", "configs/q1_cases.json",
              "scripts/build_q1_assets.py", "scripts/run-q1.ps1", "paper/sections/第一问_论文备用稿.md"]
    paths = [ROOT / name for name in single]
    for folder in ["docs/第一问", "experiments/第一问", "results/figures/第一问", "results/tables/第一问"]:
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(paths)}


def write_json(path, value, compact=False):
    """无界数值显式序列化为 null，由同一记录中的状态解释。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize_result(value), ensure_ascii=False, allow_nan=False,
                              indent=None if compact else 2) + "\n", encoding="utf-8", newline="\n")


def make_region(case, physical, half_width=None):
    return build_physical_region(case["station1"], case["bearing1_deg"],
        physical["half_width_deg"] if half_width is None else half_width,
        target_center=physical["target_center"], target_radius=physical["target_radius"],
        reception_upper=physical["max_receive_radius"])


def to_local(points, region):
    angle = np.radians(region.bearing_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (np.asarray(points) - region.station) @ rotation


def to_global(points, region):
    angle = np.radians(region.bearing_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(points) @ rotation.T + region.station


def polygon_area(points):
    p = np.asarray(points)
    return float(abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))) / 2) if len(p) >= 3 else 0.0


def region_statistics_only(records):
    """收敛实验只保存统计量，避免留下无法解析到对应点集的索引。"""
    return {key: {name: value for name, value in row.items() if name != "point_indices"}
            for key, row in records.items()}


def evaluate(region, station, physical, count=257, refine=True, verify=False, radius=None):
    return robust_objective(region, station, angle_count=count, refine=refine, verify_q1=verify,
        radius=physical["guaranteed_radius"] if radius is None else radius,
        safety_margin=physical["safety_margin"], near_radius=physical["near_radius"])


def search(region, configuration, step=None, radius=None):
    physical, options = configuration["physical"], configuration["search"]
    return search_second_station(region,
        grid_step=options["grid_step"] if step is None else step,
        angle_count=options["angle_count"], eta=options["eta"],
        final_angle_count=options["final_angle_count"],
        radius=physical["guaranteed_radius"] if radius is None else radius,
        safety_margin=physical["safety_margin"], near_radius=physical["near_radius"])


def safe_analytic_point(region, point, physical):
    """将中心线解析点沿安全种子连线回缩；仅作透明可复现基线。"""
    seed = to_global([region.reception_upper / (2 * np.cos(np.radians(region.half_width_deg))), 0.0], region)
    radius = physical["guaranteed_radius"] - physical["safety_margin"]
    if region.max_distance(seed) > radius:
        raise RuntimeError("解析回缩基线的安全种子不适用。")
    if region.max_distance(point) <= radius:
        return np.asarray(point)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if region.max_distance(seed + mid * (point - seed)) <= radius:
            lo = mid
        else:
            hi = mid
    return seed + lo * (point - seed)


def run_experiments(configuration):
    """实际执行四组主构造、策略对照、几何/采样/参数灵敏度实验。"""
    TABLES.mkdir(parents=True, exist_ok=True)
    code_names = ["src/cumcm2026_b/q1_geometry.py", "src/cumcm2026_b/q2_physical_geometry.py",
                  "src/cumcm2026_b/q2_active_localization.py", "scripts/q2_experiments.py"]
    code_hashes = {name: file_hash(ROOT / name) for name in code_names}
    original_q1 = protected_q1_files()
    physical, options = configuration["physical"], configuration["search"]
    runs = []
    for case in configuration["cases"]:
        print(f"正在搜索：{case['name_zh']}……", flush=True)
        region = make_region(case, physical)
        result = search(region, configuration)
        boundary = reliable_boundary(region, radius=physical["guaranteed_radius"], count=256,
                                     safety_margin=physical["safety_margin"])
        targets = sample_region(region, boundary_count=128, radial_levels=4)
        run = {"case": case, "region": region, "search": result, "reliable_boundary": boundary,
               "targets": targets, "reliable_area_m2": polygon_area(boundary)}
        runs.append(run)
        print(f"  当前数值最优：{result['best']['station']}，最坏直径 {result['best']['diameter_m']:.6f} 米", flush=True)

    standard, region = runs[0], runs[0]["region"]
    best = standard["search"]["best"]
    analytic = analytic_candidates(region, physical["guaranteed_radius"])
    adjusted = safe_analytic_point(region, analytic[0], physical)
    baseline_points = [
        ("沿首次示向线前进750米", to_global([750, 0], region)),
        ("原地侧向500米", to_global([0, 500], region)),
        ("中段固定侧向500米", to_global([750, 500], region)),
        ("中心线解析候选", analytic[0]),
        ("解析候选可靠化回缩", adjusted),
        ("鲁棒数值优化", np.asarray(best["station"])),
    ]
    baselines = []
    for name, point in baseline_points:
        value = evaluate(region, point, physical, verify=True)
        value["strategy_zh"] = name
        baselines.append(value)
    # 沿首次示向线还给出一个可测向但无界的直接第一问反例。
    parallel = solve_bearings([[0, 0], [750, 0]], [0, 0])
    if parallel.kind != "unbounded":
        raise AssertionError("沿线同向构造应产生无界角域交集。")

    print("正在进行观测角、目标场景和边界采样收敛核查……", flush=True)
    angle_convergence = []
    for count in configuration["sensitivity"]["angle_counts"]:
        val = evaluate(region, best["station"], physical, count=count, refine=False)
        angle_convergence.append({"angle_count": count, "evaluation": val})
    scenario_convergence = []
    for budget, errors in zip(configuration["sensitivity"]["boundary_counts"],
                              configuration["sensitivity"]["error_counts"], strict=True):
        points = sample_region(region, boundary_count=budget, radial_levels=4)
        value = sampled_robust_objective(region, best["station"], points, error_count=errors)
        boundary = reliable_boundary(region, radius=physical["guaranteed_radius"], count=budget,
                                     safety_margin=physical["safety_margin"])
        scenario_convergence.append({"boundary_budget": budget, "target_count": len(points),
            "error_count": errors, "scenario_count": len(points) * errors,
            "evaluation": value, "K_boundary_count": len(boundary),
            "K_polygon_area_m2": polygon_area(boundary),
            "K_boundary_max_distance_m": max(region.max_distance(s) for s in boundary)})

    grid_convergence = []
    for step in configuration["sensitivity"]["grid_steps_m"]:
        print(f"正在核查外层格网：{step:g} 米……", flush=True)
        result = standard["search"] if step == options["grid_step"] else search(region, configuration, step=step)
        grid_convergence.append({"grid_step_m": step, "best": result["best"],
            "candidate_regions": region_statistics_only(result["candidate_regions"]), "configuration": result["configuration"]})

    angles = []
    for width in configuration["sensitivity"]["angle_half_widths_deg"]:
        print(f"正在核查误差半宽：{width:g} 度……", flush=True)
        changed = make_region(configuration["cases"][0], physical, half_width=width)
        result = standard["search"] if width == physical["half_width_deg"] else search(changed, configuration, step=50)
        angles.append({"half_width_deg": width, "fixed_strategy": evaluate(changed, best["station"], physical),
            "optimized_strategy": result["best"], "configuration": result["configuration"]})

    radii = []
    for radius in configuration["sensitivity"]["guaranteed_radii_m"]:
        print(f"正在核查保证接收半径下界：{radius:g} 米……", flush=True)
        result = standard["search"] if radius == physical["guaranteed_radius"] else search(region, configuration, step=50, radius=radius)
        radii.append({"guaranteed_radius_m": radius, "best": result["best"], "configuration": result["configuration"]})

    distances = []
    for upper in [600.0, 900.0, 1200.0, 1500.0]:
        center = np.asarray(physical["target_center"])
        case = {"station1": (center + [physical["target_radius"] - upper, 0.0]).tolist(), "bearing1_deg": 0.0}
        changed = make_region(case, physical)
        # 1500米中心线长度对应S1=300，仍未受目标圆截断；单独计算保持输入明确。
        result = search(changed, configuration, step=50)
        distances.append({"station1": case["station1"], "bearing1_deg": 0.0,
            "centerline_interval_m": centerline_interval(changed), "best": result["best"],
            "configuration": result["configuration"]})

    first_question_angles = json.loads((ROOT / "results/tables/第一问/交会角扫描输入输出.json").read_text(encoding="utf-8"))
    angle_geometry = [{"angle_deg": row["中心射线交会角_度"], "distance_m": 1000.0,
                       "diameter_m": row["区域直径_米"]} for row in first_question_angles]
    for angle in range(5, 176, 5):
        rad = np.radians(angle)
        stations = np.array([[-500, 0], [-500 * np.cos(rad), -500 * np.sin(rad)]])
        answer = solve_bearings(stations, [0, angle])
        angle_geometry.append({"angle_deg": angle, "distance_m": 500.0, "diameter_m": answer.diameter})

    archive = {"来源": NOTICE, "configuration": configuration, "cases": []}
    summaries = []
    for run in runs:
        obj = {"input": run["case"], "physical_region": run["region"].to_dict(),
               "target_scenarios": run["targets"], "reliable_boundary": run["reliable_boundary"],
               "reliable_area_m2": run["reliable_area_m2"], "search": run["search"]}
        archive["cases"].append(obj)
        summaries.append({key: value for key, value in obj.items() if key not in ("search", "target_scenarios")}
                         | {"best": run["search"]["best"], "side_optima": run["search"]["side_optima"],
                            "analytic_candidates": run["search"]["analytic_candidates"],
                            "candidate_regions": run["search"]["candidate_regions"]})
    write_json(TABLES / "鲁棒搜索输入输出.json", archive, compact=True)
    write_json(TABLES / "标准构造案例输入输出.json", {"来源": NOTICE, "cases": summaries})
    write_json(TABLES / "候选区域统计.json", {"来源": NOTICE,
        "索引说明": "point_indices 指向鲁棒搜索输入输出.json中同名case的search.evaluations，不指向其他格网实验。", "cases": [
        {"case": run["case"], "thresholds": run["search"]["candidate_regions"]} for run in runs]})
    convergence = {"来源": NOTICE, "说明": "内层角网格与独立目标乘误差场景比较；外层最优值和规则格网面积估计分别检查。",
                   "angle_grid": angle_convergence, "target_error_grid": scenario_convergence, "station_grid": grid_convergence}
    write_json(TABLES / "采样收敛分析.json", convergence)
    write_json(TABLES / "角度误差敏感性.json", {"来源": NOTICE, "rows": angles})
    write_json(TABLES / "物理范围敏感性.json", {"来源": NOTICE, "说明": "半径1000米和误差1度才是正式题设；其他值仅为理论扰动。",
                                              "reception_radius": radii, "distance_interval": distances})
    write_json(TABLES / "策略对比输入输出.json", {"来源": NOTICE, "strategies": baselines,
                                               "沿线直接第一问核查": parallel.to_dict()})
    write_json(TABLES / "交会角与直径数据.json", {"来源": NOTICE, "rows": angle_geometry,
        "第一问数据来源": "results/tables/第一问/交会角扫描输入输出.json"})
    if code_hashes != {name: file_hash(ROOT / name) for name in code_names}:
        raise RuntimeError("计算过程中源码发生变化，结果来源不一致，请在代码稳定后重新构建。")
    if original_q1 != protected_q1_files():
        raise RuntimeError("构建改变了第一问受保护文件，请停止并检查原因。")
    import scipy
    provenance = {"来源": NOTICE, "configuration": configuration,
        "计算代码SHA256": code_hashes,
        "输入SHA256": {"configs/q2_cases.json": file_hash(ROOT / "configs/q2_cases.json"),
                      "results/tables/第一问/交会角扫描输入输出.json": file_hash(ROOT / "results/tables/第一问/交会角扫描输入输出.json")},
        "第一问受保护文件SHA256": original_q1,
        "生成时已有Git提交": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "版本说明": "源码和输入SHA256标识实际计算版本；生成时Git提交可能早于成果提交。仅重画时保留此计算记录。",
        "Python": platform.python_version(), "NumPy": np.__version__, "SciPy": scipy.__version__,
        "主构造数量": len(runs), "主构造记录数": [len(run["search"]["evaluations"]) for run in runs],
        "主构造目标验证点数": [len(run["targets"]) for run in runs],
        "独立场景核查总数": sum(row["scenario_count"] for row in scenario_convergence)}
    write_json(ROOT / "experiments/第二问/计算配置与来源.json", provenance)
    return {"runs": runs, "baselines": baselines, "convergence": convergence,
            "angle_sensitivity": angles, "radii": radii, "distances": distances, "angle_geometry": angle_geometry}
