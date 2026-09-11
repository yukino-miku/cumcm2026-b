#!/usr/bin/env python
"""两方案路线调度的六设置成对构造实验与中文图表；不连接官方模拟器。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_q3_scheme1_assets import style, save, check_run
from cumcm2026_b.q3_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_strategy import SchemeOne
from cumcm2026_b.q3_scheme2 import SchemeTwo
from cumcm2026_b.q3_scheme2_localized import SchemeTwoLocalized
from cumcm2026_b.q3_routed_strategies import SchemeOneRouted, SchemeTwoRouted, RoutedOneConfig, RoutedTwoConfig
from cumcm2026_b.q3_geometry import min_distance, search_stations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

ARCHIVE = ROOT / "experiments/第三问/路线调度/六设置完整对照.jsonl"
REPORT = ROOT / "results/tables/第三问/路线调度_六设置对照汇总.json"
VARIANTS = {"s1_old": "方案一原版", "s1_route": "方案一路线规划", "s2_old": "方案二原版",
            "s2_skip": "方案二仅停止重复扫描", "s2_route": "方案二仅路线规划", "s2_both": "方案二两项改进"}
LABELS = ["方案一\n原版", "方案一\n路线规划", "方案二\n原版", "方案二\n仅停止重复扫描", "方案二\n仅路线规划", "方案二\n两项改进"]
FIGURES = ["路线调度_六设置费用对照", "路线调度_逐场收益与代价", "路线调度_同场景轨迹", "路线调度_执行流程"]


def audit(record):
    result, truth = record["结果"], record["真值_仅评估器读取"]
    checked = check_run(result, truth)
    targets = {t["频道"]: t for t in truth["目标"]}
    position = [0., 0.]; channel = 1
    virtual = distance = 0.; measures = switches = successes = failures = skips = 0
    frozen = {}; positives = {}; cleared = set()
    for e in result["动作记录"]:
        c = e.get("频道")
        if e["动作"] in {"检测", "清除"}:
            segment = math.dist(position, e["位置"])
            distance += segment; virtual += round(segment / 5, 6); position = e["位置"]
            target = targets.get(c) if c not in cleared else None
            gap = math.dist(position, target["位置"]) if target else math.inf
            if e["动作"] == "检测":
                measures += 1; switches += c != channel; virtual += 5 + (c != channel); channel = c
                if e["用途"].startswith("固定扫描站"):
                    assert c not in frozen
                if e["结果"] == "direction":
                    assert 5 < gap <= target["接收半径"] + 1e-6
                    angle = math.degrees(math.atan2(target["位置"][1]-position[1], target["位置"][0]-position[0]))
                    assert abs((e["示向度"]-angle+180) % 360 - 180) <= 1.005 + 1e-8
                    positives.setdefault(c, set()).add(tuple(position))
                elif e["结果"] == "near":
                    assert gap <= 5 + 1e-6
                else:
                    assert target is None or gap > target["接收半径"] - 1e-6
            elif e["结果"] == "success":
                assert gap <= 20 + 1e-6
                successes += 1; virtual += 5; cleared.add(c)
            else:
                assert gap > 20 - 1e-6
                failures += 1; virtual += 3
        elif e["动作"] == "定位完成":
            assert e["不同方向站数"] == len(positives[c]) >= 2
            assert set(map(tuple, e["方向站点"])) == positives[c]
            assert np.linalg.norm(np.asarray(e["外包顶点"])-e["覆盖圆心"], axis=1).max() <= 19.5
            assert min_distance(e["外包顶点"], targets[c]["位置"]) <= 1e-5
            frozen[c] = e["步骤"]
        elif e["动作"] == "跳过固定检测":
            assert c not in cleared and e["证书步骤"] == frozen[c] < e["步骤"]
            skips += 1
        elif e["动作"] == "路线规划":
            sequence = [n["剩余站序"] for n in e["计划路径"] if n["类型"] == "固定站"]
            assert sequence == sorted(sequence)
        assert abs(virtual - e["虚拟时间_秒"]) < .001
    assert cleared == set(targets)
    for value, key in [(virtual, "虚拟总时间_秒"), (distance, "总路程_米"), (measures, "检测次数"),
                       (switches, "频道切换次数"), (successes, "清除成功次数"), (failures, "清除失败次数")]:
        assert abs(value - result[key]) < .001, key
    assert skips == result.get("跳过已定位频道检测次数", 0)
    assert result["已完成搜索站"] == list(range(len(result["已完成搜索站"])))
    return {"外包真值核查次数": checked, "动作数": measures + successes + failures,
            "停止检测证书数": len(frozen), "有证书跳过次数": skips}


def run_case(item):
    group, index, case = item
    constructors = {"s1_old": (SchemeOne, None), "s1_route": (SchemeOneRouted, RoutedOneConfig()),
                    "s2_old": (SchemeTwo, None), "s2_skip": (SchemeTwoLocalized, None),
                    "s2_route": (SchemeTwoRouted, RoutedTwoConfig(skip_localized=False)),
                    "s2_both": (SchemeTwoRouted, RoutedTwoConfig())}
    records = []
    for variant, (cls, config) in constructors.items():
        env = make_case(**case)
        client = RobotClient(env, "local-robot")
        try:
            result = cls(client, config).run() if config is not None else cls(client).run()
            record = {"数据组": group, "场景序号": index, "构造参数": case, "设置": variant,
                      "设置说明": VARIANTS[variant], "结果": result, "真值_仅评估器读取": env.truth_for_evaluation()}
            record["独立核验"] = audit(record)
            records.append(record)
        finally:
            client.close()
    assert all(r["真值_仅评估器读取"] == records[0]["真值_仅评估器读取"] for r in records)
    return records


def summarize(records):
    stats = {}; pairs = {}
    for group in sorted({r["数据组"] for r in records}):
        rs = [r for r in records if r["数据组"] == group]
        stats[group] = {}
        for variant in VARIANTS:
            subset = [r["结果"] for r in rs if r["设置"] == variant]
            stats[group][variant] = {"运行数": len(subset), "全清数": sum(r["运行成功"] for r in subset),
                **{f"平均{k}": float(np.mean([r[k] for r in subset])) for k in
                   ["虚拟总时间_秒", "总路程_米", "检测次数", "频道切换次数", "程序运行时间_秒", "平均定位清除时间_秒"]},
                "平均费用分项_秒": {k: float(np.mean([r["时间分项_秒"][k] for r in subset])) for k in subset[0]["时间分项_秒"]},
                "平均顺路清除数": float(np.mean([r.get("路线调度统计", {}).get("顺路证书清除次数", 0) for r in subset])),
                "失败清除总次数": sum(r["清除失败次数"] for r in subset)}
        pairs[group] = {}
        for changed, baseline in [("s1_route", "s1_old"), ("s2_skip", "s2_old"), ("s2_route", "s2_old"),
                                  ("s2_both", "s2_old"), ("s2_both", "s2_skip"), ("s2_both", "s1_route")]:
            a = np.array([r["结果"]["虚拟总时间_秒"] for r in rs if r["设置"] == baseline])
            b = np.array([r["结果"]["虚拟总时间_秒"] for r in rs if r["设置"] == changed])
            delta = b-a
            pairs[group][f"{changed}_vs_{baseline}"] = {"更快场景数": int((delta < -1e-6).sum()),
                "更慢场景数": int((delta > 1e-6).sum()), "持平场景数": int((abs(delta) <= 1e-6).sum()),
                "平均节省秒数": float((-delta).mean()), "平均总时间降低百分比": float((1-b.mean()/a.mean())*100),
                "最差增加秒数": float(delta.max()), "逐场新减旧_秒": delta.tolist()}
    return stats, pairs


def plots(records, stats, pairs):
    style(); plt.rcParams["svg.hashsalt"] = "q3-routed-comparison"
    groups = ["既有构造集", "新增验证集"]
    notice = "本地同场景构造对照，非官方演练或正式成绩；两组分别报告，包含变慢场景"
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, group in zip(axes, groups):
        bottom = np.zeros(6)
        for key, color in zip(["移动", "检测", "切换", "清除成功", "清除失败"],
                              ["#347CA0", "#65A5AE", "#D3A546", "#478C69", "#B85444"]):
            values = [stats[group][v]["平均费用分项_秒"][key] for v in VARIANTS]
            ax.bar(range(6), values, bottom=bottom, color=color, label=key); bottom += values
        for i, value in enumerate(bottom): ax.text(i, value+50, f"{value:.0f}", ha="center", fontsize=9)
        ax.set(xticks=range(6), xticklabels=LABELS, ylabel="平均总虚拟时间（秒）", title=group,
               ylim=(0, bottom.max()*1.2))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.suptitle("两方案路线调度：完整费用与单项改进对照", fontsize=16)
    fig.text(.5, .015, notice, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93)); save(fig, FIGURES[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for i, group in enumerate(groups):
        for j, (key, label) in enumerate([("s1_route_vs_s1_old", "方案一路线规划 − 原版"), ("s2_both_vs_s2_old", "方案二两项改进 − 原版")]):
            delta = pairs[group][key]["逐场新减旧_秒"]
            ax = axes[i, j]
            ax.bar(range(len(delta)), delta, color=["#347E79" if x <= 0 else "#BF7147" for x in delta])
            ax.axhline(0, lw=.8, color="#334155")
            ax.set(title=f"{group}：{label}", xlabel="固定场景序号", ylabel="新版本减原版（秒）")
    fig.suptitle("逐场收益与代价：负值表示更快，正值表示更慢", fontsize=16)
    fig.text(.5, .012, notice, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94)); save(fig, FIGURES[1])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13))
    for ax, variant in zip(axes.flat, ["s1_old", "s1_route", "s2_old", "s2_both"]):
        record = next(r for r in records if r["数据组"] == groups[0] and r["场景序号"] == 0 and r["设置"] == variant)
        result = record["结果"]; old = np.zeros(2)
        for e in result["动作记录"]:
            if e["动作"] not in {"检测", "清除"}: continue
            point = np.asarray(e["位置"])
            color = "#278F78" if e["动作"] == "清除" else "#CB8432" if e["用途"] == "追加测向" else "#347CA0"
            ax.plot([old[0], point[0]], [old[1], point[1]], color=color, lw=1.1, alpha=.8)
            old = point
            if e["动作"] == "清除" and e["结果"] == "success": ax.scatter(*point, s=25, facecolors="white", edgecolors="#278F78", zorder=5)
        fixed = search_stations() if variant.startswith("s1") else np.array(result["布局"]["站点"])
        ax.scatter(fixed[:, 0], fixed[:, 1], marker="s", c="#293E4D", s=20, zorder=4)
        ax.add_patch(Circle((0, 0), 1800, fill=False, ls="--", color="#7B618D", lw=1))
        ax.scatter(0, 0, marker="*", s=80, c="#202B3C", zorder=6)
        ax.set(xlim=(-2050, 2050), ylim=(-2050, 2050), aspect="equal", xlabel="东向 x（米）", ylabel="北向 y（米）",
               title=f"{VARIANTS[variant]}\n{result['总路程_米']/1000:.2f}公里；虚拟时间{result['虚拟总时间_秒']:.1f}秒")
        ax.grid(alpha=.15)
    fig.suptitle("固定使用同一构造场景0：原版与路线调度的实际行动轨迹", fontsize=15)
    fig.text(.5, .026, "蓝：前往固定站；橙：前往追加测向点；绿：前往清除点；圆圈为机器狗清除位置", ha="center", fontsize=9)
    fig.text(.5, .009, notice, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94)); save(fig, FIGURES[2])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=(0, 14), ylim=(0, 8)); ax.axis("off")
    boxes = [(0.3, 5.2, "固定站检测\n更新目标区域与覆盖排除"), (4.9, 5.2, "方案二：已有双站与清除证书\n标为已定位，跳过后续固定检测"),
             (9.5, 5.2, "把已发现目标插入剩余路线\n保留固定站先后顺序"), (9.5, 1.8, "只执行当前路段的一步\n安全清除或有限绕行追加测向"),
             (4.9, 1.8, "更新观测后重新规划\n不合适的目标延后处理"), (0.3, 1.8, "推进下一固定站\n最后处理剩余目标并核查全清")]
    for x, y, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 4.0, 1.45, boxstyle="round,pad=.12", fc="#E6F1F2", ec="#357891"))
        ax.text(x+2, y+.725, label, ha="center", va="center", fontsize=10)
    for a, b in [((4.4, 5.9), (4.75, 5.9)), ((9.0, 5.9), (9.35, 5.9)), ((11.5, 5.0), (11.5, 3.45)),
                 ((9.35, 2.5), (9.05, 2.5)), ((4.75, 2.5), (4.45, 2.5))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, color="#357891"))
    ax.text(7, .7, "未定位目标的代表点仅用于调度预测；实际清除必须有覆盖证书或执行完整有限搜索。", ha="center", fontsize=10)
    fig.suptitle("两方案顺路清除与路线调度流程", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, FIGURES[3])


def source_paths():
    return [ROOT / "src/cumcm2026_b" / name for name in ["q1_geometry.py", "q2_physical_geometry.py", "q3_geometry.py",
            "q3_strategy.py", "q3_scheme2.py", "q3_scheme2_localized.py", "q3_route_planning.py", "q3_routed_strategies.py", "q3_protocol.py", "q3_local_env.py"]] + [
            ROOT / "configs/q3_scheme1_routed.json", ROOT / "configs/q3_scheme2_routed.json",
            ROOT / "scripts/run_q3_scheme1.py", ROOT / "scripts/run_q3_scheme2.py", ROOT / "scripts/build_q3_scheme1_assets.py", Path(__file__)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-set", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.plots_only:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        for name, value in report["来源SHA256"].items():
            if ROOT / name != Path(__file__): assert sha256((ROOT/name).read_bytes()).hexdigest() == value, name
        assert sha256(ARCHIVE.read_bytes()).hexdigest() == report["完整记录SHA256"]
        records = [json.loads(line) for line in ARCHIVE.read_text(encoding="utf-8").splitlines()]
    else:
        if args.cases_per_set < 4 or args.workers < 1: parser.error("每组至少4场，进程数至少1")
        cases = []
        for group, seed in [("既有构造集", 20260911), ("新增验证集", 20261911)]:
            for i in range(args.cases_per_set):
                cases.append((group, i, {"seed": seed+i, "count": [10, 13, 16][i % 3],
                    "layout": ["uniform", "boundary", "cluster", "center"][i % 4],
                    "radius_mode": ["min", "max", "mixed"][(i//3) % 3],
                    "error_mode": ["hash", "zero", "plus", "minus", "alternating"][i % 5]}))
        records = []; ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool, ARCHIVE.open("w", encoding="utf-8", newline="\n") as stream:
            for i, batch in enumerate(pool.map(run_case, cases), 1):
                for record in batch:
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if i % 5 == 0: print(f"已完成{i}/{len(cases)}个场景，每场六设置，均已核验全清与实际费用。", flush=True)
    stats, pairs = summarize(records)
    plots(records, stats, pairs)
    report = {"说明": "本地构造六设置成对比较；新增验证集未用于调参；非官方演练或正式成绩。",
        "统计": stats, "成对比较": pairs, "完整记录": ARCHIVE.relative_to(ROOT).as_posix(),
        "完整记录SHA256": sha256(ARCHIVE.read_bytes()).hexdigest(),
        "来源SHA256": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in source_paths()},
        "核验合计": {k: sum(r["独立核验"][k] for r in records) for k in records[0]["独立核验"]},
        "环境": {"Python": platform.python_version(), "NumPy": np.__version__}, "图表": FIGURES,
        "逐场摘要": [{"数据组": r["数据组"], "场景序号": r["场景序号"], "设置": r["设置"], "构造参数": r["构造参数"],
                      **{k: r["结果"][k] for k in ["运行成功", "清除数", "虚拟总时间_秒", "总路程_米", "检测次数"]}} for r in records]}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"成对比较": {g: {k: {n: v for n, v in p.items() if n != '逐场新减旧_秒'} for k, p in ps.items()} for g, ps in pairs.items()},
                     "核验合计": report["核验合计"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
