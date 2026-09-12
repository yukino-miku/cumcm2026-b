"""原第四问几何方案的补测间距扩展；不改变前三问或历史Q4核心模块。"""
from dataclasses import dataclass
import math
import time
import numpy as np
from .q4_strategy import Q4Config, SchemeFour
from .q3_strategy import ChannelState, observation_plan
from .q2_physical_geometry import build_physical_region, analytic_candidates, centerline_interval
from .q3_geometry import (DELTA, search_stations, max_distance, min_distance, principal_axes,
                          angular_interval, add_wedge, finish_bound)


@dataclass
class SpacingConfig(Q4Config):
    probe_spacing_m: float = 0.

    def __post_init__(self):
        super().__post_init__()
        v = self.probe_spacing_m
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError('probe_spacing_m必须是非负有限数')


def spaced_observation_plan(channel: ChannelState, current, config: SpacingConfig):
    """保留原几何候选及评分，仅增加相邻站间距筛选；不足时回退原候选。"""
    started = time.monotonic()
    vertices = channel.region.vertices
    direct_bound = finish_bound(vertices, current)
    if channel.localization_count >= config.max_localization_measurements or config.planning_seconds == 0:
        return {"动作": "覆盖清除", "预计费用_秒": direct_bound, "原因": "有限测向预算回退"}
    center, axes = principal_axes(vertices)
    candidates = []
    # 第二问候选只作生成，仍对当前全部历史外包重新检查可靠接收。
    positives = [o for o in channel.region.observations if o["结果"] == "direction"]
    if positives:
        first = positives[0]
        physical = build_physical_region(first["位置"], first["示向度"], DELTA)
        # 首次角楔可与目标圆相交，而中心射线恰好落在圆外；此时跳过透镜候选。
        if centerline_interval(physical) is not None:
            for p in analytic_candidates(physical):
                candidates.extend([center+.8*(p-center), center+.9*(p-center)])
    for angle in np.arange(8)*np.pi/4:
        direction = axes@np.array([math.cos(angle), math.sin(angle)])
        candidates.extend(center+distance*direction for distance in [30, 75, 150, 300, 500, 650])
    candidates.extend(search_stations())
    feasible = []
    for point in candidates:
        point = np.asarray(point)
        if (max_distance(vertices, point) <= 999.5 and min_distance(vertices, point) > 5.5
                and all(np.linalg.norm(point-old) > .01 for old in channel.measured_sites)):
            if all(np.linalg.norm(point-old) > .01 for old in feasible): feasible.append(point)
    original_count = len(feasible)
    separated = [p for p in feasible
                 if np.linalg.norm(p-current) >= config.probe_spacing_m-1e-7
                 and (not channel.measured_sites or
                      np.linalg.norm(p-np.asarray(channel.measured_sites[-1])) >= config.probe_spacing_m-1e-7)]
    relaxed = not separated and bool(feasible)
    if separated:
        feasible = separated
    # 没有满足间距的候选时退回原集合；不因间距要求直接放弃已发现源的定位。
    # 排序只决定有限预算内考察次序，不作为最优性证明。
    feasible.sort(key=lambda p: np.linalg.norm(p-current))
    best = {"动作": "覆盖清除", "预计费用_秒": direct_bound, "原因": "当前有限覆盖费用更低"}
    evaluated = 0
    for point in feasible[:config.candidate_limit]:
        if time.monotonic()-started >= config.planning_seconds: break
        low, high = angular_interval(vertices, point)
        delta = math.radians(DELTA)
        predictions = []
        for theta in np.linspace(low-delta, high+delta, config.angle_samples):
            branch = add_wedge(vertices, point, math.degrees(theta))
            if len(branch): predictions.append(finish_bound(branch, point))
        if not predictions: continue
        evaluated += 1
        # 同目标频道通常已在当前，频道切换由外层执行器精确计入总时钟。
        score = float(np.linalg.norm(point-current)/5+5+max(predictions))
        if score < best["预计费用_秒"]:
            best = {"动作": "追加测向", "位置": point.tolist(), "预计费用_秒": score,
                    "最远可能目标距离_米": max_distance(vertices, point),
                    "最近可能目标距离_米": min_distance(vertices, point),
                    "后续费用采样最大值_秒": max(predictions)}
    return {**best, "建议相邻补测间距_米": config.probe_spacing_m,
            "间距约束已放宽": relaxed, "间距筛选前候选数": original_count,
            "间距筛选后候选数": len(feasible), "已评估候选数": evaluated, "预测角节点": config.angle_samples,
            "规划耗时_秒": time.monotonic()-started, "评分性质": "有限角采样预测；非连续极值保证"}


class SpacedFour(SchemeFour):
    def __init__(self, client, config=None):
        super().__init__(client, config or SpacingConfig())

    def _observation_plan(self, state, current):
        if self.config.probe_spacing_m == 0:
            return super()._observation_plan(state, current)
        plan = spaced_observation_plan(state, current, self.config)
        plan['评分性质'] = '原几何候选与角采样评分，优先拉开相邻补测站；不保证发现概率或下一站接收'
        if plan['动作'] == '追加测向':
            point = np.asarray(plan['位置'])
            plan['无信号后的保守回退费用_秒'] = finish_bound(state.region.vertices, point)
            plan['接收保证'] = False
            plan['距当前机器人_米'] = float(np.linalg.norm(point-current))
            plan['距该频道上次观测_米'] = float(np.linalg.norm(point-np.asarray(state.measured_sites[-1]))) if state.measured_sites else None
        return plan
