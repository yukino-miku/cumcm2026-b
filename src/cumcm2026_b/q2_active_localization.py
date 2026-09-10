"""第二问：在连续可靠检测约束下，以角域交集直径进行鲁棒选点。

物理目标域属于第二问；定位直径仍属于第一问的纯角域模型。利用凸域
的连续方向像将目标与误差的内层搜索降为一维观测角搜索。角度网格及
局部极值细化是数值近似，不据此宣称连续全局最优或严格误差上界。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, minimize_scalar

from .q1_geometry import GeometryResult, bearing_halfplanes, solve_bearings
from .q2_physical_geometry import PhysicalRegion, analytic_candidates, centerline_interval


def serialize_result(value: Any) -> Any:
    """转为严格 JSON 兼容对象；无穷大由 null 和同层状态字段表达。"""
    if isinstance(value, np.ndarray):
        return serialize_result(value.tolist())
    if isinstance(value, dict):
        return {str(key): serialize_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_result(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def linearized_diameter(distance1: float, distance2: float, angle_deg: ArrayLike,
                        half_width_deg: float = 1.0) -> np.ndarray:
    """局部平行条带交集的解析直径；不是有限角楔的精确直径。

    两条条带半宽为 w_i=d_i tan(delta)，两条对角线中较长者给出
    2 sqrt(w1²+w2²+2w1w2|cos(phi)|)/|sin(phi)|。
    """
    if min(distance1, distance2) < 0 or not 0 < half_width_deg < 90:
        raise ValueError("站源距离须非负，角误差半宽须严格介于 0 与 90 度。")
    phi = np.deg2rad(np.asarray(angle_deg, dtype=float))
    widths = np.asarray([distance1, distance2]) * np.tan(np.deg2rad(half_width_deg))
    numerator = 2 * np.sqrt(widths @ widths + 2 * np.prod(widths) * np.abs(np.cos(phi)))
    return np.divide(numerator, np.abs(np.sin(phi)), out=np.full_like(phi, np.inf),
                     where=np.abs(np.sin(phi)) > 1e-14)


def evaluate_detection(region: PhysicalRegion, station2: ArrayLike, target: ArrayLike,
                       error_deg: float = 0.0, *, tol: float = 1e-9) -> GeometryResult:
    """单一真实目标与第二误差场景，直接调用第一问算法；不加入圆域。

    本函数允许评价不能保证接收的基线位置；接收可行性由调用方另查。
    目标恰好位于第二站时无方向，明确报错；5米近距规则由选点层执行。
    """
    second = np.asarray(station2, dtype=float)
    actual_target = np.asarray(target, dtype=float)
    if second.shape != (2,) or actual_target.shape != (2,):
        raise ValueError("检测点和目标必须为二维坐标。")
    if not np.isfinite(second).all() or not np.isfinite(actual_target).all():
        raise ValueError("检测点和目标坐标必须为有限数值。")
    if not region.contains(actual_target, tol=1e-6):
        raise ValueError("目标不属于第一次物理可行域，不能作为该问题的合法场景。")
    if not math.isfinite(error_deg) or abs(error_deg) > region.half_width_deg + 1e-10:
        raise ValueError("第二示向误差超出配置的有界误差范围。")
    displacement = actual_target - second
    if np.linalg.norm(displacement) <= 1e-10:
        raise ValueError("第二检测点与目标重合，示向方向未定义。")
    angle = np.rad2deg(np.arctan2(displacement[1], displacement[0])) + error_deg
    result = solve_bearings([region.station, second], [region.bearing_deg, angle],
                            region.half_width_deg, tol=tol)
    if result.kind == "empty":
        raise RuntimeError("合法目标和误差应产生非空交集；第一问返回空集，需检查数值容差。")
    return result


def batch_two_station_diameters(station1: ArrayLike, bearing1_deg: float,
                               station2: ArrayLike, bearings2_deg: ArrayLike,
                               half_width_deg: float = 1.0,
                               *, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """批量计算两个角楔的纯几何直径，返回直径及区域状态数组。

    复用第一问半平面构造：四边界的六种交点全部枚举并验约束；两个
    正向角楔存在公共方向，当且仅当中心角的圆周距离不大于2delta。
    有可行顶点且存在公共方向则无界，否则最远可行顶点对给出直径。
    无人工大框；空集明确返回 empty，调用鲁棒评价时对异常空集报错。
    顶点筛选默认用1e-10归一化容差，避免窄楔尖端外微小违约交点
    被接受并经几何放大；独立第一问原函数仍使用其既有默认容差。
    """
    first, second = np.asarray(station1, dtype=float), np.asarray(station2, dtype=float)
    angles = np.atleast_1d(np.asarray(bearings2_deg, dtype=float))
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError("第二示向度必须是一维有限数组。")
    # 由第一问负责坐标、角度及半宽输入校验。
    first_A, _ = bearing_halfplanes([first, second], [bearing1_deg, 0.0], half_width_deg)
    scale = max(float(np.linalg.norm(second - first)), 1.0)
    displacement = (second - first) / scale
    count = len(angles)
    A = np.empty((count, 4, 2))
    A[:, :2] = first_A[:2]
    delta = np.deg2rad(half_width_deg)
    theta = np.deg2rad(np.remainder(angles, 360.0))
    A[:, 2] = np.column_stack((np.sin(theta - delta), -np.cos(theta - delta)))
    A[:, 3] = np.column_stack((-np.sin(theta + delta), np.cos(theta + delta)))
    b = np.zeros((count, 4))
    b[:, 2:] = A[:, 2:] @ displacement
    vertices = np.zeros((count, 6, 2))
    valid = np.zeros((count, 6), dtype=bool)
    pair_index = 0
    for i in range(3):
        for j in range(i + 1, 4):
            determinant = A[:, i, 0] * A[:, j, 1] - A[:, i, 1] * A[:, j, 0]
            nonparallel = np.abs(determinant) > 64 * np.finfo(float).eps
            denominator = np.where(nonparallel, determinant, 1.0)
            vertices[:, pair_index, 0] = (b[:, i] * A[:, j, 1] - b[:, j] * A[:, i, 1]) / denominator
            vertices[:, pair_index, 1] = (A[:, i, 0] * b[:, j] - A[:, j, 0] * b[:, i]) / denominator
            slack = np.einsum("nki,ni->nk", A, vertices[:, pair_index]) - b
            valid[:, pair_index] = nonparallel & np.all(slack <= tol, axis=1)
            pair_index += 1
    nonempty = valid.any(axis=1)
    angular_difference = np.abs((angles - bearing1_deg + 180.0) % 360.0 - 180.0)
    # 此容差仅补偿三角函数浮点舍入，不作为人为扩大无界区的角宽。
    recession = angular_difference <= 2.0 * half_width_deg + 2e-12
    diameter_squared = np.zeros(count)
    for i in range(5):
        for j in range(i + 1, 6):
            squared = np.sum((vertices[:, i] - vertices[:, j]) ** 2, axis=1)
            diameter_squared = np.maximum(diameter_squared, np.where(valid[:, i] & valid[:, j], squared, 0.0))
    diameters = np.sqrt(diameter_squared) * scale
    states = np.full(count, "bounded", dtype="<U12")
    states[~nonempty] = "empty"
    states[nonempty & recession] = "unbounded"
    diameters[~nonempty] = np.nan
    diameters[nonempty & recession] = np.inf
    return diameters, states


def _witness(region: PhysicalRegion, second: np.ndarray, observation_rad: float,
             interval: dict[str, Any]) -> tuple[np.ndarray, float]:
    """从最坏观测反解一个合法目标与误差，供第一问独立核验。"""
    true_angle = float(np.clip(observation_rad, interval["lower_rad"], interval["upper_rad"]))
    if abs(true_angle - interval["lower_rad"]) <= 1e-11:
        target = np.asarray(interval["lower_point"], dtype=float)
    elif abs(true_angle - interval["upper_rad"]) <= 1e-11:
        target = np.asarray(interval["upper_point"], dtype=float)
    else:
        direction = np.array([np.cos(true_angle), np.sin(true_angle)])
        ray_range = region.ray_interval(second, direction)
        if ray_range is None:
            raise RuntimeError("连续方向区间内的射线未与物理域相交，需检查边界数值计算。")
        target = second + 0.5 * (ray_range[0] + ray_range[1]) * direction
    return target, float(np.rad2deg(observation_rad - true_angle))


def robust_objective(region: PhysicalRegion, station2: ArrayLike, *, angle_count: int = 129,
                     refine: bool = True, verify_q1: bool = False, radius: float = 1000.0,
                     safety_margin: float = 1e-5, near_radius: float = 5.0) -> dict[str, Any]:
    """数值评价最坏定位直径，精确检测约束与一维角度极大化分开记录。

    当 S2 在凸目标域外，真实方向构成闭区间，其端点由线段端点及圆弧
    切点精确给出。加上误差区间后遍历所有可能观测，等价消去二维G。
    先均匀采样观测区间，再细化采样局部峰；这仍是数值极大值估计。
    """
    second = np.asarray(station2, dtype=float)
    if second.shape != (2,) or not np.isfinite(second).all():
        raise ValueError("第二检测点必须是两个有限坐标。")
    if angle_count < 3 or int(angle_count) != angle_count:
        raise ValueError("观测角采样数必须是至少为 3 的整数。")
    if not np.isfinite([radius, safety_margin, near_radius]).all() or radius <= 0 or safety_margin <= 0 or near_radius < 0:
        raise ValueError("参数须有限，可靠接收半径和安全裕度须为正，近距半径须非负。")
    farthest, nearest = region.max_distance(second), region.min_distance(second)
    result: dict[str, Any] = {
        "station": second, "diameter_m": np.inf, "status": "unbounded",
        "max_distance_m": farthest, "min_distance_m": nearest,
        "reliable_detection": bool(farthest <= radius - safety_margin),
        "reliable_bearing": bool(farthest <= radius - safety_margin and nearest >= near_radius + safety_margin),
        "movement_distance_m": float(np.linalg.norm(second - region.station)),
        "angle_count": int(angle_count), "refine": bool(refine), "q1_verified": False,
        "worst_observation_deg": None, "witness_target": None, "witness_error_deg": None,
        "interpretation_zh": "连续方向区间上的数值极大化，非严格全局最优证明。",
    }
    if nearest <= 1e-9:
        result["status"] = "direction_undefined"
        result["reason_zh"] = "第二站位于目标可行域内，存在目标与检测点重合场景，无法保证示向。"
        return result
    interval = region.angular_interval(second)
    delta = np.deg2rad(region.half_width_deg)
    lower, upper = interval["lower_rad"] - delta, interval["upper_rad"] + delta
    result["true_direction_interval_deg"] = np.rad2deg([interval["lower_rad"], interval["upper_rad"]])
    result["observation_interval_deg"] = np.rad2deg([lower, upper])
    first_angle = np.deg2rad(region.bearing_deg)
    # 若观测区间碰到两楔具有公共前向方向的角区间，则存在严格无界场景。
    unbounded_angle = None
    k_min = math.floor((lower - first_angle - 2 * delta) / (2 * np.pi))
    k_max = math.ceil((upper - first_angle + 2 * delta) / (2 * np.pi))
    for k in range(k_min, k_max + 1):
        left, right = first_angle + 2 * np.pi * k - 2 * delta, first_angle + 2 * np.pi * k + 2 * delta
        if max(lower, left) <= min(upper, right) + 1e-14:
            unbounded_angle = 0.5 * (max(lower, left) + min(upper, right))
            break
    if unbounded_angle is not None:
        worst_angle, worst_value = unbounded_angle, np.inf
    else:
        sample_angles = np.linspace(lower, upper, int(angle_count))
        diameters, states = batch_two_station_diameters(region.station, region.bearing_deg, second,
                                                        np.rad2deg(sample_angles), region.half_width_deg)
        if np.any(states == "empty"):
            bad_angle = float(np.rad2deg(sample_angles[np.flatnonzero(states == "empty")[0]]))
            raise RuntimeError(f"合法连续观测区间出现空交集，观测角为 {bad_angle:.12g} 度。")
        best_index = int(np.argmax(diameters))
        worst_angle, worst_value = float(sample_angles[best_index]), float(diameters[best_index])
        if refine:
            peaks = np.flatnonzero((diameters[1:-1] >= diameters[:-2]) & (diameters[1:-1] >= diameters[2:])) + 1
            for index in peaks:
                def negative_diameter(angle: float) -> float:
                    values, kinds = batch_two_station_diameters(region.station, region.bearing_deg, second,
                                                                [np.rad2deg(angle)], region.half_width_deg)
                    if kinds[0] == "empty":
                        raise RuntimeError("局部观测角细化出现不相容空集，不能吞掉该数值异常。")
                    return -float(values[0])
                maximum = minimize_scalar(negative_diameter,
                                          bounds=(sample_angles[index - 1], sample_angles[index + 1]),
                                          method="bounded", options={"xatol": 1e-11})
                if not maximum.success:
                    raise RuntimeError("内层观测角局部极大化未收敛。")
                if -maximum.fun > worst_value:
                    worst_value, worst_angle = float(-maximum.fun), float(maximum.x)
    target, error = _witness(region, second, worst_angle, interval)
    result.update(diameter_m=worst_value, status="unbounded" if not np.isfinite(worst_value) else "finite",
                  worst_observation_deg=float(np.rad2deg(worst_angle) % 360.0),
                  witness_target=target, witness_error_deg=error)
    if verify_q1:
        checked = evaluate_detection(region, second, target, error)
        if np.isfinite(worst_value):
            if checked.diameter is None or not np.isclose(checked.diameter, worst_value, rtol=2e-7, atol=1e-5):
                raise RuntimeError("第二问批量几何与第一问独立求解的最坏场景直径不一致。")
        elif checked.kind != "unbounded":
            raise RuntimeError("第二问判为无界的场景未得到第一问无界结果。")
        result["q1_verified"] = True
        result["q1_result"] = checked.to_dict()
    return result


def sampled_robust_objective(region: PhysicalRegion, station2: ArrayLike, targets: ArrayLike,
                            *, error_count: int = 5) -> dict[str, Any]:
    """独立二维目标×误差离散实验，用于检查方向降维与采样收敛。

    保留边界及内部目标，无概率含义；该有限样本最大值只是真实最坏值
    的下界近似，不能据此保证检测或证明连续最坏误差只在端点。
    """
    second, scenarios = np.asarray(station2, dtype=float), np.asarray(targets, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != 2 or len(scenarios) == 0:
        raise ValueError("目标场景须为非空的 (N,2) 坐标数组。")
    if not np.all(region.contains(scenarios, tol=1e-6)):
        raise ValueError("存在物理域外的目标场景。")
    if error_count < 3 or int(error_count) != error_count:
        raise ValueError("误差采样数量必须是至少为 3 的整数。")
    vectors = scenarios - second
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-10):
        raise ValueError("目标场景含与第二站重合的点，方向未定义。")
    errors = np.linspace(-region.half_width_deg, region.half_width_deg, error_count)
    directions = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    observations = directions[:, None] + errors[None, :]
    values, states = batch_two_station_diameters(region.station, region.bearing_deg, second,
                                                observations.ravel(), region.half_width_deg)
    if np.any(states == "empty"):
        raise RuntimeError("合法目标×误差采样出现空交集，需检查几何容差。")
    index = int(np.argmax(values))
    target_index, error_index = np.unravel_index(index, observations.shape)
    return {"diameter_m": float(values[index]), "status": "finite" if np.isfinite(values[index]) else "unbounded",
            "target_count": len(scenarios), "error_count": int(error_count), "scenario_count": len(values),
            "witness_target": scenarios[target_index], "witness_error_deg": float(errors[error_index]),
            "sample_mean_m": float(np.mean(values)), "sample_median_m": float(np.median(values)),
            "interpretation_zh": "有限构造场景统计，无题设概率含义。"}


def _frame(region: PhysicalRegion) -> np.ndarray:
    theta = np.deg2rad(region.bearing_deg)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def candidate_region_statistics(evaluations: list[dict[str, Any]], best_value: float,
                                grid_diameters: np.ndarray, grid_step: float,
                                etas: tuple[float, ...] = (0.02, 0.05, 0.10)) -> dict[str, Any]:
    """近优点集供第三问直接调用；面积仅用统一规则格网估计。"""
    result = {}
    for eta in etas:
        if eta < 0:
            raise ValueError("近最优相对容差不得为负。")
        indices = [i for i, item in enumerate(evaluations)
                   if item["reliable_bearing"] and item["diameter_m"] <= (1 + eta) * best_value]
        movements = [evaluations[i]["movement_distance_m"] for i in indices]
        count = int(np.count_nonzero(grid_diameters <= (1 + eta) * best_value))
        result[f"{eta:.2f}"] = {"eta": eta, "point_indices": indices, "point_count": len(indices),
                                "grid_count": count, "estimated_area_m2": count * grid_step ** 2,
                                "movement_range_m": [min(movements), max(movements)] if movements else None,
                                "area_note_zh": "统一格网中心计数×步长平方，非连续面积精确值；局部细化点不参与面积计数。"}
    return result


def search_second_station(region: PhysicalRegion, *, grid_step: float = 25.0,
                          angle_count: int = 65, local_steps: tuple[float, ...] = (10.0, 2.0, 0.4),
                          eta: float = 0.05, radius: float = 1000.0, safety_margin: float = 1e-5,
                          near_radius: float = 5.0, use_slsqp: bool = True,
                          final_angle_count: int = 257) -> dict[str, Any]:
    """规则二维格网、两侧局部格网及受约束连续微调，输出近优候选点集。

    搜索的是 K 内且到 Omega 的最小距离大于5米的可示向子域。局部
    微调只改善数值解，既不把SLSQP成功当成全局最优证明，也不将规则
    格网面积当成连续候选域的严格面积。
    """
    if not np.isfinite([grid_step, *local_steps]).all() or grid_step <= 0 or any(step <= 0 for step in local_steps):
        raise ValueError("全局和局部搜索步长必须为正。")
    if not np.isfinite([radius, safety_margin, near_radius, eta]).all() or radius <= 0 or safety_margin <= 0 or near_radius < 0 or eta < 0:
        raise ValueError("可靠半径、安全裕度须为有限正数，近距半径与候选阈值须为有限非负数。")
    if (angle_count < 3 or int(angle_count) != angle_count or final_angle_count < angle_count
            or int(final_angle_count) != final_angle_count or (final_angle_count - 1) % (angle_count - 1) != 0):
        raise ValueError("角网格须至少3点；最终角网格须包含粗角网格，例如65点加密为257点。")
    rotation = _frame(region)
    interval = centerline_interval(region)
    if interval is None:
        anchor = np.asarray(region.representative_point)
    else:
        anchor = region.station + np.mean(interval) * rotation[:, 0]
    anchor_local = (anchor - region.station) @ rotation
    # K中任意点均在B(anchor,R)内，因此这个矩形覆盖整个K，未截掉可行点。
    x = np.arange(math.ceil((anchor_local[0] - radius) / grid_step),
                  math.floor((anchor_local[0] + radius) / grid_step) + 1) * grid_step
    y = np.arange(math.ceil((anchor_local[1] - radius) / grid_step),
                  math.floor((anchor_local[1] + radius) / grid_step) + 1) * grid_step
    diameters = np.full((len(y), len(x)), np.inf)
    reliable = np.zeros_like(diameters, dtype=bool)
    directional = np.zeros_like(diameters, dtype=bool)
    records: list[dict[str, Any]] = []
    cache: dict[tuple[float, float], dict[str, Any]] = {}

    def evaluate_local(local: ArrayLike) -> dict[str, Any]:
        local_point = np.asarray(local, dtype=float)
        key = tuple(np.round(local_point, 10))
        if key in cache:
            return cache[key]
        point = region.station + rotation @ local_point
        maximum, minimum = region.max_distance(point), region.min_distance(point)
        can_receive = maximum <= radius - safety_margin
        can_bear = can_receive and minimum >= near_radius + safety_margin
        if can_bear:
            evaluated = robust_objective(region, point, angle_count=angle_count, refine=False,
                                         radius=radius, safety_margin=safety_margin, near_radius=near_radius)
        else:
            evaluated = {"station": point, "diameter_m": np.inf, "status": "outside_reliable_bearing_domain",
                         "max_distance_m": maximum, "min_distance_m": minimum,
                         "reliable_detection": bool(can_receive), "reliable_bearing": False,
                         "movement_distance_m": float(np.linalg.norm(point - region.station))}
        evaluated["local_station"] = local_point.copy()
        cache[key] = evaluated
        records.append(evaluated)
        return evaluated

    for iy, y_value in enumerate(y):
        for ix, x_value in enumerate(x):
            evaluation = evaluate_local([x_value, y_value])
            diameters[iy, ix] = evaluation["diameter_m"]
            reliable[iy, ix] = evaluation["reliable_detection"]
            directional[iy, ix] = evaluation["reliable_bearing"]
    analytic = analytic_candidates(region, radius=radius) if interval is not None else np.empty((0, 2))
    for candidate in analytic:
        evaluate_local((candidate - region.station) @ rotation)
    side_optima = []
    local_optimization = []
    for sign in (-1, 1):
        def available():
            return [item for item in records if item["local_station"][1] * sign > 0
                    and item["reliable_bearing"] and np.isfinite(item["diameter_m"])]
        candidates = available()
        if not candidates:
            continue
        # 在各侧最优的两个粗候选邻域加密，避免单起点局部网格遗漏次级谷。
        starts = sorted(candidates, key=lambda item: item["diameter_m"])[:2]
        for start in starts:
            current = start
            for step in local_steps:
                center = current["local_station"].copy()
                neighbourhood = [evaluate_local(center + [i * step, j * step])
                                 for i in range(-3, 4) for j in range(-3, 4)]
                current = min([current] + [item for item in neighbourhood
                                           if item["local_station"][1] * sign > 0],
                              key=lambda item: item["diameter_m"])
        side_best = min(available(), key=lambda item: item["diameter_m"])
        if use_slsqp:
            def objective(local: np.ndarray) -> float:
                point = region.station + rotation @ local
                if region.min_distance(point) <= 1e-7:
                    return 1e9
                evaluated = robust_objective(region, point, angle_count=angle_count, refine=False,
                                             radius=radius, safety_margin=safety_margin, near_radius=near_radius)
                return evaluated["diameter_m"] if np.isfinite(evaluated["diameter_m"]) else 1e9
            constraints = [
                {"type": "ineq", "fun": lambda local: radius - 2 * safety_margin - region.max_distance(region.station + rotation @ local)},
                {"type": "ineq", "fun": lambda local: region.min_distance(region.station + rotation @ local) - near_radius - 2 * safety_margin},
                {"type": "ineq", "fun": lambda local: sign * local[1] - safety_margin},
            ]
            refined = minimize(objective, side_best["local_station"], method="SLSQP", constraints=constraints,
                               bounds=[(float(x.min()), float(x.max())), (float(y.min()), float(y.max()))],
                               options={"ftol": 1e-9, "maxiter": 80, "eps": 1e-3})
            proposal = evaluate_local(refined.x)
            local_optimization.append({"side": sign, "success": bool(refined.success),
                                       "message": str(refined.message), "iterations": int(refined.nit),
                                       "accepted": bool(proposal["reliable_bearing"] and proposal["diameter_m"] <= side_best["diameter_m"])})
            if proposal["reliable_bearing"] and proposal["diameter_m"] <= side_best["diameter_m"]:
                side_best = proposal
        checked = robust_objective(region, side_best["station"], angle_count=final_angle_count,
                                   refine=True, verify_q1=True, radius=radius, safety_margin=safety_margin,
                                   near_radius=near_radius)
        checked["local_station"] = side_best["local_station"].copy()
        side_best.update(checked)
        side_optima.append(side_best)
    if not side_optima:
        raise RuntimeError("当前格网未找到有限且可靠的二次示向位置；应加密格网或检查题面场景。")
    best = min(side_optima, key=lambda item: item["diameter_m"])
    thresholds = tuple(sorted(set((0.02, 0.05, 0.10, float(eta)))))
    # 所有潜在近优点均用最终密度重评，防止粗角度采样将质量估计得过好。
    # 只验证阈值内点即可：粗网格极大值不大于加密的嵌套网格极大值，
    # 最终解不会因加密使原先阈值外点变为阈值内。网格数应选2^k+1。
    near_validation_count = 0
    for item in records:
        if item["reliable_bearing"] and item["diameter_m"] <= (1 + max(thresholds)) * best["diameter_m"]:
            if not (item.get("angle_count") == final_angle_count and item.get("refine")):
                checked = robust_objective(region, item["station"], angle_count=final_angle_count, refine=True,
                                           radius=radius, safety_margin=safety_margin, near_radius=near_radius)
                item.update(checked)
                near_validation_count += 1
    side_optima = []
    for sign in (-1, 1):
        eligible = [item for item in records if item["local_station"][1] * sign > 0
                    and item["reliable_bearing"] and np.isfinite(item["diameter_m"])]
        if eligible:
            selected = min(eligible, key=lambda item: item["diameter_m"])
            if not selected.get("q1_verified"):
                selected.update(robust_objective(region, selected["station"], angle_count=final_angle_count,
                                                refine=True, verify_q1=True, radius=radius,
                                                safety_margin=safety_margin, near_radius=near_radius))
            side_optima.append(selected)
    best = min(side_optima, key=lambda item: item["diameter_m"])
    # 面积和热力图同步采用更新后的规则格网数值，不混用旧密度结果。
    for iy, y_value in enumerate(y):
        for ix, x_value in enumerate(x):
            diameters[iy, ix] = cache[tuple(np.round([x_value, y_value], 10))]["diameter_m"]
    candidates = candidate_region_statistics(records, best["diameter_m"], diameters, grid_step, thresholds)
    return {"best": best, "side_optima": side_optima, "analytic_candidates": analytic,
            "grid": {"x_local_m": x, "y_local_m": y, "diameter_m": diameters,
                     "reliable_detection": reliable, "reliable_bearing": directional},
            "evaluations": records, "candidate_regions": candidates,
            "reliable_area_m2": int(reliable.sum()) * grid_step ** 2,
            "reliable_bearing_area_m2": int(directional.sum()) * grid_step ** 2,
            "local_optimization": local_optimization,
            "near_optimal_validation_count": near_validation_count,
            "analytic_status": "available" if interval is not None else "centerline_misses_target_domain",
            "configuration": {"grid_step_m": grid_step, "angle_count": angle_count,
                              "local_steps_m": local_steps, "final_angle_count": final_angle_count,
                              "eta": eta, "reception_lower_m": radius, "safety_margin_m": safety_margin,
                              "near_radius_m": near_radius, "use_slsqp": use_slsqp,
                              "model_note_zh": "物理域与可靠约束连续求值；鲁棒角度和外层选点为网格及局部细化近似。"}}
