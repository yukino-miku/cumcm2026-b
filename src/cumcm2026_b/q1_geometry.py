"""第一问：示向误差角域求交、区域直径与同直径圆覆盖判定。

示向度以 x 轴正方向为零度，逆时针为正。本模块只使用题目给定的
角度有界误差，不假定概率分布，也不将 1800 米目标圆域叠加为约束。
所有数值判定在平移、缩放后的坐标中进行；不使用大矩形截断无界区域。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


Array = NDArray[np.float64]
RegionKind = Literal["empty", "unbounded", "polygon", "segment", "point"]
KIND_ZH = {
    "empty": "空集",
    "unbounded": "无界区域",
    "polygon": "有界凸多边形",
    "segment": "线段",
    "point": "单点",
}


@dataclass
class GeometryResult:
    """求解结果；长度均已恢复到输入坐标的单位。

    cover_radius 仅指 D/2，不是最小覆盖圆的半径。
    tolerance 是归一化坐标容差，absolute_tolerance 为原坐标长度容差。
    空集、无界区域没有有限直径或圆覆盖结果，相应字段为 None。
    """

    kind: RegionKind
    vertices: Array
    diameter: float | None
    diameter_pair: Array | None
    circle_center: Array | None
    cover_radius: float | None
    max_center_distance: float | None
    coverage_excess: float | None
    covered: bool | None
    A: Array
    b: Array
    normalization_origin: Array
    normalization_scale: float
    tolerance: float
    feasible_point: Array | None

    @property
    def absolute_tolerance(self) -> float:
        return self.tolerance * self.normalization_scale

    def to_dict(self) -> dict[str, Any]:
        """转换为可严格序列化的 JSON 对象，并附中文区域名称。"""
        result: dict[str, Any] = {}
        for key, value in vars(self).items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        result["kind_zh"] = KIND_ZH[self.kind]
        result["absolute_tolerance"] = self.absolute_tolerance
        return result


def bearing_halfplanes(
    stations: ArrayLike,
    bearings_deg: ArrayLike,
    half_width_deg: float | ArrayLike = 1.0,
) -> tuple[Array, Array]:
    """把前向角楔转换为 A @ X <= b，每个检测点贡献两条约束。

    half_width_deg 可以是共同半宽，也可以为每站分别提供半宽。
    要求 0 < 半宽 < 90 度，这保证两个半平面正好表示前向凸角楔。
    """
    stations_array = np.asarray(stations, dtype=float)
    if stations_array.size == 0:
        stations_array = np.empty((0, 2), dtype=float)
    if stations_array.ndim != 2 or stations_array.shape[1] != 2:
        raise ValueError("检测点必须是形状为 (n, 2) 的有限坐标数组。")
    bearings_array = np.asarray(bearings_deg, dtype=float)
    if bearings_array.ndim != 1 or len(bearings_array) != len(stations_array):
        raise ValueError("示向度必须是一维数组，且数量与检测点相同。")
    try:
        widths = np.broadcast_to(np.asarray(half_width_deg, dtype=float), bearings_array.shape)
    except ValueError as error:
        raise ValueError("误差半宽须为单个数值或与检测点等长的数组。") from error
    if not (
        np.isfinite(stations_array).all()
        and np.isfinite(bearings_array).all()
        and np.isfinite(widths).all()
    ):
        raise ValueError("检测点、示向度和误差半宽必须都是有限数值。")
    if np.any(widths <= 0) or np.any(widths >= 90):
        raise ValueError("角楔误差半宽必须严格介于 0 度与 90 度之间。")
    angles = np.deg2rad(np.remainder(bearings_array, 360.0))
    deltas = np.deg2rad(widths)
    lower = angles - deltas
    upper = angles + deltas
    # cross(u_-, X-S) >= 0；cross(u_+, X-S) <= 0。
    A = np.empty((2 * len(stations_array), 2), dtype=float)
    A[0::2] = np.column_stack((np.sin(lower), -np.cos(lower)))
    A[1::2] = np.column_stack((-np.sin(upper), np.cos(upper)))
    b = np.einsum("ij,ij->i", A, np.repeat(stations_array, 2, axis=0))
    return A, b


def _deduplicate(points: Array, tol: float) -> Array:
    """用局部网格去掉容差内重合交点，避免全体点两两去重的四次复杂度。"""
    cells: dict[tuple[int, int], list[Array]] = {}
    kept: list[Array] = []
    for point in points:
        cell = (math.floor(point[0] / tol), math.floor(point[1] / tol))
        duplicate = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), []):
                    if np.linalg.norm(point - other) <= tol:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            kept.append(point)
            cells.setdefault(cell, []).append(point)
    return np.asarray(kept, dtype=float).reshape(-1, 2)


def _cross(a: Array, b: Array) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _convex_hull(points: Array, tol: float) -> Array:
    """Andrew 单调链；返回逆时针顶点，线段只返回两个端点。"""
    if len(points) <= 1:
        return points.copy()
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half_chain(sequence: Sequence[Array]) -> list[Array]:
        chain: list[Array] = []
        for point in sequence:
            while len(chain) >= 2:
                edge = chain[-1] - chain[-2]
                # 叉积除以边长为垂距，使容差保持长度量纲。
                if _cross(edge, point - chain[-2]) > tol * np.linalg.norm(edge):
                    break
                chain.pop()
            chain.append(point)
        return chain

    lower = half_chain(ordered)
    upper = half_chain(ordered[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _farthest_pair(vertices: Array) -> tuple[float, Array]:
    """枚举顶点对，时间复杂度 O(m²)，额外空间 O(m)。"""
    if len(vertices) == 1:
        return 0.0, np.repeat(vertices, 2, axis=0)
    largest_squared = -1.0
    selected = (0, 1)
    for i in range(len(vertices) - 1):
        displacements = vertices[i + 1 :] - vertices[i]
        squared = np.einsum("ij,ij->i", displacements, displacements)
        relative_j = int(np.argmax(squared))
        if squared[relative_j] > largest_squared:
            largest_squared = float(squared[relative_j])
            selected = (i, i + 1 + relative_j)
    return math.sqrt(max(0.0, largest_squared)), vertices[list(selected)].copy()


def solve_halfplanes(
    A: ArrayLike,
    b: ArrayLike,
    *,
    tol: float = 1e-9,
    origin: ArrayLike | None = None,
    scale: float | None = None,
) -> GeometryResult:
    """求二维闭半平面交、直径和同直径圆覆盖判定。

    先以线性规划判定可行性，再沿正负两坐标轴检查有界性。
    有界时枚举边界线对并验证全部约束，因此 n 条约束的交点阶段
    为 O(n³)，不是 O(n log n) 的专门半平面交算法。

    默认平移原点取单位法向量系统的最小二乘解；默认尺度取平移后
    边界距离的最大值，精确退化时取 1。可显式传入 origin 和 scale。
    tol 同时用于归一化后的 LP 可行性、顶点筛选及几何判断。
    """
    matrix = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float)
    if matrix.size == 0:
        matrix = np.empty((0, 2), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("A 必须是形状为 (n, 2) 的数组。")
    if rhs.ndim != 1 or len(rhs) != len(matrix):
        raise ValueError("b 必须是一维数组，长度与 A 的行数相同。")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("半平面系数必须为有限数值。")
    if not np.isfinite(tol) or not (1e-10 <= tol <= 1e-4):
        raise ValueError("归一化容差必须位于 [1e-10, 1e-4]，以匹配线性规划求解器。")

    row_norms = np.hypot(matrix[:, 0], matrix[:, 1])
    nonzero = row_norms > 0
    contradictory_zero_row = bool(np.any(rhs[~nonzero] < 0))
    unit_A = matrix[nonzero] / row_norms[nonzero, None]
    unit_b = rhs[nonzero] / row_norms[nonzero]
    if origin is None:
        shift = np.linalg.lstsq(unit_A, unit_b, rcond=None)[0] if len(unit_A) else np.zeros(2)
    else:
        shift = np.asarray(origin, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("归一化原点必须是两个有限数值。")
    shifted_b = unit_b - unit_A @ shift
    if scale is None:
        characteristic = float(np.max(np.abs(shifted_b))) if len(shifted_b) else 0.0
        # 边界同交一点时，最小二乘的舍入误差不应成为极小的缩放尺度。
        roundoff = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(unit_b), initial=0.0)))
        factor = characteristic if characteristic > roundoff else 1.0
    else:
        factor = float(scale)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("归一化尺度必须为正的有限数值。")
    normalized_b = shifted_b / factor

    def result_for(kind: RegionKind, feasible: Array | None = None) -> GeometryResult:
        return GeometryResult(
            kind=kind,
            vertices=np.empty((0, 2), dtype=float),
            diameter=None,
            diameter_pair=None,
            circle_center=None,
            cover_radius=None,
            max_center_distance=None,
            coverage_excess=None,
            covered=None,
            A=matrix.copy(),
            b=rhs.copy(),
            normalization_origin=shift.copy(),
            normalization_scale=factor,
            tolerance=tol,
            feasible_point=None if feasible is None else feasible * factor + shift,
        )

    if contradictory_zero_row:
        return result_for("empty")
    if not len(unit_A):
        return result_for("unbounded", np.zeros(2))

    lp_options = {"primal_feasibility_tolerance": tol, "dual_feasibility_tolerance": tol}

    def linear_program(objective: ArrayLike):
        # 必须显式释放变量上下界：SciPy 默认 x>=0 会错误删去负坐标区域。
        return linprog(
            objective,
            A_ub=unit_A,
            b_ub=normalized_b,
            bounds=[(None, None), (None, None)],
            method="highs",
            options=lp_options,
        )

    feasibility = linear_program([0.0, 0.0])
    if feasibility.status == 2:
        return result_for("empty")
    if not feasibility.success:
        raise RuntimeError(f"线性规划未能可靠判定可行性：状态 {feasibility.status}，{feasibility.message}")
    feasible = np.asarray(feasibility.x, dtype=float)
    for objective in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        extremum = linear_program(objective)
        if extremum.status == 3:
            return result_for("unbounded", feasible)
        if not extremum.success:
            raise RuntimeError(f"线性规划未能可靠判定有界性：状态 {extremum.status}，{extremum.message}")

    candidates: list[Array] = []
    for i in range(len(unit_A) - 1):
        for j in range(i + 1, len(unit_A)):
            determinant = _cross(unit_A[i], unit_A[j])
            # 仅以机器精度识别数值平行；不能用长度容差删掉窄交会角。
            if abs(determinant) <= 64 * np.finfo(float).eps:
                continue
            point = np.linalg.solve(unit_A[[i, j]], normalized_b[[i, j]])
            if np.all(unit_A @ point - normalized_b <= tol):
                candidates.append(point)
    if not candidates:
        raise RuntimeError("LP 判定交集有界，但未得到可信顶点；请检查病态约束并调整坐标单位或容差。")
    unique = _deduplicate(np.asarray(candidates), tol)
    vertices = _convex_hull(unique, tol)
    diameter, pair = _farthest_pair(vertices)
    if diameter <= tol:
        # 候选点均可行，均值仍可行；将容差内退化结果统一记为单点。
        vertices = np.mean(unique, axis=0, keepdims=True)
        diameter, pair = _farthest_pair(vertices)
        kind: RegionKind = "point"
    elif len(vertices) <= 2:
        vertices = pair.copy()
        kind = "segment"
    else:
        perpendicular = np.abs(
            (vertices[:, 0] - pair[0, 0]) * (pair[1, 1] - pair[0, 1])
            - (vertices[:, 1] - pair[0, 1]) * (pair[1, 0] - pair[0, 0])
        ) / diameter
        if float(np.max(perpendicular)) <= tol:
            vertices = pair.copy()
            kind = "segment"
        else:
            kind = "polygon"

    center = np.mean(pair, axis=0)
    radius = diameter / 2.0
    max_distance = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    covered = bool(max_distance <= radius + tol)
    result = result_for(kind, feasible)
    result.vertices = vertices * factor + shift
    result.diameter = diameter * factor
    result.diameter_pair = pair * factor + shift
    result.circle_center = center * factor + shift
    result.cover_radius = radius * factor
    result.max_center_distance = max_distance * factor
    result.coverage_excess = max(0.0, max_distance - radius) * factor
    result.covered = covered
    return result


def solve_bearings(
    stations: ArrayLike,
    bearings_deg: ArrayLike,
    half_width_deg: float | ArrayLike = 1.0,
    *,
    tol: float = 1e-9,
) -> GeometryResult:
    """从检测点和示向度直接求解；0/360 度跨越由正弦余弦自然处理。"""
    station_array = np.asarray(stations, dtype=float)
    if station_array.size == 0:
        station_array = np.empty((0, 2), dtype=float)
    # 先验证输入；随后在中心化坐标构造 b，降低大平移的相消误差。
    A, b = bearing_halfplanes(station_array, bearings_deg, half_width_deg)
    origin = np.mean(station_array, axis=0) if len(station_array) else np.zeros(2)
    centered = station_array - origin
    scale = float(np.max(np.linalg.norm(centered, axis=1))) if len(centered) else 0.0
    if scale == 0:
        scale = 1.0
    local_b = np.einsum("ij,ij->i", A, np.repeat(centered / scale, 2, axis=0))
    result = solve_halfplanes(A, local_b, tol=tol, origin=[0.0, 0.0], scale=1.0)
    result.vertices = result.vertices * scale + origin
    for attribute in ("diameter_pair", "circle_center", "feasible_point"):
        value = getattr(result, attribute)
        if value is not None:
            setattr(result, attribute, value * scale + origin)
    for attribute in ("diameter", "cover_radius", "max_center_distance", "coverage_excess"):
        value = getattr(result, attribute)
        if value is not None:
            setattr(result, attribute, value * scale)
    result.A = A
    result.b = b
    result.normalization_origin = origin
    result.normalization_scale = scale
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="第一问：示向交会区域、直径与同直径圆覆盖计算。")
    parser.add_argument("--cases", type=Path, required=True, help="中文说明的案例 JSON 文件")
    parser.add_argument("--output", type=Path, help="输出结果 JSON 文件；省略则显示到终端")
    arguments = parser.parse_args()
    configuration = json.loads(arguments.cases.read_text(encoding="utf-8-sig"))
    outputs = []
    for case in configuration["cases"]:
        solved = solve_bearings(case["stations"], case["bearings_deg"], case.get("half_width_deg", 1.0))
        outputs.append({"id": case["id"], "name_zh": case["name_zh"], "result": solved.to_dict()})
    text = json.dumps({"说明": "第一问人工构造几何验证案例，并非模拟器实测数据。", "cases": outputs}, ensure_ascii=False, indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
        print(f"已完成 {len(outputs)} 个几何案例，结果保存至：{arguments.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
