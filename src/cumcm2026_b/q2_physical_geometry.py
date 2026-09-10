"""第二问物理层：窄角域与圆盘的交集、连续距离和可靠检测域。

边界用有限条直线段和圆弧保存。距离、方向和面积查询直接使用这些
边界的解析极值；采样只供绘图、数值场景和候选点搜索使用。
本模块不修改第一问纯角域定位区域的定义。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .q1_geometry import bearing_halfplanes

Array = NDArray[np.float64]
TAU = 2.0 * math.pi


def _point(value: ArrayLike, name: str = "坐标") -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name}必须由两个有限数值组成。")
    return result


def _unique(points: list[Array] | Array, tol: float) -> Array:
    """邻格去重，避免密集场景采样时逐点两两比较。"""
    kept: list[Array] = []
    cells: dict[tuple[int, int], list[Array]] = {}
    for point in points:
        key = (math.floor(point[0] / tol), math.floor(point[1] / tol))
        duplicate = any(np.linalg.norm(point - other) <= tol for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1) for other in cells.get((key[0] + dx, key[1] + dy), []))
        if not duplicate:
            value = np.asarray(point, dtype=float)
            kept.append(value)
            cells.setdefault(key, []).append(value)
    return np.asarray(kept, dtype=float).reshape(-1, 2)


def _on_arc(angle: float, start: float, end: float, tol: float = 1e-12) -> bool:
    return bool((angle - start) % TAU <= end - start + tol or abs((angle - start) % TAU - TAU) <= tol)


@dataclass
class PhysicalRegion:
    """第一次测向的保守物理可行域；几何内部坐标以第一站为原点。

公开的 segments、arcs、vertices 均使用题面全局坐标。内部局部坐标
减少大平移带来的浮点相消。保留第一站五米内区域是显式保守扩张。
    """

    station: Array
    bearing_deg: float
    half_width_deg: float
    target_center: Array
    target_radius: float
    reception_upper: float
    A: Array
    local_b: Array
    circles: list[dict[str, Any]]
    _segments: list[dict[str, Array]]
    _arcs: list[dict[str, Any]]
    _vertices: Array
    tolerance: float = 1e-8

    @property
    def vertices(self) -> Array:
        """边界连接点；圆弧内部极值不一定出现在这些点上。"""
        return self._vertices + self.station

    @property
    def segments(self) -> list[dict[str, Array]]:
        return [{"start": item["start"] + self.station, "end": item["end"] + self.station} for item in self._segments]

    @property
    def arcs(self) -> list[dict[str, Any]]:
        return [{**item, "center": item["center"] + self.station} for item in self._arcs]

    @property
    def is_empty(self) -> bool:
        return not self._arcs and not self._segments and not len(self._vertices)

    def _contains_local(self, points: Array, tol: float) -> Array:
        p = np.atleast_2d(points)
        valid = np.all(p @ self.A.T <= self.local_b + tol, axis=1)
        for circle in self.circles:
            valid &= np.linalg.norm(p - circle["center"], axis=1) <= circle["radius"] + tol
        return valid

    def contains(self, points: ArrayLike, tol: float | None = None) -> bool | Array:
        """判断点或点数组是否属于连续物理可行域。"""
        value = np.asarray(points, dtype=float)
        if value.shape != (2,) and (value.ndim != 2 or value.shape[1] != 2):
            raise ValueError("待判断坐标应为 (2,) 或 (n, 2) 数组。")
        if not np.isfinite(value).all():
            raise ValueError("待判断坐标必须为有限数值。")
        result = self._contains_local(value - self.station, self.tolerance if tol is None else tol)
        return bool(result[0]) if value.ndim == 1 else result

    def _ensure_nonempty(self) -> None:
        if self.is_empty:
            raise ValueError("第一次物理可行域为空；请核查检测信息与物理范围。")

    @property
    def representative_point(self) -> Array:
        """若干边界点的均值；由凸性保证属于物理可行域。"""
        self._ensure_nonempty()
        points = list(self._vertices)
        for arc in self._arcs:
            angle = (arc["start_rad"] + arc["end_rad"]) / 2
            points.append(arc["center"] + arc["radius"] * np.array([math.cos(angle), math.sin(angle)]))
        return np.mean(points, axis=0) + self.station

    @property
    def area(self) -> float:
        """用格林公式逐段积分得到连续区域面积，单位平方米。"""
        total = 0.0
        for line in self._segments:
            p, q = line["start"], line["end"]
            total += p[0] * q[1] - p[1] * q[0]
        for arc in self._arcs:
            cx, cy = arc["center"]
            radius, start, end = arc["radius"], arc["start_rad"], arc["end_rad"]
            total += radius * cx * (math.sin(end) - math.sin(start))
            total += radius * cy * (math.cos(start) - math.cos(end))
            total += radius**2 * (end - start)
        if total < -100 * self.tolerance * max(self.reception_upper, self.target_radius):
            raise RuntimeError("物理区域边界方向异常，面积积分得到负值。")
        return max(0.0, total / 2)

    def _distance_candidates(self, point: ArrayLike, farthest: bool) -> tuple[Array, Array]:
        self._ensure_nonempty()
        local = _point(point) - self.station
        candidates = list(self._vertices)
        for line in self._segments:
            p, q = line["start"], line["end"]
            candidates.extend([p, q])
            if not farthest:
                edge = q - p
                fraction = np.clip(np.dot(local - p, edge) / np.dot(edge, edge), 0.0, 1.0)
                candidates.append(p + fraction * edge)
        for arc in self._arcs:
            center, radius = arc["center"], arc["radius"]
            for angle in (arc["start_rad"], arc["end_rad"]):
                candidates.append(center + radius * np.array([math.cos(angle), math.sin(angle)]))
            delta = (center - local) if farthest else (local - center)
            if np.linalg.norm(delta) > self.tolerance:
                angle = math.atan2(delta[1], delta[0])
                if _on_arc(angle, arc["start_rad"], arc["end_rad"]):
                    candidates.append(center + radius * delta / np.linalg.norm(delta))
        return local, np.asarray(candidates)

    def farthest_point(self, point: ArrayLike) -> Array:
        """连续区域中到查询点最远的一点；包含圆弧内部的径向驻点。"""
        local, candidates = self._distance_candidates(point, True)
        return candidates[np.argmax(np.linalg.norm(candidates - local, axis=1))] + self.station

    def max_distance(self, point: ArrayLike) -> float:
        return float(np.linalg.norm(self.farthest_point(point) - _point(point)))

    farthest_distance = max_distance

    def nearest_point(self, point: ArrayLike) -> Array:
        """区域内点的最近距离为零；区域外点检查线段投影及圆弧驻点。"""
        self._ensure_nonempty()
        if self.contains(point, tol=0.0):
            return _point(point).copy()
        local, candidates = self._distance_candidates(point, False)
        return candidates[np.argmin(np.linalg.norm(candidates - local, axis=1))] + self.station

    def min_distance(self, point: ArrayLike) -> float:
        return float(np.linalg.norm(self.nearest_point(point) - _point(point)))

    nearest_distance = min_distance

    def angular_interval(self, point: ArrayLike) -> dict[str, Any]:
        """返回域外点观察整个凸区域的连续方向区间及两端支撑点。

角度单位为弧度，围绕代表点方向展开，不强制落在 [0, 2π)。线段
只需端点；圆弧还检查从查询点到所在圆的切点。近距五米限制由调用者
另行检查，本接口只要求查询点在区域外。
        """
        self._ensure_nonempty()
        query = _point(point)
        if self.min_distance(query) <= self.tolerance:
            raise ValueError("连续可见方向区间要求查询点严格位于物理可行域外。")
        local = query - self.station
        candidates = list(self._vertices)
        for arc in self._arcs:
            center, radius = arc["center"], arc["radius"]
            for angle in (arc["start_rad"], arc["end_rad"]):
                candidates.append(center + radius * np.array([math.cos(angle), math.sin(angle)]))
            offset = local - center
            distance = float(np.linalg.norm(offset))
            if distance >= radius:
                direction = math.atan2(offset[1], offset[0])
                deviation = math.acos(np.clip(radius / distance, -1.0, 1.0))
                for angle in (direction - deviation, direction + deviation):
                    if _on_arc(angle, arc["start_rad"], arc["end_rad"]):
                        candidates.append(center + radius * np.array([math.cos(angle), math.sin(angle)]))
        vectors = np.asarray(candidates) - local
        raw = np.arctan2(vectors[:, 1], vectors[:, 0])
        reference = self.representative_point - query
        pivot = math.atan2(reference[1], reference[0])
        angles = pivot + np.arctan2(np.sin(raw - pivot), np.cos(raw - pivot))
        low, high = int(np.argmin(angles)), int(np.argmax(angles))
        if angles[high] - angles[low] >= math.pi + 1e-10:
            raise RuntimeError("域外凸区域的可见方向展开异常。")
        return {"lower_rad": float(angles[low]), "upper_rad": float(angles[high]),
                "lower_point": candidates[low] + self.station, "upper_point": candidates[high] + self.station}

    def ray_interval(self, point: ArrayLike, direction: ArrayLike) -> tuple[float, float] | None:
        """解析裁剪射线 Q+t*u，返回 t≥0 区间；u 自动单位化，t 单位为米。"""
        origin = _point(point) - self.station
        unit = _point(direction, "射线方向")
        norm = np.linalg.norm(unit)
        if norm == 0:
            raise ValueError("射线方向不能为零向量。")
        return _clip_line(origin, unit / norm, self.A, self.local_b, self.circles, self.tolerance, 0.0, math.inf)

    def ray_intersection_point(self, point: ArrayLike, direction: ArrayLike) -> Array:
        interval = self.ray_interval(point, direction)
        if interval is None:
            raise ValueError("给定射线与物理可行域不相交。")
        unit = _point(direction).copy()
        unit /= np.linalg.norm(unit)
        return _point(point) + np.mean(interval) * unit

    def to_dict(self) -> dict[str, Any]:
        """保留精确圆弧参数；第三问无需从图片反推几何。"""
        return {"type": "convex_disk_halfplane_intersection", "说明": "连续圆弧与线段表示；保留首次五米内区域作为保守扩张。",
                "station": self.station.tolist(), "bearing_deg": self.bearing_deg,
                "half_width_deg": self.half_width_deg, "target_center": self.target_center.tolist(),
                "target_radius": self.target_radius, "reception_upper": self.reception_upper,
                "halfplanes": {"A": self.A.tolist(), "b": (self.local_b + self.A @ self.station).tolist()},
                "segments": [{k: v.tolist() for k, v in line.items()} for line in self.segments],
                "arcs": [{**arc, "center": arc["center"].tolist()} for arc in self.arcs],
                "vertices": self.vertices.tolist(), "area_m2": self.area, "empty": self.is_empty,
                "tolerance_m": self.tolerance}


def _clip_line(origin: Array, unit: Array, A: Array, b: Array, circles: list[dict[str, Any]],
               tol: float, lower: float = -math.inf, upper: float = math.inf) -> tuple[float, float] | None:
    """用线性和二次不等式裁剪直线参数，浮点切触保留为零长区间。"""
    for normal, rhs in zip(A, b):
        coefficient, remaining = float(normal @ unit), float(rhs - normal @ origin)
        if abs(coefficient) <= 64 * np.finfo(float).eps:
            if remaining < -tol:
                return None
        elif coefficient > 0:
            upper = min(upper, remaining / coefficient)
        else:
            lower = max(lower, remaining / coefficient)
    for circle in circles:
        offset = origin - circle["center"]
        along = float(offset @ unit)
        # 垂距用二维叉积，避免大 along² 与 |offset|² 相减。
        perpendicular = float(offset[0] * unit[1] - offset[1] * unit[0])
        square = circle["radius"]**2 - perpendicular**2
        if square < -tol * max(1.0, 2 * circle["radius"]):
            return None
        width = math.sqrt(max(0.0, square))
        lower, upper = max(lower, -along - width), min(upper, -along + width)
    if lower > upper + tol:
        return None
    if lower > upper:
        lower = upper = (lower + upper) / 2
    return float(lower), float(upper)


def build_physical_region(station: ArrayLike, bearing_deg: float, half_width_deg: float = 1.0,
                          target_center: ArrayLike = (0.0, 0.0), target_radius: float = 1800.0,
                          reception_upper: float = 1500.0, *, tolerance: float = 1e-8) -> PhysicalRegion:
    """构造 W(S1,θ1,δ)∩B(S1,1500)∩B(O,1800)，O 正式值为 (0,0)。"""
    first, center = _point(station, "第一检测点"), _point(target_center, "目标区域中心")
    for name, number in (("目标区域半径", target_radius), ("最大接收半径", reception_upper), ("数值容差", tolerance)):
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name}必须为正的有限数值。")
    A, b = bearing_halfplanes([[0.0, 0.0]], [bearing_deg], half_width_deg)
    circles = [{"center": np.zeros(2), "radius": float(reception_upper)},
               {"center": center - first, "radius": float(target_radius)}]
    if np.linalg.norm(circles[0]["center"] - circles[1]["center"]) <= tolerance and abs(target_radius - reception_upper) <= tolerance:
        circles = circles[:1]
    region = PhysicalRegion(first.copy(), float(bearing_deg) % 360, float(half_width_deg), center.copy(),
                            float(target_radius), float(reception_upper), A, b, circles, [], [], np.empty((0, 2)), tolerance)
    crossings: list[Array] = [np.zeros(2)]
    for normal, rhs in zip(A, b):
        origin = normal * rhs
        unit = np.array([-normal[1], normal[0]])
        interval = _clip_line(origin, unit, A, b, circles, tolerance)
        if interval is not None:
            start, end = origin + interval[0] * unit, origin + interval[1] * unit
            crossings.extend([start, end])
            if np.linalg.norm(end - start) > tolerance:
                region._segments.append({"start": start, "end": end})
        for circle in circles:
            distance = float(normal @ circle["center"] - rhs)
            square = circle["radius"]**2 - distance**2
            if square >= -tolerance * 2 * circle["radius"]:
                foot = circle["center"] - distance * normal
                delta = math.sqrt(max(0.0, square)) * unit
                crossings.extend([foot + delta, foot - delta])
    if len(circles) == 2:
        c0, c1 = circles
        vector = c1["center"] - c0["center"]
        distance = float(np.linalg.norm(vector))
        r0, r1 = c0["radius"], c1["radius"]
        if distance > tolerance and abs(r0 - r1) - tolerance <= distance <= r0 + r1 + tolerance:
            along = (r0**2 - r1**2 + distance**2) / (2 * distance)
            square = r0**2 - along**2
            if square >= -tolerance * 2 * r0:
                unit = vector / distance
                foot = c0["center"] + along * unit
                delta = math.sqrt(max(0.0, square)) * np.array([-unit[1], unit[0]])
                crossings.extend([foot + delta, foot - delta])
    all_crossings = _unique(crossings, tolerance)
    region._vertices = all_crossings[region._contains_local(all_crossings, tolerance)]
    for circle in circles:
        relative = all_crossings - circle["center"]
        on_circle = np.abs(np.linalg.norm(relative, axis=1) - circle["radius"]) <= 10 * tolerance
        raw = np.mod(np.arctan2(relative[on_circle, 1], relative[on_circle, 0]), TAU)
        cuts = sorted(set([0.0, TAU, *raw.tolist()]))
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end - start <= 1e-14:
                continue
            angle = (start + end) / 2
            probe = circle["center"] + circle["radius"] * np.array([math.cos(angle), math.sin(angle)])
            if region._contains_local(probe, tolerance)[0]:
                arc = {"center": circle["center"].copy(), "radius": circle["radius"], "start_rad": start, "end_rad": end}
                region._arcs.append(arc)
                for theta in (start, end):
                    crossings.append(circle["center"] + circle["radius"] * np.array([math.cos(theta), math.sin(theta)]))
    candidates = _unique(crossings, tolerance)
    region._vertices = candidates[region._contains_local(candidates, tolerance)]
    return region


def sample_region(region: PhysicalRegion, boundary_count: int = 128, radial_levels: int = 4) -> Array:
    """采样所有边界连接点、按长度加密边界及代表点向边界的内部线段。

boundary_count 为边界分配预算而非严格点数；连接点强制保留，返回数
量需按实际数组记录。radial_levels=0 时仅返回边界。
    """
    region._ensure_nonempty()
    if boundary_count < 4 or radial_levels < 0:
        raise ValueError("边界采样预算至少为 4，内部层数应为非负整数。")
    points = list(region.vertices)
    perimeter = sum(float(np.linalg.norm(line["end"] - line["start"])) for line in region._segments)
    perimeter += sum(arc["radius"] * (arc["end_rad"] - arc["start_rad"]) for arc in region._arcs)
    for line in region.segments:
        count = max(2, int(math.ceil(boundary_count * np.linalg.norm(line["end"] - line["start"]) / max(perimeter, 1e-30))) + 1)
        points.extend(np.linspace(line["start"], line["end"], count))
    for arc in region.arcs:
        count = max(2, int(math.ceil(boundary_count * arc["radius"] * (arc["end_rad"] - arc["start_rad"]) / max(perimeter, 1e-30))) + 1)
        angles = np.linspace(arc["start_rad"], arc["end_rad"], count)
        points.extend(arc["center"] + arc["radius"] * np.column_stack([np.cos(angles), np.sin(angles)]))
    boundary = _unique(points, region.tolerance)
    if radial_levels == 0:
        return boundary
    representative = region.representative_point
    blocks = [boundary, representative[None, :]]
    for fraction in np.arange(1, radial_levels + 1) / (radial_levels + 1):
        blocks.append(representative + fraction * (boundary - representative))
    return _unique(np.vstack(blocks), region.tolerance)


def centerline_interval(region: PhysicalRegion) -> tuple[float, float] | None:
    """首次中心示向射线被两个物理圆裁剪后的距离区间。"""
    angle = math.radians(region.bearing_deg)
    return region.ray_interval(region.station, [math.cos(angle), math.sin(angle)])


def analytic_candidates(region: PhysicalRegion, radius: float = 1000.0) -> Array:
    """中心线近似透镜的最大侧向交点；不声称其满足完整角域可靠约束。"""
    interval = centerline_interval(region)
    if interval is None:
        raise ValueError("首次中心示向线未穿过物理可行域，中心线近似候选点不适用。")
    lower, upper = interval
    middle, half = (lower + upper) / 2, (upper - lower) / 2
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("可靠接收半径必须为正的有限数值。")
    if half > radius:
        return np.empty((0, 2))
    angle = math.radians(region.bearing_deg)
    unit = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-unit[1], unit[0]])
    height = math.sqrt(max(0.0, radius**2 - half**2))
    center = region.station + middle * unit
    return np.asarray([center + height * normal, center - height * normal])


def reliable_membership(region: PhysicalRegion, points: ArrayLike, radius: float = 1000.0,
                        safety_margin: float = 1e-6, require_bearing: bool = False,
                        near_radius: float = 5.0) -> bool | Array:
    """连续最远距离检验 K；require_bearing 再剔除距 Omega 不超过五米者。

安全裕度向内收缩接收条件、向外扩张近距排除条件。K 自身只保证接收，
不能保证得到示向度。该计算使用浮点解析边界，不是区间算术认证。
    """
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("接收半径须为正数，安全裕度须为非负有限数。")
    if require_bearing and safety_margin <= 0:
        raise ValueError("保证获得示向度时必须提供正安全裕度，使最近距离严格超过近距阈值。")
    value = np.asarray(points, dtype=float)
    if value.shape != (2,) and (value.ndim != 2 or value.shape[1] != 2):
        raise ValueError("候选检测点应为 (2,) 或 (n, 2) 数组。")
    result = np.array([region.max_distance(point) <= radius - safety_margin and
                       (not require_bearing or region.min_distance(point) >= near_radius + safety_margin)
                       for point in np.atleast_2d(value)], dtype=bool)
    return bool(result[0]) if value.ndim == 1 else result


def reliable_bbox(region: PhysicalRegion, radius: float = 1000.0) -> Array:
    """由可行域已知边界点的接收圆包围盒求交，给出 K 的保守外包矩形。"""
    region._ensure_nonempty()
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("接收半径必须为正的有限数值。")
    points = region.vertices
    if not len(points):
        points = region.representative_point[None, :]
    return np.vstack([np.max(points - radius, axis=0), np.min(points + radius, axis=0)])


def reliable_boundary(region: PhysicalRegion, radius: float = 1000.0, count: int = 256,
                      safety_margin: float = 1e-6) -> Array:
    """从解析安全种子径向二分绘制 K 的内缩边界；连线是绘图近似。

完整窄扇形由半径 L/(2 cosδ) 的圆覆盖，因此在该半径小于可靠半径时
存在显式 K 内点。当前题设及小幅敏感性实验满足此条件；更宽扇形或
更小接收半径不静默猜测 K 是否为空，而明确报告需另求安全种子。
    """
    if count < 8:
        raise ValueError("可靠域边界采样数至少为 8。")
    delta = math.radians(region.half_width_deg)
    cover_radius = region.reception_upper / (2 * math.cos(delta))
    angle = math.radians(region.bearing_deg)
    center = region.station + cover_radius * np.array([math.cos(angle), math.sin(angle)])
    if delta > math.pi / 4 or not reliable_membership(region, center, radius, safety_margin):
        raise ValueError("解析安全种子不适用；不能据此断定可靠域为空，请另求覆盖圆中心。")
    result = []
    for direction in np.linspace(0.0, TAU, count, endpoint=False):
        unit = np.array([math.cos(direction), math.sin(direction)])
        lower, upper = 0.0, 2 * radius + region.reception_upper
        for _ in range(45):
            middle = (lower + upper) / 2
            if reliable_membership(region, center + middle * unit, radius, safety_margin):
                lower = middle
            else:
                upper = middle
        result.append(center + lower * unit)
    return np.asarray(result)
