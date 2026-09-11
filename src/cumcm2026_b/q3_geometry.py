"""第三问多次测向的保守凸外包、清除证书和有限覆盖路径。

圆约束用外切多边形，所有真实可行点仍包含在外包内；负观测另存，
不把有孔集合假装成凸多边形。所有长度以米计。
"""
from __future__ import annotations

import itertools
import math
import numpy as np

from .q1_geometry import bearing_halfplanes

DELTA = 1.005  # 显式包含两位小数舍入的0.005度余量。
TOL = 1e-7
NORMALS = np.column_stack((np.cos(np.arange(128)*2*np.pi/128), np.sin(np.arange(128)*2*np.pi/128)))


def circle_polygon(center=(0, 0), radius=1800.0):
    angles = (np.arange(128)+.5)*2*np.pi/128
    return np.asarray(center)+radius/math.cos(math.pi/128)*np.column_stack((np.cos(angles), np.sin(angles)))


def clip(vertices, normal, bound):
    """凸多边形裁剪；将半平面放宽TOL以保守处理舍入。"""
    p = np.asarray(vertices, dtype=float)
    if len(p) == 0:
        return p.reshape(0, 2)
    slack = p@normal-(bound+TOL)
    inside = slack <= 0
    if inside.all(): return p.copy()
    if not inside.any(): return np.empty((0, 2))
    result = []
    for i in range(len(p)):
        j = (i+1) % len(p)
        if inside[i]: result.append(p[i])
        if inside[i] != inside[j]:
            result.append(p[i]+slack[i]/(slack[i]-slack[j])*(p[j]-p[i]))
    result = np.asarray(result)
    if len(result) > 1:
        keep = np.linalg.norm(result-np.roll(result, 1, axis=0), axis=1) > 1e-10
        if keep.any(): result = result[keep]
        else: result = result[:1]
    return result.reshape(-1, 2)


def add_wedge(vertices, station, bearing, delta=DELTA):
    A, b = bearing_halfplanes([station], [bearing], delta)
    result = np.asarray(vertices).copy()
    for normal, bound in zip(A, b): result = clip(result, normal, bound)
    return result


def add_circle(vertices, center, radius):
    result = np.asarray(vertices).copy()
    for normal in NORMALS:
        result = clip(result, normal, radius+np.dot(normal, center))
        if not len(result): break
    return result


def minimum_circle(vertices):
    p = np.asarray(vertices)
    if not len(p): raise ValueError("空区域不能计算清除证书")
    candidates = list(p)
    candidates += [(a+b)/2 for a, b in itertools.combinations(p, 2)]
    for a, b, c in itertools.combinations(p, 3):
        matrix = 2*np.vstack((b-a, c-a))
        if abs(np.linalg.det(matrix)) > 1e-9:
            center = a+np.linalg.solve(matrix, [np.dot(b-a, b-a), np.dot(c-a, c-a)])
            candidates.append(center)
    centers = np.asarray(candidates)
    radii = np.linalg.norm(centers[:, None, :]-p[None, :, :], axis=2).max(axis=1)
    best = int(radii.argmin())
    return centers[best].copy(), float(radii[best])


def max_distance(vertices, point):
    return float(np.linalg.norm(np.asarray(vertices)-point, axis=1).max())


def min_distance(vertices, point):
    p = np.asarray(vertices)
    point = np.asarray(point)
    if len(p) == 1: return float(np.linalg.norm(p[0]-point))
    a, b = p, np.roll(p, -1, axis=0)
    edge = b-a
    cross = edge[:, 0]*(point[1]-a[:, 1])-edge[:, 1]*(point[0]-a[:, 0])
    area2 = abs(np.sum(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]))
    if len(p) > 2 and area2 > 1e-9 and (np.all(cross >= -TOL) or np.all(cross <= TOL)):
        return 0.0
    denominator = (edge*edge).sum(axis=1)
    t = np.divide(((point-a)*edge).sum(axis=1), denominator, out=np.zeros(len(p)), where=denominator > 1e-20)
    nearest = a+np.clip(t, 0, 1)[:, None]*edge
    return float(np.linalg.norm(nearest-point, axis=1).min())


def clear_point(vertices, current, radius=19.5):
    center, cover = minimum_circle(vertices)
    if cover > radius: return None
    current = np.asarray(current)
    if max_distance(vertices, current) <= radius: return current.copy()
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo+hi)/2
        if max_distance(vertices, current+mid*(center-current)) <= radius: hi = mid
        else: lo = mid
    result = current+hi*(center-current)
    if max_distance(vertices, result) > radius+1e-8:
        raise RuntimeError("清除位置未通过外包顶点证书")
    return result


def principal_axes(vertices):
    p = np.asarray(vertices)
    center = p.mean(axis=0)
    _, vectors = np.linalg.eigh((p-center).T@(p-center))
    major = vectors[:, -1]
    return center, np.column_stack((major, [-major[1], major[0]]))


def sweep_path(vertices, current, spacing=20.0):
    """覆盖整个旋转包围盒；不会只保留中心落在区域内的单元。"""
    center, axes = principal_axes(vertices)
    local = (np.asarray(vertices)-center)@axes
    low, high = local.min(axis=0), local.max(axis=0)
    count = np.maximum(1, np.ceil((high-low)/spacing).astype(int))
    middle = (low+high)/2
    xs = middle[0]+(np.arange(count[0])-(count[0]-1)/2)*spacing
    ys = middle[1]+(np.arange(count[1])-(count[1]-1)/2)*spacing
    paths = []
    for reverse_y in (False, True):
        for reverse_x in (False, True):
            points = []
            for row, y in enumerate(ys[::-1] if reverse_y else ys):
                line = xs[::-1] if (row % 2 == 1) != reverse_x else xs
                points.extend([[x, y] for x in line])
            paths.append(np.asarray(points)@axes.T+center)
    return min(paths, key=lambda p: float(np.linalg.norm(p[0]-current)))


def finish_bound(vertices, current):
    """明确后续策略的费用：安全圆清除，或有限方格蛇形覆盖的最坏费用。"""
    point, radius = minimum_circle(vertices)
    if radius <= 19.5:
        return float(np.linalg.norm(point-current)/5+5)
    path = sweep_path(vertices, current)
    length = np.linalg.norm(path[0]-current)+np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    return float(length/5+3*(len(path)-1)+5)


def angular_interval(vertices, station):
    if min_distance(vertices, station) <= 5:
        raise ValueError("观测角区间要求检测站位于区域外且排除near")
    v = np.asarray(vertices)-station
    center = v.mean(axis=0)
    pivot = math.atan2(center[1], center[0])
    raw = np.arctan2(v[:, 1], v[:, 0])
    angles = pivot+np.arctan2(np.sin(raw-pivot), np.cos(raw-pivot))
    return float(angles.min()), float(angles.max())


class TargetRegion:
    def __init__(self, delta=DELTA):
        self.delta = delta
        self.vertices = circle_polygon()
        self.observations = []
        self.exclusions = []

    def observe(self, point, result, bearing=None):
        point = np.asarray(point, dtype=float)
        self.observations.append({"位置": point.tolist(), "结果": result, "示向度": bearing})
        if result == "direction":
            updated = add_circle(add_wedge(self.vertices, point, bearing, self.delta), point, 1500)
        elif result == "near":
            updated = add_circle(self.vertices, point, 5)
        elif result == "no_signal":
            self.exclusions.append({"位置": point.tolist(), "半径": 1000.0, "来源": result})
            return
        else: raise ValueError("未知观测类型")
        if len(updated) == 0:
            raise RuntimeError("正向观测使物理外包为空；停止核对模型与反馈")
        self.vertices = updated

    def failed_clear(self, point):
        self.exclusions.append({"位置": list(point), "半径": 20.0, "来源": "清除失败"})

    def maybe_contains(self, point):
        return min_distance(self.vertices, point) <= 1e-6 and all(
            np.linalg.norm(np.asarray(point)-e["位置"]) > e["半径"]-1e-6 for e in self.exclusions)


def search_stations():
    return np.vstack(([0., 0.], 1200*np.column_stack((np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3)))))


def coverage_certificate(sites, max_nodes=3000):
    """七站解析证书优先；其他无信号站用保守四叉树充分证书。"""
    sites = np.asarray(sites, dtype=float).reshape(-1, 2)
    if len(sites) < 4: return None
    if all(np.linalg.norm(sites-s, axis=1).min() <= 1e-6 for s in search_stations()):
        return {"方式": "七站连续覆盖", "最大最近距离_米": 968.9015717043836}
    if len(sites) < 7: return None
    stack = [(0., 0., 1800.)]
    count = 0
    while stack and count < max_nodes:
        x, y, half = stack.pop()
        count += 1
        if max(abs(x)-half, 0)**2+max(abs(y)-half, 0)**2 > 1800**2: continue
        far = np.sqrt((np.abs(sites[:, 0]-x)+half)**2+(np.abs(sites[:, 1]-y)+half)**2)
        if far.min() <= 1000-1e-6: continue
        if half <= 10: return None
        step = half/2
        stack.extend((x+dx*step, y+dy*step, step) for dx in (-1, 1) for dy in (-1, 1))
    return {"方式": "保守四叉树覆盖", "已核查单元": count} if not stack else None
