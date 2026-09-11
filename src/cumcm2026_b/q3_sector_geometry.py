"""闭扇形责任区域、逐频道保守覆盖证书和仅供调度的盲区近似。"""
from functools import lru_cache
import math
import numpy as np

from .q3_geometry import circle_polygon, clip


def intersect_sector(vertices, sector):
    if isinstance(sector, bool) or sector not in range(6):
        raise ValueError('扇形编号必须为0至5')
    start, end = sector*math.pi/3, (sector+1)*math.pi/3
    result = clip(np.asarray(vertices), [math.sin(start), -math.cos(start)], 0.)
    return clip(result, [-math.sin(end), math.cos(end)], 0.)


def sector_polygon(sector):
    # 外切圆多边形包含真实1800米圆；覆盖它可充分证明真实扇形覆盖。
    return intersect_sector(circle_polygon(), sector)


def sector_midpoint(sector):
    angle = (sector+.5)*math.pi/3
    return 1200*np.array([math.cos(angle), math.sin(angle)])


def triangles(vertices):
    p = np.asarray(vertices, dtype=float)
    if not len(p): return np.empty((0, 3, 2))
    if len(p) == 1: return np.repeat(p, 3, axis=0)[None, :, :]
    if len(p) == 2: return np.array([[p[0], p[1], (p[0]+p[1])/2]])
    center = p.mean(axis=0)
    return np.stack([np.repeat(center[None, :], len(p), axis=0), p, np.roll(p, -1, axis=0)], axis=1)


def cover_polygon(vertices, sites, radius=999.5, cell_m=25., max_nodes=4096):
    """一个三角形仅在其全部顶点被同一圆覆盖时通过；预算不足返回未证实。"""
    points = np.asarray(sites, dtype=float).reshape(-1, 2)
    pending = triangles(vertices)
    holes, checked = [], 0
    while len(pending):
        if checked+len(pending) > max_nodes:
            holes.extend(pending)
            break
        checked += len(pending)
        if not len(points):
            holes.extend(pending)
            break
        distances = np.linalg.norm(pending[:, None, :, :]-points[None, :, None, :], axis=-1)
        covered = distances.max(axis=2).min(axis=1) <= radius
        pending = pending[~covered]
        if not len(pending): break
        center = pending.mean(axis=1)
        extent = np.linalg.norm(pending-center[:, None, :], axis=2).max(axis=1)
        outside = (np.linalg.norm(center[:, None, :]-points[None, :, :], axis=2)-extent[:, None]).min(axis=1) > radius
        lengths = np.linalg.norm(pending-np.roll(pending, -1, axis=1), axis=2)
        stop = outside | (lengths.max(axis=1) <= cell_m)
        holes.extend(pending[stop])
        pending, lengths = pending[~stop], lengths[~stop]
        if not len(pending): break
        edge = lengths.argmax(axis=1)
        rows = np.arange(len(pending))
        a, b, c = pending[rows, edge], pending[rows, (edge+1)%3], pending[rows, (edge+2)%3]
        middle = (a+b)/2
        pending = np.concatenate([np.stack([a, middle, c], axis=1), np.stack([middle, b, c], axis=1)])
    return {'covered': not holes, 'checked': checked, 'holes': np.asarray(holes).reshape(-1, 3, 2)}


@lru_cache(maxsize=512)
def cached_sector_cover(sector, sites, radius, cell_m, max_nodes):
    return cover_polygon(sector_polygon(sector), sites, radius, cell_m, max_nodes)


def cover_gain(holes, point, radius=999.5):
    """剩余三角形质心覆盖比例，仅作候选评分，不用作完成证书。"""
    if not len(holes): return 0.
    a, b = holes[:, 1]-holes[:, 0], holes[:, 2]-holes[:, 0]
    weights = np.maximum(np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]), 1e-8)
    selected = np.linalg.norm(holes.mean(axis=1)-point, axis=1) <= radius
    return float(weights[selected].sum()/weights.sum())


def covers_holes(holes, point, radius=999.5):
    return not len(holes) or float(np.linalg.norm(holes-point, axis=2).max()) <= radius
