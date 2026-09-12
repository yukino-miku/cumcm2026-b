"""定向源发现的连续几何证书；只使用实际检测站，不读取源的真值。

950米三角网中与1800米圆盘相交的42个闭三角形覆盖整个搜索域。
每个单元须包含于可靠距离内的实测站凸包。任意180度发射半面
至少含有一个凸组合顶点，因此不可能在这些站全部返回无信号。
"""
from functools import lru_cache
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .q3_geometry import min_distance


@lru_cache(maxsize=1)
def discovery_mesh():
    h = 950 * math.sqrt(3) / 2
    def point(k):
        return np.array([950 * (k[0] + k[1] / 2), h * k[1]])
    cells, indices = [], set()
    # 网格外单元到圆心距离下界 > 1800，因此此枚举没有漏掉相交单元。
    assert 950 * math.sqrt(3) * 6 / 2 - 950 > 1800
    for i in range(-6, 6):
        for j in range(-6, 6):
            for cell in (((i, j), (i + 1, j), (i, j + 1)),
                         ((i + 1, j), (i + 1, j + 1), (i, j + 1))):
                if min_distance(np.array([point(k) for k in cell]), np.zeros(2)) <= 1800 + 1e-7:
                    cells.append(cell)
                    indices.update(cell)
    indices = sorted(indices)
    stations = np.array([point(k) for k in indices])
    triangles = np.array([[indices.index(k) for k in cell] for cell in cells])
    assert len(stations) == 31 and len(triangles) == 42
    stations.flags.writeable = triangles.flags.writeable = False
    return stations, triangles


def coverage_status(negative_sites):
    """逐单元证书。严格凸包内替代与原三顶点两条分支，绝不由采样认证。

    替代站必须距单元的三个顶点均<=999.5米，且凸包严格包含三个顶点
    （1微米裕量）；原网格顶点则要求浮点坐标逐项相等，使用解析闭三角形证明。
    保守拒绝近边界替代站只会增加补扫，不会生成虚假的不存在证书。
    """
    stations, triangles = discovery_mesh()
    sites = np.asarray(negative_sites, dtype=float).reshape(-1, 2)
    covered, witnesses = [], []
    for ids in triangles:
        vertices = stations[ids]
        exact = [next((i for i, p in enumerate(sites) if np.array_equal(p, v)), None) for v in vertices]
        if all(i is not None for i in exact):
            covered.append(True)
            witnesses.append({'类型': '原三角形顶点', '负观测序号': exact})
            continue
        eligible = np.flatnonzero(np.linalg.norm(sites[:, None, :] - vertices, axis=2).max(axis=1) <= 999.5) if len(sites) else []
        witness = None
        if len(eligible) >= 3:
            try:
                hull = ConvexHull(sites[eligible])
                if np.max(vertices @ hull.equations[:, :2].T + hull.equations[:, 2]) < -1e-6:
                    witness = {'类型': '严格凸包替代', '负观测序号': np.asarray(eligible)[hull.vertices].tolist()}
            except QhullError:
                pass
        covered.append(witness is not None)
        witnesses.append(witness)
    return {'已覆盖单元数': sum(covered), '总单元数': len(triangles),
            '未覆盖单元': [i for i, ok in enumerate(covered) if not ok], '单元证据': witnesses}


def open_route(current, points):
    """最近邻起始、严格下降的开放路径2-opt；不要求返回起点，不称全局最优。"""
    points = np.asarray(points, float).reshape(-1, 2)
    remaining, order, here = list(range(len(points))), [], np.asarray(current)
    while remaining:
        k = min(remaining, key=lambda i: (np.linalg.norm(points[i] - here), i))
        order.append(k)
        remaining.remove(k)
        here = points[k]
    def length(seq):
        path = np.vstack((current, points[seq]))
        return np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    score = length(order)
    changed = True
    while changed:
        changed = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                trial = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                cost = length(trial)
                if cost < score - 1e-6:
                    order, score, changed = trial, cost, True
    return points[order]
