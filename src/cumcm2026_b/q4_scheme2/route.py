"""第四问方案二：950米三角网格的19站开放最短路及下界证书。"""
import math
import numpy as np

from ..q4_strategy import search_stations


def station_pool():
    """先保留原13站编号，再按逆时针加入半径1900米上的6站。"""
    a, h = 950., 950. * math.sqrt(3) / 2
    outer = np.array([(2*a, 0), (a, 2*h), (-a, 2*h),
                      (-2*a, 0), (-a, -2*h), (a, -2*h)])
    return np.vstack((search_stations(), outer))


def shortest_route():
    """在最短边图中精确搜索哈密顿路径；找到即达到全局距离下界。

    不是把最近邻贪心结果宣称最优：下界为18倍最小站距，逐边验证
    返回路径达到下界。若构造发生改变、无法找到该证书，则明确报错。
    """
    points = station_pool()
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    minimum = float(distances[np.triu_indices(len(points), 1)].min())
    adjacent = [list(map(int, np.flatnonzero(np.isclose(row, minimum, atol=1e-7, rtol=0))))
                for row in distances]

    def visit(path, used):
        if len(path) == len(points):
            return path
        # 仅决定搜索顺序；遇到死路会回溯，不是贪心路线优化。
        options = sorted((j for j in adjacent[path[-1]] if j not in used),
                         key=lambda j: (sum(k not in used for k in adjacent[j]), j))
        for j in options:
            result = visit(path + [j], used | {j})
            if result is not None:
                return result
        return None

    order = visit([0], {0})
    if order is None:
        raise RuntimeError('当前站点构造无法取得最短边下界证书，不能宣称路线全局最短')
    layout = points[order]
    edges = np.linalg.norm(np.diff(layout, axis=0), axis=1)
    lower_bound = (len(points) - 1) * minimum
    length = float(edges.sum())
    if not math.isclose(length, lower_bound, abs_tol=1e-6, rel_tol=0):
        raise RuntimeError('路线未达到全局下界')
    certificate = {
        '站数': len(points), '原站数': 13, '新增站数': 6,
        '源区域半径_米': 1800., '新增站半径_米': 1900.,
        '约束': '圆心出发、终点自由、不要求返回圆心；仅固定站间欧氏路程',
        '站点池访问索引_从零开始': order,
        '最小两站距离_米': minimum, '所需最少路段数': len(points) - 1,
        '路线逐段长度_米': edges.tolist(), '全局下界_米': lower_bound,
        '达到的总长度_米': length, '达到全局下界': True,
        '证明': '访问19个不同站至少连接18段，每段至少950米；本路径18段均为950米，故全局最短。',
        '访问表': [{'访问序号': i + 1, '原站点池编号': j + 1,
                    '类型': '原有站' if j < 13 else '新增外围站',
                    '位置': points[j].tolist()} for i, j in enumerate(order)]}
    return layout, certificate
