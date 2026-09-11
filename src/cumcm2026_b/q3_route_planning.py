"""保持未来固定站顺序，向路线插入已发现目标，再执行一个动作并重规划。

已具证书目标使用安全清除区域选点；未精确定位目标的圆心仅作调度预测，
不能用于无证书清除。贪心插入及局部反转均为启发式，不宣称全局最优。
"""
import math
import numpy as np

from .q3_geometry import minimum_circle, max_distance


def _safe_on_segment(vertices, center, anchor):
    """从anchor到已验证圆心的线段上寻找较近安全点。"""
    anchor = np.asarray(anchor, dtype=float)
    if max_distance(vertices, anchor) <= 19.5:
        return anchor
    lo, hi = 0., 1.
    for _ in range(32):
        mid = (lo + hi) / 2
        if max_distance(vertices, anchor + mid * (center - anchor)) <= 19.5:
            hi = mid
        else:
            lo = mid
    return anchor + hi * (center - anchor)


def _straight_safe_point(vertices, a, b):
    """求直线段与所有19.5米圆盘交集，无交集则返回None。"""
    direction = b - a
    square = float(direction @ direction)
    if square < 1e-12:
        return a.copy() if max_distance(vertices, a) <= 19.5 else None
    offset = a - vertices
    linear = 2 * (offset @ direction)
    constant = (offset * offset).sum(axis=1) - 19.5**2
    discriminant = linear * linear - 4 * square * constant
    if np.any(discriminant < 0):
        return None
    low = max(0., float(np.max((-linear - np.sqrt(discriminant)) / (2 * square))))
    high = min(1., float(np.min((-linear + np.sqrt(discriminant)) / (2 * square))))
    if low > high:
        return None
    point = a + ((low + high) / 2) * direction
    return point if max_distance(vertices, point) <= 19.5 + 1e-7 else None


def task_point(task, a, b=None):
    center = task["center"]
    if not task["ready"]:
        return center.copy()
    candidates = [center, _safe_on_segment(task["vertices"], center, a)]
    if b is not None:
        candidates.append(_safe_on_segment(task["vertices"], center, b))
        straight = _straight_safe_point(task["vertices"], a, b)
        if straight is not None:
            candidates.append(straight)
    valid = [p for p in candidates if max_distance(task["vertices"], p) <= 19.5 + 1e-7]
    if not valid:
        raise RuntimeError("路线清除候选未通过覆盖证书")
    return min(valid, key=lambda p: np.linalg.norm(p-a) + (np.linalg.norm(p-b) if b is not None else 0)).copy()


def route_length(nodes):
    return sum(math.dist(a["位置"], b["位置"]) for a, b in zip(nodes, nodes[1:]))


def plan_route(states, current, fixed_stations):
    """把所有候选目标插入剩余固定站路线，包括最后一个固定站之后的尾段。"""
    tasks = {}
    for state in states:
        center, radius = minimum_circle(state.region.vertices)
        tasks[state.channel] = {"center": center, "radius": radius, "ready": radius <= 19.5,
                                "vertices": state.region.vertices}
    nodes = [{"类型": "当前位置", "位置": list(map(float, current))}]
    nodes += [{"类型": "固定站", "位置": list(map(float, p)), "剩余站序": i} for i, p in enumerate(fixed_stations)]
    pending = set(tasks)
    while pending:
        best = None
        for channel in sorted(pending):
            task = tasks[channel]
            for index in range(1, len(nodes) + 1):
                a = np.asarray(nodes[index-1]["位置"])
                b = np.asarray(nodes[index]["位置"]) if index < len(nodes) else None
                p = task_point(task, a, b)
                delta = float(np.linalg.norm(p-a) + (np.linalg.norm(p-b)-np.linalg.norm(a-b) if b is not None else 0))
                key = (delta, channel, index)
                if best is None or key < best[0]:
                    best = (key, index, {"类型": "目标", "频道": channel, "位置": p.tolist(),
                                         "可安全清除": task["ready"], "外包覆盖圆半径_米": task["radius"]})
        _, index, node = best
        nodes.insert(index, node)
        pending.remove(node["频道"])
    # 仅反转固定站之间的连续目标段，禁止改变固定站先后次序。
    for _ in range(12):
        changed = False
        base = route_length(nodes)
        for i in range(1, len(nodes)-1):
            if nodes[i]["类型"] != "目标":
                continue
            for j in range(i+1, len(nodes)):
                if nodes[j]["类型"] != "目标":
                    break
                candidate = nodes[:i] + nodes[i:j+1][::-1] + nodes[j+1:]
                if route_length(candidate) < base - 1e-6:
                    nodes = candidate
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return {"计划路径": nodes, "预计几何路程_米": route_length(nodes),
            "性质": "保持固定站次序的贪心插入与局部反转；未定位目标位置仅为调度预测"}
