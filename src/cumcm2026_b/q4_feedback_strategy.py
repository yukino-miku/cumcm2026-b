"""第四问改进：小范围覆盖顺路处理、方向反馈评分和完整任务路线预测。

位置/朝向采样只用于调度，绝不替换正观测连续外包或清除证书。
"""
from dataclasses import dataclass
import itertools
import math
import time

import numpy as np

from .q4_spacing_strategy import SpacingConfig, SpacedFour
from .q3_geometry import (DELTA, minimum_circle, max_distance, min_distance,
                          principal_axes, sweep_path, finish_bound, angular_interval, add_wedge)
from .q3_route_planning import plan_route

TAU = 2 * math.pi


@dataclass
class FeedbackConfig(SpacingConfig):
    probe_spacing_m: float = 100.
    small_sweep_limit: int = 4
    direction_feedback: bool = True
    task_routing: bool = True
    position_samples: int = 41

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.small_sweep_limit, bool) or not isinstance(self.small_sweep_limit, int)
                or not 0 <= self.small_sweep_limit <= 6):
            raise ValueError('small_sweep_limit须为0至6的整数；0关闭顺路小范围覆盖')
        if not isinstance(self.direction_feedback, bool) or not isinstance(self.task_routing, bool):
            raise ValueError('方向反馈与任务路线开关必须为布尔值')
        if isinstance(self.position_samples, bool) or not isinstance(self.position_samples, int) or not 9 <= self.position_samples <= 121:
            raise ValueError('position_samples须为9至121的整数')


def half_circle(vector, front=True):
    """朝向角可行半圆，按[0,2π]分段；容差仅用于启发式评分。"""
    if np.linalg.norm(vector) < 1e-9:
        return [(0., TAU)] if front else []
    pivot = math.atan2(vector[1], vector[0]) + (0 if front else math.pi)
    low = (pivot - math.pi / 2) % TAU
    high = low + math.pi
    return [(low, high)] if high <= TAU else [(low, TAU), (0., high - TAU)]


def intersect_arcs(a, b):
    return [(max(x, u), min(y, v)) for x, y in a for u, v in b
            if min(y, v) - max(x, u) > 1e-10]


def orientation_arcs(position, observations):
    """给定假设位置计算朝向弧；有距离歧义的负反馈不被当成背侧。

    已接收站要求真实半径至少为该距离；再与1000米下界一起使用。
    此计算不是连续位置可行域的认证算法。
    """
    positives = [o for o in observations if o['结果'] in ('direction', 'near')]
    if not positives:
        return []
    distances = [math.dist(position, o['位置']) for o in positives]
    if max(distances) > 1500 + 1e-5:
        return []
    lower_radius = max(1000., max(distances))
    arcs = [(0., TAU)]
    for o in positives:
        vector = np.asarray(o['位置']) - position
        if o['结果'] == 'direction' and np.linalg.norm(vector) <= 5:
            return []
        arcs = intersect_arcs(arcs, half_circle(vector))
    for o in observations:
        if o['结果'] == 'no_signal' and math.dist(position, o['位置']) < lower_radius - 1e-6:
            arcs = intersect_arcs(arcs, half_circle(np.asarray(o['位置']) - position, False))
    return arcs


def joint_hypotheses(state, count=41):
    """在正观测外包中选代表位置，对每个位置精确求朝向弧；不读取真值。"""
    vertices = state.region.vertices
    negatives = [o for o in state.region.observations if o['结果'] == 'no_signal'
                 and max_distance(vertices, np.asarray(o['位置'])) <= 999.5]
    # 仅在确认该已知源有背侧负反馈后启用；全向/未知类型先保持原几何方法。
    if not negatives:
        return []
    center = vertices.mean(axis=0)
    points = [center] + [center + t * (v - center)
                         for t in (.2, .4, .6, .8, .99) for v in vertices]
    if len(points) > count:
        points = [points[i] for i in np.linspace(0, len(points) - 1, count).astype(int)]
    result = []
    for p in points:
        if np.linalg.norm(p) > 1800 + 1e-6 or any(np.linalg.norm(p - s[0]) < 1e-7 for s in result):
            continue
        arcs = orientation_arcs(p, state.region.observations)
        if arcs:
            result.append((np.asarray(p), arcs))
    return result


def reception_support(hypotheses, point):
    """位置样本与朝向弧的接收支持度；不是已校准的真实概率。"""
    total = received = 0.
    for p, arcs in hypotheses:
        total += sum(b - a for a, b in arcs)
        if np.linalg.norm(point - p) <= 1000:
            received += sum(b - a for a, b in intersect_arcs(arcs, half_circle(point - p)))
    return received / total if total > 0 else 0.


def ordered_small_sweep(vertices, current, destination, limit):
    """少量覆盖点的完整插入最短排列；格点集合不变，保留联合覆盖保证。"""
    if limit == 0 or minimum_circle(vertices)[1] <= 19.5:
        return None
    center, axes = principal_axes(vertices)
    sizes = np.ptp((vertices - center) @ axes, axis=0)
    count = int(np.prod(np.maximum(1, np.ceil(sizes / 20).astype(int))))
    if count > limit:
        return None
    grid = sweep_path(vertices, current)
    best = None
    for order in itertools.permutations(range(len(grid))):
        path = grid[list(order)]
        length = math.dist(current, path[0]) + float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if destination is not None:
            length += math.dist(path[-1], destination) - math.dist(current, destination)
        key = (length, order)
        if best is None or key < best[0]:
            best = (key, path)
    detour = best[0][0]
    return {'位置序列': best[1].tolist(), '覆盖点数': len(grid), '完整绕行_米': detour,
            '增加费用上界_秒': detour / 5 + 3 * (len(grid) - 1) + 5,
            '外包顶点': vertices.tolist(), '性质': '完整20米方格覆盖，费用不含可选顺路扫描'}


def task_projection(current, probe, target, remaining):
    """按代表位置预测一次补测及清除的完整插入；不作为接收或清除证书。"""
    def extra(a, b):
        length = math.dist(a, probe) + math.dist(probe, target)
        return length + math.dist(target, b) - math.dist(a, b) if b is not None else length
    now = extra(current, remaining[0]) if len(remaining) else extra(current, None)
    future = [extra(a, remaining[i + 1] if i + 1 < len(remaining) else None)
              for i, a in enumerate(remaining)]
    return {'预测完整任务绕行_米': now, '预测延后最小绕行_米': min(future) if future else None,
            '预测目标代表位置': np.asarray(target).tolist(), '性质': '单次补测后清除的几何预测；非真实位置或最坏费用保证'}


class FeedbackFour(SpacedFour):
    def __init__(self, client, config=None):
        super().__init__(client, config or FeedbackConfig())
        self._remaining = np.empty((0, 2))

    def _observation_plan(self, state, current):
        if (not self.config.direction_feedback or state.localization_count >= self.config.max_localization_measurements
                or self.config.planning_seconds == 0):
            return super()._observation_plan(state, current)
        hypotheses = joint_hypotheses(state, self.config.position_samples)
        if not hypotheses:
            return super()._observation_plan(state, current)
        started = time.monotonic()
        vertices = state.region.vertices
        center, axes = principal_axes(vertices)
        positives = [o for o in state.region.observations if o['结果'] == 'direction']
        anchor = np.asarray(positives[-1]['位置'])
        ray = center - anchor
        ray /= max(np.linalg.norm(ray), 1e-9)
        lateral = np.array([-ray[1], ray[0]])
        candidates = [center + axes @ np.array([math.cos(t), math.sin(t)]) * d
                      for t in np.arange(8) * math.pi / 4 for d in (75, 150, 300, 500, 650)]
        # 有效接收站附近可落在旧“整个外包可靠距离域”之外，避免被迫越过近处目标。
        candidates += [anchor + ray * forward + lateral * side
                       for forward in (0, 100, 250, 450) for side in (-300, -150, -75, 75, 150, 300)]
        feasible = []
        for p in candidates:
            if (min_distance(vertices, p) <= 5.5 or min_distance(vertices, p) > 1000
                    or any(np.linalg.norm(p - np.asarray(s)) < .01 for s in state.measured_sites)):
                continue
            if not any(np.linalg.norm(p - q) < .01 for q in feasible):
                feasible.append(p)
        separated = [p for p in feasible if np.linalg.norm(p - current) >= self.config.probe_spacing_m - 1e-7
                     and np.linalg.norm(p - np.asarray(state.measured_sites[-1])) >= self.config.probe_spacing_m - 1e-7]
        relaxed = not separated and bool(feasible)
        feasible = separated or feasible
        masses = np.array([sum(b - a for a, b in arc) for _, arc in hypotheses])
        representative = np.average([p for p, _ in hypotheses], axis=0, weights=masses)
        scored = []
        for p in feasible:
            support = reception_support(hypotheses, p)
            if support <= 1e-8:
                continue
            projection = task_projection(current, p, representative, self._remaining)
            travel = projection['预测完整任务绕行_米'] if self.config.task_routing and len(self._remaining) else math.dist(current, p)
            scored.append((travel / 5 + (1 - support) * finish_bound(vertices, p), -support, p, projection))
        scored.sort(key=lambda x: (x[0], x[1], float(x[2][0]), float(x[2][1])))
        best = None
        evaluated = 0
        for _, negative_support, p, projection in scored[:min(12, self.config.candidate_limit)]:
            if evaluated and time.monotonic() - started >= self.config.planning_seconds:
                break
            support = -negative_support
            low, high = angular_interval(vertices, p)
            predictions = []
            for theta in np.linspace(low - math.radians(DELTA), high + math.radians(DELTA), self.config.angle_samples):
                branch = add_wedge(vertices, p, math.degrees(theta))
                if len(branch):
                    predictions.append(finish_bound(branch, p))
            if not predictions:
                continue
            evaluated += 1
            score = math.dist(current, p) / 5 + 6 + support * max(predictions) + (1 - support) * finish_bound(vertices, p)
            if self.config.task_routing and len(self._remaining):
                score += (math.dist(representative, self._remaining[0]) - math.dist(current, self._remaining[0])) / 5
            if best is None or score < best['预计费用_秒']:
                best = {'动作': '追加测向', '位置': p.tolist(), '预计费用_秒': score,
                        '接收样本支持度': support, '位置假设数': len(hypotheses),
                        '预测接收后费用_秒': max(predictions),
                        '无信号后的保守回退费用_秒': finish_bound(vertices, p),
                        '最远可能目标距离_米': max_distance(vertices, p), **projection}
        if best is None:
            plan = super()._observation_plan(state, current)
            plan['方向反馈回退'] = '采样或候选不足，保留原几何；连续外包未删减'
            return plan
        best.update(接收保证=False, 评分性质='位置样本与朝向弧的启发式混合费用；支持度不是真实概率，外包未缩小',
                    方向反馈已启用=True, 距离保证=best['最远可能目标距离_米'] <= 999.5,
                    建议相邻补测间距_米=self.config.probe_spacing_m, 间距约束已放宽=relaxed,
                    距当前机器人_米=math.dist(current, best['位置']),
                    距该频道上次观测_米=math.dist(state.measured_sites[-1], best['位置']),
                    已评估候选数=evaluated, 规划耗时_秒=time.monotonic() - started)
        return best

    def _small_task(self, known, current, remaining, allowance):
        if not len(remaining) or not self.config.route_planning:
            return None
        choices = []
        for state in known:
            plan = ordered_small_sweep(state.region.vertices, current, remaining[0], min(self.config.small_sweep_limit, allowance))
            if plan is None or plan['完整绕行_米'] > self.config.localization_detour_limit_m:
                continue
            future = [ordered_small_sweep(state.region.vertices, a, remaining[i + 1] if i + 1 < len(remaining) else None,
                                         self.config.small_sweep_limit)['增加费用上界_秒'] for i, a in enumerate(remaining)]
            if plan['增加费用上界_秒'] > min(future) + 1e-6:
                continue
            plan['预计延后最小费用_秒'] = min(future)
            choices.append((plan['增加费用上界_秒'], state.channel, plan))
        return min(choices, key=lambda x: (x[0], x[1])) if choices else None

    def _service_known(self, remaining=()):
        remaining = np.asarray(remaining, dtype=float).reshape(-1, 2)
        self._remaining = remaining
        if not self.config.small_sweep_limit and not self.config.direction_feedback and not self.config.task_routing:
            return super()._service_known(remaining)
        deferred, misses = set(), {}
        starting_actions = self.actions
        while not self.complete():
            self._guard()
            known = [s for s in self.channels.values() if s.status == 'found' and s.channel not in deferred]
            if not known:
                return
            allowance = self.config.route_action_limit - (self.actions - starting_actions)
            if len(remaining) and allowance <= 0:
                self._log({'动作': '继续固定搜索', '原因': '达到本路段服务动作上限'})
                return
            current = np.asarray(self.client.state.position)
            small = self._small_task(known, current, remaining, max(0, allowance))
            if small:
                _, channel, plan = small
                state = self.channels[channel]
                self._log({'动作': '顺路小范围覆盖规划', '频道': channel, **plan,
                           '下一固定站': remaining[0].tolist(), '完整绕行门槛_米': self.config.localization_detour_limit_m})
                for point in plan['位置序列']:
                    self._clear(state, np.asarray(point), '顺路小范围覆盖', certified=False)
                    if state.status == 'cleared':
                        self._opportunistic(np.asarray(point), channel)
                        break
                if state.status != 'cleared':
                    raise RuntimeError('小范围完整覆盖仍未清除，需核对物理与协议')
                continue
            route = plan_route(known, current, remaining if self.config.route_planning else [])
            self._log({'动作': '路线规划', **route})
            node = route['计划路径'][1]
            if node['类型'] == '固定站':
                return
            state = self.channels[node['频道']]
            if node['可安全清除']:
                point = np.asarray(node['位置'])
                self._clear(state, point, '顺路证书清除' if len(remaining) else '末尾证书清除', certified=True)
                self._opportunistic(point, state.channel)
                continue
            plan = self._observation_plan(state, current)
            if plan['动作'] == '追加测向':
                point = np.asarray(plan['位置'])
                detour = (math.dist(current, point) + math.dist(point, remaining[0]) - math.dist(current, remaining[0])) if len(remaining) else 0.
                if len(remaining) and self.config.route_planning:
                    projection = task_projection(current, point, plan.get('预测目标代表位置', minimum_circle(state.region.vertices)[0]), remaining)
                    plan.update(projection)
                    if detour > self.config.localization_detour_limit_m:
                        deferred.add(state.channel)
                        self._log({'动作': '延后目标', '频道': state.channel, '原因': '本次补测偏离下一固定站路线过多',
                                   '本次测向绕行_米': detour, '阈值_米': self.config.localization_detour_limit_m})
                        continue
                    if (self.config.task_routing and projection['预测完整任务绕行_米'] > self.config.localization_detour_limit_m
                            and projection['预测完整任务绕行_米'] > projection['预测延后最小绕行_米'] + 1e-6):
                        deferred.add(state.channel)
                        self._log({'动作': '延后目标', '频道': state.channel, '原因': '完整任务预测延后更省路程', **projection})
                        continue
                self._log({'动作': '规划', '频道': state.channel, '方案': plan, '本次测向绕行_米': detour})
                state.localization_count += 1
                feedback = self._measure(state, point, '追加测向')
                if feedback == 'no_signal':
                    misses[state.channel] = misses.get(state.channel, 0) + 1
                    self._log({'动作': '补测无信号', '频道': state.channel, '位置': point.tolist(),
                               '处理': '保留连续外包；下次按正负方向反馈重评分'})
                    if len(remaining) and misses[state.channel] >= self.config.consecutive_no_signal_limit:
                        deferred.add(state.channel)
                        self._log({'动作': '延后目标', '频道': state.channel, '原因': '本路段连续无信号，继续固定扫描'})
                else:
                    misses[state.channel] = 0
                self._opportunistic(point, state.channel)
            elif len(remaining):
                deferred.add(state.channel)
                self._log({'动作': '延后目标', '频道': state.channel, '原因': '较大有限覆盖留到固定扫描后'})
            else:
                path = sweep_path(state.region.vertices, current)
                self._log({'动作': '有限覆盖', '频道': state.channel, '覆盖点数': len(path), '单元边长_米': 20})
                for point in path:
                    self._clear(state, point, '有限方格覆盖', certified=False)
                    if state.status == 'cleared':
                        self._opportunistic(point, state.channel)
                        break
                if state.status != 'cleared':
                    raise RuntimeError('有限覆盖完成仍未清除，需核对物理与协议')

    def run(self):
        result = super().run()
        result['方案'] = '第四问13站改进：小范围顺路覆盖、方向反馈选点与完整任务预测'
        return result
