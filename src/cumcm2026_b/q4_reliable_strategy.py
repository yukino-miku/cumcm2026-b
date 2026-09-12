"""第四问清除优先版：路线可行候选、近距离顺路补扫、连续发现保障。

旧13站版本留作独立对照；发现保障只在主路线后仍无全清证据时启动。
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from .q4_discovery_strategy import DiscoveryConfig, DiscoveryFour
from .q4_feedback_strategy import joint_hypotheses, reception_support, task_projection
from .q4_coverage import discovery_mesh, coverage_status, open_route
from .q2_physical_geometry import build_physical_region, analytic_candidates, centerline_interval
from .q3_geometry import (DELTA, min_distance, max_distance, minimum_circle, principal_axes,
                          search_stations, finish_bound, angular_interval, add_wedge)


@dataclass
class ReliableConfig(DiscoveryConfig):
    localization_detour_limit_m: float = 800.
    discovery_safeguard: bool = True
    route_feasible_candidates: bool = True
    skip_out_of_range: bool = True
    unknown_scan_spacing_m: float = 100.
    coverage_mesh: str = 'compact'
    unknown_novelty: bool = True

    def __post_init__(self):
        super().__post_init__()
        if any(not isinstance(getattr(self, name), bool) for name in
               ('discovery_safeguard', 'route_feasible_candidates', 'skip_out_of_range', 'unknown_novelty')):
            raise ValueError('清除优先版开关必须为布尔值')
        value = self.unknown_scan_spacing_m
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('unknown_scan_spacing_m必须是非负有限数')
        if self.coverage_mesh not in {'compact', 'lattice'}:
            raise ValueError('coverage_mesh必须为compact或lattice')


def candidate_pool(state, hypotheses):
    vertices = state.region.vertices
    center, axes = principal_axes(vertices)
    positives = [o for o in state.region.observations if o['结果'] == 'direction']
    distances = (75, 150, 300, 500, 650) if hypotheses else (30, 75, 150, 300, 500, 650)
    candidates = [center + axes @ np.array([math.cos(t), math.sin(t)]) * d
                  for t in np.arange(8) * math.pi / 4 for d in distances]
    if hypotheses:
        anchor = np.asarray(positives[-1]['位置'])
        ray = center - anchor
        ray /= max(np.linalg.norm(ray), 1e-9)
        lateral = np.array([-ray[1], ray[0]])
        candidates += [anchor + ray * f + lateral * s for f in (0, 100, 250, 450)
                       for s in (-300, -150, -75, 75, 150, 300)]
    else:
        if positives:
            first = positives[0]
            physical = build_physical_region(first['位置'], first['示向度'], DELTA)
            if centerline_interval(physical) is not None:
                for p in analytic_candidates(physical):
                    candidates.extend([center + .8 * (p - center), center + .9 * (p - center)])
        candidates.extend(search_stations())
    feasible = []
    for p in candidates:
        near, far = min_distance(vertices, p), max_distance(vertices, p)
        if near <= 5.5 or (near > 1000 if hypotheses else far > 999.5):
            continue
        if any(np.linalg.norm(p - np.asarray(q)) < .01 for q in state.measured_sites + feasible):
            continue
        feasible.append(p)
    return feasible


class ReliableFour(DiscoveryFour):
    def __init__(self, client, config=None):
        super().__init__(client, config or ReliableConfig())
        self._cover_cache = {}
        self._finishing = False
        self.discovery_stations = []
        self._sample_visibility = {}
        self._unknown_samples = {}

    def complete(self):
        return super().complete() or all(s.status in {'cleared', 'absent'} for s in self.channels.values())

    def _skip_measure(self, state, point):
        if state.status == 'unknown' and self._all_sources_found():
            self._log({'动作': '跳过检测', '频道': state.channel, '位置': np.asarray(point).tolist(),
                       '原因': '已发现16个不同频道源，达到数量上界，继续处理已知源'})
            return True
        if super()._skip_measure(state, point):
            return True
        if self.config.skip_out_of_range and state.status == 'found':
            lower = min_distance(state.region.vertices, point)
            if lower > 1500 + 1e-5:
                self._log({'动作': '跳过检测', '频道': state.channel, '位置': np.asarray(point).tolist(),
                           '原因': '整个连续外包均超出1500米最大接收半径', '距离下界_米': lower,
                           '外包顶点': state.region.vertices.tolist()})
                return True
        return False

    def _opportunistic(self, point, primary):
        if not self.config.opportunistic or self.complete():
            return
        if self.config.unknown_scan_spacing_m == 600 and not self.config.unknown_novelty:
            return super()._opportunistic(point, primary)
        point = np.asarray(point)
        known, unknown = [], []
        for state in self.channels.values():
            if state.channel == primary or state.status not in {'unknown', 'found'}:
                continue
            gap = min((math.dist(p, point) for p in state.measured_sites), default=math.inf)
            limit = self.config.unknown_scan_spacing_m if state.status == 'unknown' else 600.
            if gap < limit - 1e-6 or self._skip_measure(state, point):
                continue
            if state.status == 'found':
                if max_distance(state.region.vertices, point) <= 999.5:
                    known.append(state)
            else:
                if self.config.unknown_novelty and gap < 600 and not self._has_sample_gain(state, point):
                    self._log({'动作': '暂缓近站未知扫描', '频道': state.channel, '位置': point.tolist(),
                               '原因': '离散位置朝向样本没有新增接收机会；不构成不存在证书，仍保留末尾发现保障'})
                    continue
                unknown.append(state)
        known.sort(key=lambda s: (len(s.measured_sites), s.channel))
        unknown.sort(key=lambda s: (s.channel != self.client.state.channel, len(s.measured_sites), s.channel))
        pending = known[:self.config.opportunistic_limit] + unknown
        if not self.config.scan_all_unknown:
            pending = pending[:self.config.opportunistic_limit]
        if pending:
            self._log({'动作': '近距离停点补扫', '位置': point.tolist(), '候选频道': [s.channel for s in pending],
                       '未知频道建议间隔_米': self.config.unknown_scan_spacing_m,
                       '说明': '只复用当前停点；间隔是成本启发式，发现保证由末尾连续证书承担'})
        for state in pending:
            if self.complete():
                break
            self._measure(state, point, '顺路检测')

    def _all_sources_found(self):
        return sum(s.status in {'found', 'cleared'} for s in self.channels.values()) == 16

    def _sample_mask(self, point):
        key = tuple(map(float, point))
        if key not in self._sample_visibility:
            grid = np.arange(-1800, 1801, 100.)
            x, y = np.meshgrid(grid, grid)
            positions = np.column_stack((x.ravel(), y.ravel()))
            positions = positions[np.linalg.norm(positions, axis=1) <= 1800]
            boundary = np.arange(360) * math.pi / 180
            positions = np.vstack((positions, 1800 * np.column_stack((np.cos(boundary), np.sin(boundary)))))
            axes = np.arange(36) * math.pi / 18
            vectors = np.asarray(point) - positions
            self._sample_visibility[key] = ((np.linalg.norm(vectors, axis=1) <= 1000)[:, None]
                & ((vectors[:, :1] * np.cos(axes) + vectors[:, 1:] * np.sin(axes)) >= 0))
        return self._sample_visibility[key]

    def _has_sample_gain(self, state, point):
        visible = self._sample_mask(point)
        count, remaining = self._unknown_samples.get(state.channel, (0, np.ones_like(visible)))
        for negative in state.negative_sites[count:]:
            remaining &= ~self._sample_mask(negative)
        self._unknown_samples[state.channel] = (len(state.negative_sites), remaining)
        return bool(np.any(remaining & visible))

    def _observation_plan(self, state, current):
        if (not self.config.route_feasible_candidates or state.localization_count >= self.config.max_localization_measurements
                or self.config.planning_seconds == 0):
            return super()._observation_plan(state, current)
        # 先保留旧版已经合格的选择，仅在原候选会被路线规则拒绝时搜索替代。
        original = super()._observation_plan(state, current)
        if original['动作'] != '追加测向' or not len(self._remaining) or not self.config.route_planning:
            return original
        p = original['位置']
        proj = task_projection(current, p, original.get('预测目标代表位置', minimum_circle(state.region.vertices)[0]), self._remaining)
        detour = math.dist(current, p) + math.dist(p, self._remaining[0]) - math.dist(current, self._remaining[0])
        rejected = detour > self.config.localization_detour_limit_m or (self.config.task_routing
            and proj['预测完整任务绕行_米'] > self.config.localization_detour_limit_m
            and proj['预测完整任务绕行_米'] > proj['预测延后最小绕行_米'] + 1e-6)
        if not rejected:
            return original
        started = time.monotonic()
        vertices = state.region.vertices
        hypotheses = joint_hypotheses(state, self.config.position_samples) if self.config.direction_feedback else []
        representative = (np.average([p for p, _ in hypotheses], axis=0,
                          weights=[sum(b - a for a, b in arcs) for _, arcs in hypotheses])
                          if hypotheses else minimum_circle(vertices)[0])
        candidates = candidate_pool(state, hypotheses)
        before = len(candidates)
        admissible = []
        for p in candidates:
            projection = task_projection(current, p, representative, self._remaining)
            if len(self._remaining) and self.config.route_planning:
                detour = math.dist(current, p) + math.dist(p, self._remaining[0]) - math.dist(current, self._remaining[0])
                if detour > self.config.localization_detour_limit_m:
                    continue
                if (self.config.task_routing and projection['预测完整任务绕行_米'] > self.config.localization_detour_limit_m
                        and projection['预测完整任务绕行_米'] > projection['预测延后最小绕行_米'] + 1e-6):
                    continue
            admissible.append((p, projection))
        separated = [(p, proj) for p, proj in admissible
                     if math.dist(p, current) >= self.config.probe_spacing_m - 1e-7
                     and math.dist(p, state.measured_sites[-1]) >= self.config.probe_spacing_m - 1e-7]
        relaxed = bool(admissible) and not separated
        ranked = []
        for p, proj in separated or admissible:
            support = reception_support(hypotheses, p) if hypotheses else 1.
            if support <= 1e-8:
                continue
            travel = proj['预测完整任务绕行_米'] if self.config.task_routing and len(self._remaining) else math.dist(current, p)
            preliminary = travel / 5 + (1 - support) * finish_bound(vertices, p) if hypotheses else math.dist(current, p)
            ranked.append((preliminary, -support, p, proj))
        ranked.sort(key=lambda a: (a[0], a[1], *a[2]))
        best = {'动作': '覆盖清除', '预计费用_秒': finish_bound(vertices, current),
                '原因': '无合适路线候选或有限覆盖费用更低；有后续站时延后，末尾执行有限覆盖'}
        evaluated = 0
        limit = min(12, self.config.candidate_limit) if hypotheses else self.config.candidate_limit
        for _, negative_support, p, proj in ranked[:limit]:
            if evaluated and time.monotonic() - started >= self.config.planning_seconds:
                break
            low, high = angular_interval(vertices, p)
            costs = []
            for theta in np.linspace(low - math.radians(DELTA), high + math.radians(DELTA), self.config.angle_samples):
                branch = add_wedge(vertices, p, math.degrees(theta))
                if len(branch):
                    costs.append(finish_bound(branch, p))
            if not costs:
                continue
            evaluated += 1
            support = -negative_support
            score = math.dist(current, p) / 5 + (6 if hypotheses else 5) + support * max(costs)
            score += (1 - support) * finish_bound(vertices, p)
            if hypotheses and self.config.task_routing and len(self._remaining):
                score += (math.dist(representative, self._remaining[0]) - math.dist(current, self._remaining[0])) / 5
            if (hypotheses and best['动作'] != '追加测向') or score < best['预计费用_秒']:
                best = {'动作': '追加测向', '位置': p.tolist(), '预计费用_秒': score, **proj,
                        '接收保证': False, '距离保证': max_distance(vertices, p) <= 999.5,
                        '接收样本支持度': support if hypotheses else None,
                        '方向反馈已启用': bool(hypotheses), '最远可能目标距离_米': max_distance(vertices, p),
                        '距当前机器人_米': math.dist(current, p),
                        '距该频道上次观测_米': math.dist(state.measured_sites[-1], p)}
        best.update(路线筛选前候选数=before, 路线可行候选数=len(admissible), 已评估候选数=evaluated,
                    间距约束已放宽=relaxed, 建议相邻补测间距_米=self.config.probe_spacing_m,
                    规划耗时_秒=time.monotonic() - started,
                    评分性质='先满足绕行与完整任务约束，再作有限候选评分；支持度不是概率或接收保证')
        best['原候选绕行_米'] = detour
        return best

    def _coverage(self, state):
        key = tuple(map(tuple, state.negative_sites))
        if key not in self._cover_cache:
            self._cover_cache[key] = coverage_status(state.negative_sites, self.config.coverage_mesh)
        return self._cover_cache[key]

    def _refresh_absence(self):
        if not self.config.discovery_safeguard:
            return
        for state in self.channels.values():
            if state.status != 'unknown':
                continue
            certificate = self._coverage(state)
            if not certificate['未覆盖单元']:
                state.absence_certificate = {'方法': '三角网连续凸包发现证书', '网格': self.config.coverage_mesh, **certificate,
                                             '实际负观测站': state.negative_sites.copy()}
                state.status = 'absent'
                self._log({'动作': '频道不存在证书', '频道': state.channel, '证书': state.absence_certificate})

    def _needed_stations(self):
        if self._all_sources_found():
            return np.empty((0, 2))
        stations, triangles = discovery_mesh(self.config.coverage_mesh)
        needed = set()
        for state in self.channels.values():
            if state.status != 'unknown':
                continue
            for cell in self._coverage(state)['未覆盖单元']:
                for k in triangles[cell]:
                    if not any(np.array_equal(stations[k], p) for p in state.negative_sites):
                        needed.add(int(k))
        return open_route(self.client.state.position, stations[sorted(needed)])

    def _service_known(self, remaining=()):
        if len(remaining) or self._finishing or not self.config.discovery_safeguard:
            return super()._service_known(remaining)
        self._finishing = True
        self.phase = '连续发现保障与尾部清除'
        self._log({'动作': '开始发现保障', '说明': '未证全清时补足未知频道连续覆盖；复用实测站，必要时到区域外'})
        while not self.complete():
            self._guard()
            self._refresh_absence()
            route = self._needed_stations()
            super()._service_known(route)
            if self.complete():
                break
            self._refresh_absence()
            route = self._needed_stations()
            if not len(route):
                if not self.complete():
                    raise RuntimeError('无后续覆盖站但仍无全清证据，不能报告完成')
                break
            point = route[0]
            self._log({'动作': '发现保障路线', '位置序列': route.tolist(), '说明': '开放2-opt路线；处理顺路目标后再重规划'})
            pending = [s for s in self.channels.values() if s.status == 'unknown'
                       and not any(np.array_equal(point, p) for p in s.negative_sites)]
            pending.sort(key=lambda s: (s.channel != self.client.state.channel, s.channel))
            if not pending:
                raise RuntimeError('发现保障未取得进展')
            self.discovery_stations.append(point.tolist())
            self._log({'动作': '开始发现保障站', '位置': point.tolist(), '编号': len(self.discovery_stations),
                       '候选频道': [s.channel for s in pending]})
            for state in pending:
                if self.complete():
                    break
                self._measure(state, point, '发现保障补扫')
            # 新发现源由下一轮统一顺路服务，避免先将新源一次性全部清除再折返。
        self._finishing = False

    def assessment(self):
        return {'固定十三站是否全部完成': self.fixed_scan_finished,
                '已发现未清除频道': [c for c, s in self.channels.items() if s.status == 'found'],
                '待核查的未发现频道': [] if self.complete() else [c for c, s in self.channels.items() if s.status == 'unknown'],
                '仍需评估增站': not self.complete(), '自动新增未知源搜索站数': len(self.discovery_stations),
                '依据': '已清除16源或其余频道均有连续不存在证书' if self.complete() else '仍缺发现或清除证据',
                '本版处理': '保留13站主路线；必要时自动补足连续发现覆盖' if self.config.discovery_safeguard else '消融试验关闭发现保障，不能保证全清'}

    def run(self):
        result = super().run()
        result['方案'] = '第四问清除优先版：路线可行选点、近距离顺路补扫与连续发现保障'
        result['发现保障站坐标'] = self.discovery_stations
        result['已证明不存在频道'] = [c for c, s in self.channels.items() if s.status == 'absent']
        result['终止依据'] = ('达到16个不同频道目标上界' if sum(s.status == 'cleared' for s in self.channels.values()) == 16
                            else '已发现源全部清除，其他频道具备连续不存在证书' if self.complete()
                            else '尚无全清证据，必须检查异常或未核查频道')
        for row in result['频道记录']:
            row['不存在证书'] = self.channels[row['频道']].absence_certificate
        return result
