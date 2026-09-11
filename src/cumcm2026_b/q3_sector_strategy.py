"""方案一扇形闭环：智能站共用、按需补漏、完整区域清除证书。"""
from dataclasses import dataclass
import math
import time
import numpy as np

from .q2_physical_geometry import build_physical_region, analytic_candidates, centerline_interval
from .q3_geometry import (DELTA, minimum_circle, max_distance, min_distance, principal_axes,
                         angular_interval, add_wedge, sweep_path, search_stations)
from .q3_route_planning import task_point, route_length
from .q3_strategy import SchemeOne, StrategyConfig
from .q3_routed_strategies import RoutePlanningMixin
from .q3_sector_geometry import (intersect_sector, sector_midpoint, cached_sector_cover,
                                  cover_polygon, cover_gain, covers_holes)


@dataclass
class SectorConfig(StrategyConfig):
    coverage_radius_m: float = 999.5
    coverage_cell_m: float = 25.
    coverage_max_nodes: int = 4096
    reuse_min_distance_m: float = 40.
    reuse_min_gain: float = .5
    nearby_clear_detour_m: float = 80.
    route_action_limit: int = 24
    localization_detour_limit_m: float = 400.

    def __post_init__(self):
        super().__post_init__()
        for value in [self.coverage_radius_m, self.coverage_cell_m, self.reuse_min_distance_m, self.reuse_min_gain, self.nearby_clear_detour_m]:
            if isinstance(value, bool) or not math.isfinite(value): raise ValueError('扇形配置必须为有限数值')
        if not (969 < self.coverage_radius_m < 1000 and 0 < self.coverage_cell_m <= 100
                and self.reuse_min_distance_m >= 0 and 0 <= self.reuse_min_gain <= 1 and self.nearby_clear_detour_m >= 0):
            raise ValueError('覆盖余量、细分尺度或复用条件非法')
        if isinstance(self.coverage_max_nodes, bool) or not isinstance(self.coverage_max_nodes, int) or self.coverage_max_nodes < 100:
            raise ValueError('覆盖检查预算必须为不小于100的整数')
        if isinstance(self.route_action_limit, bool) or not isinstance(self.route_action_limit, int) or self.route_action_limit < 1:
            raise ValueError('初始段动作预算非法')
        if isinstance(self.localization_detour_limit_m, bool) or not math.isfinite(self.localization_detour_limit_m) or self.localization_detour_limit_m < 0:
            raise ValueError('初始段绕行阈值非法')


def task(state):
    center, radius = minimum_circle(state.region.vertices)
    return {'channel': state.channel, 'center': center, 'radius': radius,
            'ready': radius <= 19.5, 'vertices': state.region.vertices}


def sector_route(states, current, terminal):
    """所有本扇形任务均插在终点前；末扇形使用开放终点。"""
    tasks = {s.channel: task(s) for s in states}
    nodes = [{'类型': '当前位置', '位置': list(map(float, current))}]
    if terminal is not None: nodes.append({'类型': '下一固定站', '位置': list(map(float, terminal))})
    pending = set(tasks)
    while pending:
        best = None
        for channel in sorted(pending):
            for index in range(1, len(nodes)+(terminal is None)):
                a = np.asarray(nodes[index-1]['位置'])
                b = np.asarray(nodes[index]['位置']) if index < len(nodes) else None
                p = task_point(tasks[channel], a, b)
                delta = float(np.linalg.norm(p-a)+(np.linalg.norm(p-b)-np.linalg.norm(a-b) if b is not None else 0))
                key = delta, channel, index
                node = {'类型': '目标', '频道': channel, '位置': p.tolist(), '可安全清除': tasks[channel]['ready']}
                if best is None or key < best[0]: best = key, index, node
        _, index, node = best
        nodes.insert(index, node); pending.remove(node['频道'])
    for _ in range(8):
        base, improvement = route_length(nodes), None
        stop = len(nodes)-(terminal is not None)
        for i in range(1, stop-1):
            for j in range(i+1, stop):
                trial = nodes[:i]+nodes[i:j+1][::-1]+nodes[j+1:]
                if route_length(trial) < base-1e-6:
                    improvement = trial; break
            if improvement is not None: break
        if improvement is None: break
        nodes = improvement
    return {'计划路径': nodes, '预计几何路程_米': route_length(nodes),
            '性质': '本扇形所有目标必须在下一站之前处理；不确定圆心仅用于排列预测'}


def finish_to(vertices, current, terminal):
    """反馈分支的明确后续策略：安全点清除或完整有限覆盖，再衔接终点。"""
    center, radius = minimum_circle(vertices)
    if radius <= 19.5:
        item = {'center': center, 'ready': True, 'vertices': vertices}
        p = task_point(item, np.asarray(current), terminal)
        length = np.linalg.norm(p-current)+(np.linalg.norm(p-terminal) if terminal is not None else 0)
        return float(length/5+5)
    path = sweep_path(vertices, current)
    # 成功可能发生在任意覆盖点；每一前缀及其接续费用取最大。
    prefixes = np.linalg.norm(path[0]-current)+np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    tails = np.linalg.norm(path-terminal, axis=1) if terminal is not None else np.zeros(len(path))
    return float(np.max((prefixes+tails)/5+3*np.arange(len(path))+5))


class SchemeOneSector(SchemeOne):
    def __init__(self, client, config=None):
        super().__init__(client, config or SectorConfig())
        self.sector = None
        self.finished_sectors = []
        self.sector_absence = {c: set() for c in self.channels}
        self.fill_sites = []

    def _cover(self, state, vertices=None):
        cfg = self.config
        sites = tuple(tuple(p) for p in state.negative_sites)
        if vertices is None:
            return cached_sector_cover(self.sector, sites, cfg.coverage_radius_m, cfg.coverage_cell_m, cfg.coverage_max_nodes)
        return cover_polygon(vertices, sites, cfg.coverage_radius_m, cfg.coverage_cell_m, cfg.coverage_max_nodes)

    def _responsibility(self, state):
        if state.status in {'cleared', 'absent'}: return {'方式': state.status}
        if state.status == 'unknown':
            cover = self._cover(state)
            return {'方式': '无信号圆联合覆盖', '核查单元': cover['checked']} if cover['covered'] else None
        clipped = intersect_sector(state.region.vertices, self.sector)
        if not len(clipped): return {'方式': '完整可能区域与扇形无交集'}
        if state.negative_sites:
            cover = self._cover(state, clipped)
            if cover['covered']: return {'方式': '扇形交集被该频道无信号圆排除', '核查单元': cover['checked']}
        return None

    def _unknown_holes(self):
        return [(s, self._cover(s)['holes']) for s in self.channels.values()
                if s.status == 'unknown' and self._responsibility(s) is None]

    def _reuse_location(self, point, primary=None):
        """实际到站后逐频道检测，不把仅经过或别的频道观测计入覆盖。"""
        if not self.config.opportunistic:
            self._clear_nearby_ready()
            return
        pending = []
        for state in self.channels.values():
            if state.channel == primary or state.status not in {'unknown', 'found'}: continue
            distance = min((np.linalg.norm(point-p) for p in state.measured_sites), default=math.inf)
            if distance < 1e-6: continue
            if state.status == 'unknown':
                holes = self._cover(state)['holes']
                if not len(holes): continue
                if cover_gain(holes, point, self.config.coverage_radius_m) < self.config.reuse_min_gain and not covers_holes(holes, point, self.config.coverage_radius_m): continue
            else:
                if distance < self.config.reuse_min_distance_m: continue
                if minimum_circle(state.region.vertices)[1] <= 19.5: continue
                # 共用站是机会检测，允许无信号；不冒充主目标的保证接收测向。
                if min_distance(state.region.vertices, point) > 999.5: continue
            pending.append(state)
        pending.sort(key=lambda s: (s.channel != self.client.state.channel, s.channel))
        for state in pending:
            if self.complete(): break
            if state.status not in {'unknown', 'found'}: continue
            self._measure(state, point, '智能站复用扫描')
        self._clear_nearby_ready()

    def _clear_nearby_ready(self):
        # 已定位且非常顺路的未来扇形目标允许提前清除，避免近圆心源分六次返回处理。
        terminal = search_stations()[self.sector+2] if self.sector < 5 else None
        while not self.complete():
            current = np.asarray(self.client.state.position)
            choices = []
            for state in self.channels.values():
                if state.status != 'found': continue
                item = task(state)
                if not item['ready']: continue
                q = task_point(item, current, terminal)
                distance = float(np.linalg.norm(q-current))
                extra = distance+(float(np.linalg.norm(q-terminal)-np.linalg.norm(current-terminal)) if terminal is not None else 0)
                if extra <= self.config.nearby_clear_detour_m and distance <= 200:
                    choices.append((extra, distance, state.channel, q))
            if not choices: break
            _, _, channel, q = min(choices, key=lambda x: x[:3])
            self._clear(self.channels[channel], q, '邻近已定位目标提前清除', certified=True)

    def _measurement_plan(self, state, current, terminal, holes):
        cfg, vertices = self.config, state.region.vertices
        started = time.monotonic()
        best = {'动作': '覆盖清除', '预计服务费用_秒': finish_to(vertices, current, terminal)}
        if state.localization_count >= cfg.max_localization_measurements or cfg.planning_seconds == 0: return best
        center, axes = principal_axes(vertices)
        candidates = [np.asarray(current), sector_midpoint(self.sector)]
        if terminal is not None:
            direction = terminal-current
            normal = np.array([-direction[1], direction[0]])/max(1., np.linalg.norm(direction))
            for t in [.25, .5, .75]:
                candidates.extend(current+t*direction+offset*normal for offset in [-150, 0, 150])
        for angle in np.arange(8)*math.pi/4:
            direction = axes@np.array([math.cos(angle), math.sin(angle)])
            candidates.extend(center+r*direction for r in [30, 75, 150, 300, 500, 650])
        positives = [o for o in state.region.observations if o['结果'] == 'direction']
        if positives:
            first = positives[0]
            physical = build_physical_region(first['位置'], first['示向度'], DELTA)
            if centerline_interval(physical) is not None:
                for p in analytic_candidates(physical): candidates.extend([center+.8*(p-center), center+.9*(p-center)])
        feasible = []
        for p in candidates:
            p = np.asarray(p)
            if max_distance(vertices, p) > 999.5 or min_distance(vertices, p) <= 5.5: continue
            if any(np.linalg.norm(p-old) < .01 for old in state.measured_sites+feasible): continue
            feasible.append(p)
        # 两种费用都使用相同服务终点。覆盖收益仅用于接近的候选排序，不能充当证书。
        def gain(p): return sum(cover_gain(h, p, cfg.coverage_radius_m) for _, h in holes)
        feasible.sort(key=lambda p: (np.linalg.norm(p-current)+(np.linalg.norm(p-terminal) if terminal is not None else 0))/5-6*gain(p))
        evaluated = 0
        for p in feasible[:cfg.candidate_limit]:
            if time.monotonic()-started >= cfg.planning_seconds: break
            low, high = angular_interval(vertices, p)
            predictions = []
            for angle in np.linspace(low-math.radians(DELTA), high+math.radians(DELTA), cfg.angle_samples):
                branch = add_wedge(vertices, p, math.degrees(angle))
                if len(branch): predictions.append(finish_to(branch, p, terminal))
            if not predictions: continue
            evaluated += 1
            fee = float(np.linalg.norm(p-current)/5+5+(state.channel != self.client.state.channel)+max(predictions))
            credit = 6*gain(p)  # 一个完整未知频道覆盖机会以一次检测及切换的6秒计作排序启发式。
            if fee-credit < best.get('评分_秒', best['预计服务费用_秒']):
                best = {'动作': '追加测向', '位置': p.tolist(), '预计服务费用_秒': fee,
                        '覆盖机会启发式_秒': credit, '评分_秒': fee-credit}
        return {**best, '已评估候选数': evaluated, '预测角节点': cfg.angle_samples,
                '规划耗时_秒': time.monotonic()-started, '评分性质': '有限反馈采样与覆盖机会启发式，非连续最优保证'}

    def _fill_coverage(self, terminal):
        holes = self._unknown_holes()
        if not holes: return
        current = np.asarray(self.client.state.position)
        candidates = [current, sector_midpoint(self.sector)]
        for _, h in holes:
            centers = h.mean(axis=1)
            candidates.append(centers.mean(axis=0))
        if terminal is not None:
            candidates.extend(current+t*(terminal-current) for t in [.25, .5, .75])
        # 只有能补足全部待覆盖频道的候选才可作为本轮独立补漏点，拒绝无界逐小块追逐。
        feasible = []
        for p in candidates:
            valid = True
            for state, _ in holes:
                sites = tuple(tuple(q) for q in state.negative_sites)+ (tuple(p),)
                if not cached_sector_cover(self.sector, sites, self.config.coverage_radius_m,
                                           self.config.coverage_cell_m, self.config.coverage_max_nodes)['covered']:
                    valid = False; break
            if valid: feasible.append(p)
        if not feasible: raise RuntimeError('扇形覆盖候选未取得证书，不允许推进')
        point = min(feasible, key=lambda p: np.linalg.norm(p-current)+(np.linalg.norm(p-terminal) if terminal is not None else 0))
        self._log({'动作': '按需补漏站', '扇形': self.sector, '位置': list(map(float, point)),
                   '需扫描频道': [s.channel for s, _ in holes], '是否新增移动': bool(np.linalg.norm(point-current) > 1e-6)})
        self.fill_sites.append(list(map(float, point)))
        for state, _ in holes:
            if self.complete(): break
            if state.status != 'unknown': continue
            if any(np.linalg.norm(point-old) < 1e-6 for old in state.measured_sites):
                raise RuntimeError('补漏点重复且覆盖仍未完成')
            self._measure(state, point, '按需覆盖补漏')
        self._reuse_location(point)

    def _service_known(self):
        if not self.completed_search_stations: return
        if self.completed_search_stations[-1] == 0:
            # 原有初始段滚动服务避免先到东方、再分六次返回中央处理近源。
            # 初始段不验收扇形；400米/24动作仅约束这里，不能否决后续本扇形责任。
            return RoutePlanningMixin._route_service(self, search_stations()[1:])
        sector = self.completed_search_stations[-1]-1
        if sector in self.finished_sectors: return
        self.sector = sector
        terminal = search_stations()[sector+2] if sector < 5 else None
        self._log({'动作': '开始扇形', '扇形': sector, '角度范围_度': [60*sector, 60*(sector+1)]})
        while not self.complete():
            self._guard()
            known = [s for s in self.channels.values() if s.status == 'found' and self._responsibility(s) is None]
            if not known:
                if self._unknown_holes():
                    self._fill_coverage(terminal); continue
                break
            current = np.asarray(self.client.state.position)
            route = sector_route(known, current, terminal)
            self._log({'动作': '扇形路线规划', '扇形': sector, **route})
            node = route['计划路径'][1]
            state = self.channels[node['频道']]
            if node['可安全清除']:
                point = np.asarray(node['位置'])
                if max_distance(state.region.vertices, point) > 19.5+1e-7: raise RuntimeError('完整区域清除证书失效')
                self._clear(state, point, '扇形顺路证书清除', certified=True)
                self._reuse_location(point, state.channel)
            else:
                # 以计划中紧随当前目标的节点为接续点，兼顾同扇形目标组与下一固定站。
                following = np.asarray(route['计划路径'][2]['位置']) if len(route['计划路径']) > 2 else terminal
                plan = self._measurement_plan(state, current, following, self._unknown_holes())
                self._log({'动作': '规划', '扇形': sector, '频道': state.channel, '方案': plan})
                if plan['动作'] == '追加测向':
                    state.localization_count += 1
                    point = np.asarray(plan['位置'])
                    code = self._measure(state, point, '扇形追加测向')
                    if code == 'no_signal': raise RuntimeError('可靠候选出现无信号，停止核对模型')
                    self._reuse_location(point, state.channel)
                else:
                    path = sweep_path(state.region.vertices, current)
                    self._log({'动作': '有限覆盖', '扇形': sector, '频道': state.channel, '覆盖点数': len(path)})
                    for point in path:
                        self._clear(state, point, '扇形有限方格覆盖', certified=False)
                        if state.status == 'cleared': break
                    if state.status != 'cleared': raise RuntimeError('完整有限覆盖未清除目标')
                    self._reuse_location(np.asarray(self.client.state.position), state.channel)
        proofs = {}
        for c, state in self.channels.items():
            proof = self._responsibility(state)
            if proof is None and self.complete(): proof = {'方式': '全局已达16目标上界'}
            if proof is None: raise RuntimeError('扇形仍有未完成频道，禁止推进')
            proofs[c] = proof
            if state.status == 'unknown':
                self.sector_absence[c].add(sector)
                if len(self.sector_absence[c]) == 6:
                    state.status = 'absent'
                    state.absence_certificate = {'方式': '六闭扇形逐频道排除', '扇形': list(range(6)), '接收半径_米': self.config.coverage_radius_m}
        self.finished_sectors.append(sector)
        self._log({'动作': '扇形清空证书', '扇形': sector, '逐频道证据': proofs,
                   '覆盖核验半径_米': self.config.coverage_radius_m})

    def run(self):
        result = super().run()
        result['方案'] = '方案一：七固定站、扇形清空、智能站复用及按需补漏'
        result['已验收扇形'] = self.finished_sectors
        result['扇形调度统计'] = {
            '路线重规划次数': sum(e['动作'] == '扇形路线规划' for e in self.events),
            '补漏站决策次数': len(self.fill_sites),
            '补漏新增移动次数': sum(e.get('是否新增移动', False) for e in self.events),
            '智能站复用检测次数': sum(e.get('用途') == '智能站复用扫描' for e in self.events),
            '补漏检测次数': sum(e.get('用途') == '按需覆盖补漏' for e in self.events),
        }
        return result
