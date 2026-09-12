"""第四问13站试验版：沿用路线调度，允许定向补测无信号，末尾只评估不自动外围搜索。"""
from dataclasses import asdict, dataclass, field
import math
import time
import numpy as np

from .q3_geometry import (TargetRegion, minimum_circle, max_distance, min_distance,
                          clear_point, sweep_path, finish_bound)
from .q3_protocol import BudgetExceeded, ProtocolError
from .q3_route_planning import plan_route
from .q3_routed_strategies import RoutedOneConfig
from .q3_strategy import SchemeOne, ChannelState, observation_plan


def search_stations():
    a, h = 950., 950.*math.sqrt(3)/2
    return np.array([(0, 0), (a, 0), (1.5*a, h), (.5*a, h), (0, 2*h),
                     (-.5*a, h), (-1.5*a, h), (-a, 0), (-1.5*a, -h),
                     (-.5*a, -h), (0, -2*h), (.5*a, -h), (1.5*a, -h)])


class DirectionalRegion(TargetRegion):
    def observe(self, point, result, bearing=None):
        if result == 'no_signal':
            self.observations.append({'位置': list(map(float, point)), '结果': result, '示向度': None,
                                      '解释': '不存在、超出半径或处于发射背侧；不作1000米负圆排除'})
            return
        super().observe(point, result, bearing)


@dataclass
class DirectionalChannel(ChannelState):
    region: DirectionalRegion = field(default_factory=DirectionalRegion)


@dataclass
class Q4Config(RoutedOneConfig):
    skip_localized: bool = True
    consecutive_no_signal_limit: int = 2

    def __post_init__(self):
        super().__post_init__()
        for name in ['angle_samples', 'candidate_limit', 'max_localization_measurements', 'max_actions',
                     'opportunistic_limit', 'route_action_limit', 'consecutive_no_signal_limit']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name}必须为整数')
        if (self.consecutive_no_signal_limit < 1 or not isinstance(self.skip_localized, bool)
                or not isinstance(self.opportunistic, bool)
                or not math.isfinite(self.planning_seconds) or not math.isfinite(self.runtime_limit)):
            raise ValueError('第四问策略参数不合法')


class SchemeFour(SchemeOne):
    def __init__(self, client, config=None):
        super().__init__(client, config or Q4Config())
        self.channels = {c: DirectionalChannel(c) for c in range(1, 21)}
        self.layout = search_stations()
        self.phase = '未开始'
        self.fixed_scan_finished = False
        self.flow_finished = False

    def _log(self, event):
        super()._log({'阶段': self.phase, **event})

    def _measure(self, state, point, reason):
        self._guard()
        if state.status not in {'unknown', 'found'}:
            raise RuntimeError('不能重复检测已完成频道')
        point = np.asarray(point, dtype=float)
        result = self.client.measure(point, state.channel)
        self.actions += 1
        state.measured_sites.append(point.tolist())
        code = result['measure_result']
        state.region.observe(point, code, result.get('svd_deg'))
        if code == 'no_signal':
            state.negative_sites.append(point.tolist())
        else:
            state.status = 'found'
        event = {'动作': '检测', '用途': reason, '频道': state.channel, '位置': point.tolist(), '结果': code,
                 '示向度': result.get('svd_deg'), '频道状态': state.status}
        if code in {'direction', 'near'}:
            # near的5米外包已经以检测点为圆心，无需对128边形穷举最小覆盖圆。
            radius = max_distance(state.region.vertices, point) if code == 'near' else minimum_circle(state.region.vertices)[1]
            event.update({'外包顶点': state.region.vertices.tolist(), '外包覆盖圆半径_米': radius})
        self._log(event)
        if code == 'near':
            self._clear(state, point, 'near原地清除', certified=True)
        return code

    def complete(self):
        # 13站无完整方向覆盖证书；未发现频道始终保持unknown。
        return sum(s.status == 'cleared' for s in self.channels.values()) == 16

    def _skip_measure(self, state, point):
        if any(np.linalg.norm(point-np.asarray(p)) < 1e-6 for p in state.measured_sites):
            self._log({'动作': '跳过检测', '频道': state.channel, '位置': point.tolist(), '原因': '同一位置已测过该频道'})
            return True
        if self.config.skip_localized and state.status == 'found':
            center, radius = minimum_circle(state.region.vertices)
            if radius <= 19.5:
                self._log({'动作': '跳过检测', '频道': state.channel, '位置': point.tolist(),
                           '原因': '已具备安全清除位置', '清除位置': center.tolist(),
                           '外包覆盖圆半径_米': radius, '外包顶点': state.region.vertices.tolist()})
                return True
        return False

    def _opportunistic(self, point, primary):
        if not self.config.opportunistic:
            return
        eligible = []
        for state in self.channels.values():
            if state.channel == primary or state.status not in {'unknown', 'found'}:
                continue
            if state.measured_sites and min(np.linalg.norm(np.asarray(p)-point) for p in state.measured_sites) < 600:
                continue
            if state.status == 'found' and max_distance(state.region.vertices, point) > 999.5:
                continue
            if self._skip_measure(state, np.asarray(point)):
                continue
            eligible.append(state)
        eligible.sort(key=lambda s: (s.status != 'found', len(s.measured_sites), s.channel))
        for state in eligible[:self.config.opportunistic_limit]:
            self._measure(state, point, '顺路检测')
            if self.complete():
                break

    def _observation_plan(self, state, current):
        # 第一版沿用第三问的有限候选和角采样评分，接收条件改为正常反馈分支。
        # 只使用正观测外包，不以未知朝向的采样近似删除真实可能位置。
        plan = observation_plan(state, current, self.config)
        plan['评分性质'] = '收到方向后的几何费用启发式；不保证下一站接收，不代表定向源期望费用'
        if plan['动作'] == '追加测向':
            point = np.asarray(plan['位置'])
            plan['无信号后的保守回退费用_秒'] = finish_bound(state.region.vertices, point)
            plan['接收保证'] = False
        return plan

    def _service_known(self, remaining=()):
        remaining = np.asarray(remaining, dtype=float).reshape(-1, 2)
        deferred, misses = set(), {}
        starting_actions = self.actions
        while not self.complete():
            self._guard()
            known = [s for s in self.channels.values() if s.status == 'found' and s.channel not in deferred]
            if not known:
                return
            if len(remaining) and self.actions-starting_actions >= self.config.route_action_limit:
                self._log({'动作': '继续固定搜索', '原因': '达到本路段服务动作上限'})
                return
            current = np.asarray(self.client.state.position)
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
                detour = float(np.linalg.norm(current-point)+np.linalg.norm(point-remaining[0])
                               -np.linalg.norm(current-remaining[0])) if len(remaining) else 0.
                if len(remaining) and self.config.route_planning and detour > self.config.localization_detour_limit_m:
                    deferred.add(state.channel)
                    self._log({'动作': '延后目标', '频道': state.channel, '原因': '本次补测偏离下一固定站路线过多',
                               '本次测向绕行_米': detour, '阈值_米': self.config.localization_detour_limit_m})
                    continue
                self._log({'动作': '规划', '频道': state.channel, '方案': plan, '本次测向绕行_米': detour})
                state.localization_count += 1
                feedback = self._measure(state, point, '追加测向')
                if feedback == 'no_signal':
                    misses[state.channel] = misses.get(state.channel, 0)+1
                    self._log({'动作': '补测无信号', '频道': state.channel, '位置': point.tolist(),
                               '处理': '保留完整位置可能区域；重规划或等待后续固定站'})
                    if len(remaining) and misses[state.channel] >= self.config.consecutive_no_signal_limit:
                        deferred.add(state.channel)
                        self._log({'动作': '延后目标', '频道': state.channel, '原因': '本路段连续无信号，继续固定扫描'})
                else:
                    misses[state.channel] = 0
                self._opportunistic(point, state.channel)
            elif len(remaining):
                deferred.add(state.channel)
                self._log({'动作': '延后目标', '频道': state.channel, '原因': '有限光学覆盖留到固定扫描后'})
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
        self.started = time.monotonic()
        try:
            self.client.enter()
            self.client.state.deadline = min(self.client.state.deadline, self.client.clock()+self.config.runtime_limit)
            self.phase = '十三站固定扫描'
            for index, point in enumerate(self.layout):
                if self.complete():
                    break
                self._log({'动作': '开始固定站', '固定站编号': index+1, '位置': point.tolist()})
                pending = [s for s in self.channels.values() if s.status in {'unknown', 'found'}]
                pending.sort(key=lambda s: (s.channel != self.client.state.channel, s.channel))
                for state in pending:
                    if state.status not in {'unknown', 'found'} or self._skip_measure(state, point):
                        continue
                    self._measure(state, point, f'固定搜索站{index+1}')
                    if self.complete():
                        break
                self.completed_search_stations.append(index)
                self._log({'动作': '固定站完成', '固定站编号': index+1, '位置': point.tolist()})
                if index < len(self.layout)-1:
                    self._service_known(self.layout[index+1:])
            self.fixed_scan_finished = len(self.completed_search_stations) == 13
            self.phase = '固定扫描后处理已发现源'
            self._log({'动作': '固定扫描完成', '已完成站数': len(self.completed_search_stations),
                       '全清提前停止': self.complete()})
            self._service_known()
            if any(s.status == 'found' for s in self.channels.values()):
                raise RuntimeError('仍有已发现目标尚未完成处理')
            self.phase = '末尾增站评估'
            self._log({'动作': '末尾增站评估', **self.assessment()})
            self.flow_finished = True
        except (BudgetExceeded, ProtocolError, RuntimeError, ValueError) as error:
            self.error = f'{type(error).__name__}: {error}'
            self._log({'动作': '异常停止', '说明': self.error})
        finally:
            if self.client.state.entered and not self.client.state.exited and self.client.pending is None:
                try:
                    self.client.exit()
                except (BudgetExceeded, ProtocolError) as error:
                    self.error = (self.error+'；' if self.error else '')+f'退出未确认：{error}'
        robot = self.client.state
        cleared = [c for c, s in self.channels.items() if s.status == 'cleared']
        flow_ok = self.flow_finished and robot.exited and self.error is None
        return {'方案': '第四问13站试验版：顺路定位清除；末尾评估，不自动外围搜索',
                '配置': asdict(self.config), '运行成功': flow_ok and self.complete(),
                '流程正常完成': flow_ok, '全部完成证据': self.complete(),
                '正常退出': robot.exited, '异常': self.error,
                '结束状态': '已证全清' if flow_ok and self.complete() else '试验流程完成，未发现频道待核查' if flow_ok else '流程未完成',
                '清除数': len(cleared), '已清除频道': cleared,
                '未发现频道': [c for c, s in self.channels.items() if s.status == 'unknown'],
                '已发现未清除频道': [c for c, s in self.channels.items() if s.status == 'found'],
                '已证明不存在频道': [], '终止依据': '达到16个不同频道目标上界' if self.complete() else '用户指定13站试验结束，未宣称全清',
                '虚拟总时间_秒': robot.virtual_time, '总路程_米': robot.distance,
                '平均定位清除时间_秒': robot.virtual_time/len(cleared) if cleared else None,
                '程序运行时间_秒': time.monotonic()-self.started,
                '检测次数': robot.measurements, '频道切换次数': robot.switches,
                '清除成功次数': robot.clear_success, '清除失败次数': robot.clear_failure,
                '时间分项_秒': robot.time_parts(), '已完成搜索站': self.completed_search_stations,
                '固定站坐标': self.layout.tolist(), '末尾增站评估': self.assessment(),
                '路线调度统计': {'重规划次数': sum(e['动作'] == '路线规划' for e in self.events),
                                 '延后决策次数': sum(e['动作'] == '延后目标' for e in self.events),
                                 '顺路证书清除次数': sum(e.get('用途') == '顺路证书清除' for e in self.events),
                                 '补测无信号次数': sum(e['动作'] == '补测无信号' for e in self.events)},
                '频道记录': [{'频道': c, '状态': s.status, '测向记录': s.region.observations,
                             '排除约束': s.region.exclusions, '不存在证书': None,
                             '追加测向次数': s.localization_count} for c, s in self.channels.items()],
                '动作记录': self.events}

    def assessment(self):
        return {'固定十三站是否全部完成': self.fixed_scan_finished,
                '已发现未清除频道': [c for c, s in self.channels.items() if s.status == 'found'],
                '待核查的未发现频道': [] if self.complete() else [c for c, s in self.channels.items() if s.status == 'unknown'],
                '仍需评估增站': not self.complete(), '自动新增未知源搜索站数': 0,
                '依据': '已达到16个目标上界，无需继续搜索' if self.complete() else
                        '13站不提供任意朝向的完整发现证书；未发现频道可能不存在或尚未被接收，不能从无信号推断剩余目标数',
                '本版处理': '保存结果并退出，不自动执行外侧发现补测'}
