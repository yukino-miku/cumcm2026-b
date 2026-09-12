"""第四问方案二：只扩展固定站集合及顺序，动态服务继承当前方案一。"""
from dataclasses import asdict, dataclass
import time

from ..q3_protocol import BudgetExceeded, ProtocolError
from ..q4_reliable_strategy import ReliableConfig, ReliableFour
from .route import shortest_route


@dataclass
class SchemeTwoConfig(ReliableConfig):
    discovery_safeguard: bool = False
    unknown_scan_spacing_m: float = 600.
    unknown_novelty: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.discovery_safeguard:
            raise ValueError('方案二只增加6个固定站；不得开启已取消的自动补漏保障')


class SchemeTwo(ReliableFour):
    def __init__(self, client, config=None):
        config = config or SchemeTwoConfig()
        if config.discovery_safeguard:
            raise ValueError('方案二不启用自动补漏保障')
        super().__init__(client, config)
        self.layout, self.route_certificate = shortest_route()

    def assessment(self):
        return {'固定十九站是否全部完成': self.fixed_scan_finished,
                '计划固定站数': len(self.layout),
                '已发现未清除频道': [c for c, s in self.channels.items() if s.status == 'found'],
                '待核查的未发现频道': [] if self.complete() else [c for c, s in self.channels.items() if s.status == 'unknown'],
                '仍需评估增站': not self.complete(), '自动新增未知源搜索站数': 0,
                '依据': '达到16个目标上界' if self.complete() else
                        '19站不构成任意发射朝向的全域发现证明；未知频道可能不存在，也可能尚未被接收',
                '本版处理': '19固定站及已发现源处理后退出；不自动为未知频道追加补漏站'}

    # 沿用原固定扫描循环，仅把13站常量改为布局长度及19站结果说明。
    # _service_known/_observation_plan/_opportunistic/_skip_measure/_clear均直接继承。
    def run(self):
        self.started = time.monotonic()
        try:
            self.client.enter()
            self.client.state.deadline = min(self.client.state.deadline, self.client.clock()+self.config.runtime_limit)
            self.phase = '十九站固定扫描'
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
            self.fixed_scan_finished = len(self.completed_search_stations) == len(self.layout)
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
        return {'方案': '第四问方案二：19站最短固定路线，沿用顺路定位清除及路线修复',
                '配置': asdict(self.config), '运行成功': flow_ok and self.complete(),
                '流程正常完成': flow_ok, '全部完成证据': self.complete(),
                '正常退出': robot.exited, '异常': self.error,
                '结束状态': '已证全清' if flow_ok and self.complete() else '试验流程完成，未发现频道待核查' if flow_ok else '流程未完成',
                '清除数': len(cleared), '已清除频道': cleared,
                '未发现频道': [c for c, s in self.channels.items() if s.status == 'unknown'],
                '已发现未清除频道': [c for c, s in self.channels.items() if s.status == 'found'],
                '已证明不存在频道': [], '终止依据': '达到16个不同频道目标上界' if self.complete() else '19固定站及已发现源处理完成；不自动未知源补漏，未宣称全清',
                '虚拟总时间_秒': robot.virtual_time, '总路程_米': robot.distance,
                '平均定位清除时间_秒': robot.virtual_time/len(cleared) if cleared else None,
                '程序运行时间_秒': time.monotonic()-self.started,
                '检测次数': robot.measurements, '频道切换次数': robot.switches,
                '清除成功次数': robot.clear_success, '清除失败次数': robot.clear_failure,
                '时间分项_秒': robot.time_parts(), '已完成搜索站': self.completed_search_stations,
                '固定站坐标': self.layout.tolist(), '末尾增站评估': self.assessment(),
                '固定路线最优证书': self.route_certificate, '发现保障站坐标': [],
                '路线调度统计': {'重规划次数': sum(e['动作'] == '路线规划' for e in self.events),
                                 '延后决策次数': sum(e['动作'] == '延后目标' for e in self.events),
                                 '顺路证书清除次数': sum(e.get('用途') == '顺路证书清除' for e in self.events),
                                 '补测无信号次数': sum(e['动作'] == '补测无信号' for e in self.events)},
                '频道记录': [{'频道': c, '状态': s.status, '测向记录': s.region.observations,
                             '排除约束': s.region.exclusions, '不存在证书': None,
                             '追加测向次数': s.localization_count} for c, s in self.channels.items()],
                '动作记录': self.events}

