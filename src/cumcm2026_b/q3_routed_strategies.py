"""两方案共享路线调度：顺路处理目标，保留固定搜索推进与最终清除闭环。"""
from dataclasses import dataclass
import math
import numpy as np

from .q3_geometry import max_distance, search_stations, sweep_path
from .q3_protocol import ProtocolError
from .q3_route_planning import plan_route
from .q3_strategy import SchemeOne, StrategyConfig, observation_plan
from .q3_scheme2_localized import SchemeTwoLocalized, SchemeTwoLocalizedConfig


def validate_route_config(config):
    if (not isinstance(config.route_planning, bool) or isinstance(config.route_action_limit, bool)
            or not isinstance(config.route_action_limit, int) or config.route_action_limit < 1
            or isinstance(config.localization_detour_limit_m, bool)
            or not math.isfinite(config.localization_detour_limit_m) or config.localization_detour_limit_m < 0):
        raise ValueError("路线开关、动作上限或测向绕行阈值不合法")


@dataclass
class RoutedOneConfig(StrategyConfig):
    route_planning: bool = True
    route_action_limit: int = 24
    localization_detour_limit_m: float = 400.0

    def __post_init__(self):
        super().__post_init__()
        validate_route_config(self)


@dataclass
class RoutedTwoConfig(SchemeTwoLocalizedConfig):
    route_planning: bool = True
    route_action_limit: int = 24
    localization_detour_limit_m: float = 400.0

    def __post_init__(self):
        super().__post_init__()
        validate_route_config(self)


class RoutePlanningMixin:
    def _route_service(self, remaining_stations):
        """有下一固定站时只做被分配到当前路段的服务；尾段保证处理全部已知目标。"""
        remaining = np.asarray(remaining_stations, dtype=float).reshape(-1, 2)
        deferred = set()
        starting_actions = self.actions
        while not self.complete():
            self._guard()
            known = [s for s in self.channels.values() if s.status == "found" and s.channel not in deferred]
            if not known:
                return
            if len(remaining) and self.actions - starting_actions >= self.config.route_action_limit:
                self._log({"动作": "继续固定搜索", "原因": "达到本路段服务动作上限"})
                return
            current = np.asarray(self.client.state.position)
            route = plan_route(known, current, remaining)
            self._log({"动作": "路线规划", **route})
            next_node = route["计划路径"][1]
            if next_node["类型"] == "固定站":
                return
            state = self.channels[next_node["频道"]]
            point = np.asarray(next_node["位置"])
            if next_node["可安全清除"]:
                if max_distance(state.region.vertices, point) > 19.5 + 1e-7:
                    raise RuntimeError("路线规划清除点证书失效")
                self._clear(state, point, "顺路证书清除" if len(remaining) else "路线规划证书清除", certified=True)
                continue
            plan = observation_plan(state, current, self.config)
            if plan["动作"] == "追加测向":
                if state.channel != self.client.state.channel:
                    plan["预计费用_秒"] += 1
                point = np.asarray(plan["位置"])
                detour = (float(np.linalg.norm(current-point) + np.linalg.norm(point-remaining[0])
                                - np.linalg.norm(current-remaining[0])) if len(remaining) else 0.)
                if len(remaining) and detour > self.config.localization_detour_limit_m:
                    deferred.add(state.channel)
                    self._log({"动作": "延后目标", "频道": state.channel, "原因": "本次测向偏离下一站路线过多",
                               "本次测向绕行_米": detour, "阈值_米": self.config.localization_detour_limit_m})
                    continue
                self._log({"动作": "规划", "频道": state.channel, "方案": plan,
                           "路线阶段": "站间顺路处理" if len(remaining) else "最后集中处理", "本次测向绕行_米": detour})
                state.localization_count += 1
                feedback = self._measure(state, point, "追加测向")
                if feedback == "no_signal":
                    raise RuntimeError("可靠接收候选出现无信号，停止核对模型")
                self._opportunistic(point, state.channel)
            elif len(remaining):
                deferred.add(state.channel)
                self._log({"动作": "延后目标", "频道": state.channel, "原因": "暂不在固定站之间执行有限光学搜索"})
            else:
                path = sweep_path(state.region.vertices, current)
                self._log({"动作": "有限覆盖", "频道": state.channel, "覆盖点数": len(path), "单元边长_米": 20})
                for point in path:
                    self._clear(state, point, "有限方格覆盖", certified=False)
                    if state.status == "cleared":
                        break
                if state.status != "cleared":
                    raise RuntimeError("有限覆盖完成仍未清除，模型或协议异常")

    def run(self):
        result = super().run()
        result["启用路线规划"] = self.config.route_planning
        if self.config.route_planning:
            result["方案"] += "；固定站间顺路处理与滚动路线规划"
        result["路线调度统计"] = {
            "重规划次数": sum(e["动作"] == "路线规划" for e in self.events),
            "延后决策次数": sum(e["动作"] == "延后目标" for e in self.events),
            "顺路证书清除次数": sum(e.get("用途") == "顺路证书清除" for e in self.events),
        }
        return result


class SchemeOneRouted(RoutePlanningMixin, SchemeOne):
    def __init__(self, client, config=None):
        super().__init__(client, config or RoutedOneConfig())

    def _service_known(self):
        if not self.config.route_planning:
            return super()._service_known()
        next_index = self.completed_search_stations[-1] + 1 if self.completed_search_stations else 0
        return self._route_service(search_stations()[next_index:])


class SchemeTwoRouted(RoutePlanningMixin, SchemeTwoLocalized):
    def __init__(self, client, config=None):
        super().__init__(client, config or RoutedTwoConfig())
        self.station_hook_enabled = self.config.route_planning

    def _after_fixed_station(self, index, point):
        remaining = self.layout["站点"][index+1:]
        if self.config.route_planning and remaining:
            self._route_service(remaining)

    def _service_known(self):
        if not self.config.route_planning:
            return super()._service_known()
        return self._route_service([])
