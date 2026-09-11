"""方案一：七站发现保证、逐源自适应测向及时间调度的可执行基线。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import time
import numpy as np

from .q2_physical_geometry import build_physical_region, analytic_candidates, centerline_interval
from .q3_geometry import (TargetRegion, DELTA, search_stations, coverage_certificate, minimum_circle,
    max_distance, min_distance, clear_point, principal_axes, angular_interval, add_wedge, finish_bound, sweep_path)
from .q3_protocol import RobotClient, ProtocolError, BudgetExceeded


@dataclass
class StrategyConfig:
    angle_samples: int = 9
    candidate_limit: int = 24
    planning_seconds: float = .8
    max_localization_measurements: int = 8
    max_actions: int = 10000
    opportunistic: bool = True
    opportunistic_limit: int = 4
    runtime_limit: float = 1200.0

    def __post_init__(self):
        if (self.angle_samples < 3 or self.candidate_limit < 1 or self.planning_seconds < 0
                or self.max_localization_measurements < 0 or self.max_actions < 1
                or self.opportunistic_limit < 0 or not 0 < self.runtime_limit <= 1200):
            raise ValueError("第三问策略参数不合法")


@dataclass
class ChannelState:
    channel: int
    status: str = "unknown"
    region: TargetRegion = field(default_factory=TargetRegion)
    negative_sites: list = field(default_factory=list)
    measured_sites: list = field(default_factory=list)
    absence_certificate: dict | None = None
    localization_count: int = 0


def observation_plan(channel: ChannelState, current, config: StrategyConfig):
    """有限候选、观测角采样预测。安全性独立验证；评分不称连续最坏值。"""
    started = time.monotonic()
    vertices = channel.region.vertices
    direct_bound = finish_bound(vertices, current)
    if channel.localization_count >= config.max_localization_measurements or config.planning_seconds == 0:
        return {"动作": "覆盖清除", "预计费用_秒": direct_bound, "原因": "有限测向预算回退"}
    center, axes = principal_axes(vertices)
    candidates = []
    # 第二问候选只作生成，仍对当前全部历史外包重新检查可靠接收。
    positives = [o for o in channel.region.observations if o["结果"] == "direction"]
    if positives:
        first = positives[0]
        physical = build_physical_region(first["位置"], first["示向度"], DELTA)
        # 首次角楔可与目标圆相交，而中心射线恰好落在圆外；此时跳过透镜候选。
        if centerline_interval(physical) is not None:
            for p in analytic_candidates(physical):
                candidates.extend([center+.8*(p-center), center+.9*(p-center)])
    for angle in np.arange(8)*np.pi/4:
        direction = axes@np.array([math.cos(angle), math.sin(angle)])
        candidates.extend(center+distance*direction for distance in [30, 75, 150, 300, 500, 650])
    candidates.extend(search_stations())
    feasible = []
    for point in candidates:
        point = np.asarray(point)
        if (max_distance(vertices, point) <= 999.5 and min_distance(vertices, point) > 5.5
                and all(np.linalg.norm(point-old) > .01 for old in channel.measured_sites)):
            if all(np.linalg.norm(point-old) > .01 for old in feasible): feasible.append(point)
    # 排序只决定有限预算内考察次序，不作为最优性证明。
    feasible.sort(key=lambda p: np.linalg.norm(p-current))
    best = {"动作": "覆盖清除", "预计费用_秒": direct_bound, "原因": "当前有限覆盖费用更低"}
    evaluated = 0
    for point in feasible[:config.candidate_limit]:
        if time.monotonic()-started >= config.planning_seconds: break
        low, high = angular_interval(vertices, point)
        delta = math.radians(DELTA)
        predictions = []
        for theta in np.linspace(low-delta, high+delta, config.angle_samples):
            branch = add_wedge(vertices, point, math.degrees(theta))
            if len(branch): predictions.append(finish_bound(branch, point))
        if not predictions: continue
        evaluated += 1
        # 同目标频道通常已在当前，频道切换由外层执行器精确计入总时钟。
        score = float(np.linalg.norm(point-current)/5+5+max(predictions))
        if score < best["预计费用_秒"]:
            best = {"动作": "追加测向", "位置": point.tolist(), "预计费用_秒": score,
                    "最远可能目标距离_米": max_distance(vertices, point),
                    "最近可能目标距离_米": min_distance(vertices, point),
                    "后续费用采样最大值_秒": max(predictions)}
    return {**best, "已评估候选数": evaluated, "预测角节点": config.angle_samples,
            "规划耗时_秒": time.monotonic()-started, "评分性质": "有限角采样预测；非连续极值保证"}


class SchemeOne:
    def __init__(self, client: RobotClient, config: StrategyConfig | None = None):
        self.client = client
        self.config = config or StrategyConfig()
        self.channels = {c: ChannelState(c) for c in range(1, 21)}
        self.events = []
        self.completed_search_stations = []
        self.actions = 0
        self.error = None
        self.started = 0.0

    def _log(self, event):
        event = {"步骤": len(self.events)+1, "虚拟时间_秒": self.client.state.virtual_time, **event}
        self.events.append(event)
        self.client.record({"类型": "策略", **event})

    def _guard(self):
        self.client.check_budget()
        if self.actions >= self.config.max_actions: raise BudgetExceeded("达到动作次数上限")

    def _clear(self, state, point, reason, *, certified):
        self._guard()
        if state.status != "found": raise RuntimeError("禁止清除未确认存在或已完成频道")
        distance_bound = max_distance(state.region.vertices, point)
        if certified and distance_bound > 20-1e-4: raise RuntimeError("清除点未通过连续外包证书")
        result = self.client.clear(point, state.channel)
        self.actions += 1
        self._log({"动作": "清除", "用途": reason, "频道": state.channel, "位置": list(map(float, point)),
                   "结果": result["clear_result"], "具有覆盖证书": certified,
                   "最远可能目标距离_米": distance_bound})
        if result["clear_result"] == "success":
            state.status = "cleared"
        else:
            state.region.failed_clear(point)
            if certified: raise RuntimeError("具有覆盖证书的清除失败，请核对物理与误差模型")

    def _measure(self, state, point, reason):
        self._guard()
        point = np.asarray(point, dtype=float)
        result = self.client.measure(point, state.channel)
        self.actions += 1
        state.measured_sites.append(point.tolist())
        code = result["measure_result"]
        state.region.observe(point, code, result.get("svd_deg"))
        if code == "no_signal":
            state.negative_sites.append(point.tolist())
            if state.status == "unknown":
                certificate = coverage_certificate(state.negative_sites)
                if certificate is not None:
                    state.status, state.absence_certificate = "absent", certificate
        else:
            state.status = "found"
        event = {"动作": "检测", "用途": reason, "频道": state.channel, "位置": point.tolist(), "结果": code,
                 "示向度": result.get("svd_deg"), "频道状态": state.status}
        if code == "direction":
            _, radius = minimum_circle(state.region.vertices)
            event.update({"外包顶点": state.region.vertices.tolist(), "外包覆盖圆半径_米": radius})
        self._log(event)
        if code == "near": self._clear(state, point, "near原地清除", certified=True)
        return code

    def _opportunistic(self, point, primary):
        if not self.config.opportunistic: return
        eligible = []
        for state in self.channels.values():
            if state.channel == primary or state.status not in {"unknown", "found"}: continue
            if state.measured_sites and min(np.linalg.norm(np.asarray(p)-point) for p in state.measured_sites) < 600: continue
            if state.status == "found" and max_distance(state.region.vertices, point) > 999.5: continue
            eligible.append(state)
        eligible.sort(key=lambda s: (s.status != "found", len(s.measured_sites), s.channel))
        for state in eligible[:self.config.opportunistic_limit]:
            self._measure(state, point, "顺路检测")
            if self.complete(): break

    def _service_known(self):
        while not self.complete():
            known = [s for s in self.channels.values() if s.status == "found"]
            if not known: return
            current = np.asarray(self.client.state.position)
            state = min(known, key=lambda s: finish_bound(s.region.vertices, current))
            safe = clear_point(state.region.vertices, current)
            if safe is not None:
                self._clear(state, safe, "20米覆盖证书清除", certified=True)
                continue
            plan = observation_plan(state, current, self.config)
            # 对跨目标切换计入预测；仅影响测向与直接清除的选择，清除不切频道。
            if plan["动作"] == "追加测向" and state.channel != self.client.state.channel:
                plan["预计费用_秒"] += 1
                if plan["预计费用_秒"] >= finish_bound(state.region.vertices, current):
                    plan = {"动作": "覆盖清除", "预计费用_秒": finish_bound(state.region.vertices, current),
                            "原因": "计入频道切换后有限覆盖费用更低"}
            self._log({"动作": "规划", "频道": state.channel, "方案": plan})
            if plan["动作"] == "追加测向":
                state.localization_count += 1
                point = np.asarray(plan["位置"])
                result = self._measure(state, point, "追加测向")
                if result == "no_signal": raise RuntimeError("可靠接收候选出现无信号，停止核对模型")
                self._opportunistic(point, state.channel)
            else:
                path = sweep_path(state.region.vertices, current)
                self._log({"动作": "有限覆盖", "频道": state.channel, "覆盖点数": len(path), "单元边长_米": 20})
                for point in path:
                    self._clear(state, point, "有限方格覆盖", certified=False)
                    if state.status == "cleared": break
                if state.status != "cleared": raise RuntimeError("有限覆盖全部完成仍未清除，模型或协议异常")

    def complete(self):
        return (sum(s.status == "cleared" for s in self.channels.values()) == 16
                or all(s.status in {"cleared", "absent"} for s in self.channels.values()))

    def run(self):
        self.started = time.monotonic()
        try:
            self.client.enter()
            self.client.state.deadline = min(self.client.state.deadline, self.client.clock()+self.config.runtime_limit)
            for index, point in enumerate(search_stations()):
                if self.complete(): break
                pending = [s for s in self.channels.values() if s.status in {"unknown", "found"}]
                pending.sort(key=lambda s: (s.channel != self.client.state.channel, s.channel))
                for state in pending:
                    if state.status not in {"unknown", "found"}: continue
                    if any(np.linalg.norm(point-p) < 1e-6 for p in state.measured_sites): continue
                    self._measure(state, point, f"搜索站{index}")
                    if self.complete(): break
                self.completed_search_stations.append(index)
                # 每站的已发现目标有限处理完毕，再推进搜索，避免未发现频道长期搁置。
                self._service_known()
            self._service_known()
            if not self.complete(): raise RuntimeError("七站流程结束但仍缺少完成证据")
        except (BudgetExceeded, ProtocolError, RuntimeError, ValueError) as error:
            self.error = f"{type(error).__name__}: {error}"
            self._log({"动作": "异常停止", "说明": self.error})
        finally:
            if self.client.state.entered and not self.client.state.exited and self.client.pending is None:
                try: self.client.exit()
                except (BudgetExceeded, ProtocolError) as error:
                    self.error = (self.error+"；" if self.error else "")+f"退出未确认：{error}"
        state = self.client.state
        cleared = [c for c, s in self.channels.items() if s.status == "cleared"]
        result = {"方案": "方案一：七站搜索与自适应定位清除", "配置": asdict(self.config),
                  "运行成功": self.complete() and state.exited and self.error is None,
                  "全部完成证据": self.complete(), "正常退出": state.exited, "异常": self.error,
                  "清除数": len(cleared), "已清除频道": cleared,
                  "已证明不存在频道": [c for c, s in self.channels.items() if s.status == "absent"],
                  "终止依据": "达到16个目标上界" if len(cleared) == 16 else "逐频道清除或覆盖排除" if self.complete() else "未完成",
                  "虚拟总时间_秒": state.virtual_time, "平均定位清除时间_秒": state.virtual_time/len(cleared) if cleared else None,
                  "程序运行时间_秒": time.monotonic()-self.started,
                  "总路程_米": state.distance, "检测次数": state.measurements, "频道切换次数": state.switches,
                  "清除成功次数": state.clear_success, "清除失败次数": state.clear_failure,
                  "时间分项_秒": state.time_parts(), "已完成搜索站": self.completed_search_stations,
                  "频道记录": [{"频道": c, "状态": s.status, "测向记录": s.region.observations,
                              "排除约束": s.region.exclusions, "不存在证书": s.absence_certificate,
                              "追加测向次数": s.localization_count} for c, s in self.channels.items()],
                  "动作记录": self.events}
        return result
