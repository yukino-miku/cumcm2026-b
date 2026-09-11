"""方案二改进：已有双站观测且能安全清除的频道，停止后续固定扫描。

保留原SchemeTwo作为关闭开关时的对照。localized是扫描层的状态；基础
频道仍为found，确保原有清除执行器继续要求实际清除成功才能完成任务。
"""
from dataclasses import asdict, dataclass
import time
import numpy as np

from .q3_geometry import max_distance, minimum_circle
from .q3_protocol import BudgetExceeded, ProtocolError
from .q3_scheme2 import SchemeTwo, SchemeTwoConfig


@dataclass
class SchemeTwoLocalizedConfig(SchemeTwoConfig):
    skip_localized: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.skip_localized, bool):
            raise ValueError("skip_localized须为布尔值")


class SchemeTwoLocalized(SchemeTwo):
    station_hook_enabled = False
    def __init__(self, client, config=None):
        super().__init__(client, config or SchemeTwoLocalizedConfig())
        self.localized = {}
        self.skipped_measurements = 0
        self.scanning = False
        self.scan_stop_reason = None

    def scan_status(self, state):
        if state.status == "found" and state.channel in self.localized:
            return "localized"
        return state.status

    def _mark_localized(self, state):
        if (not self.config.skip_localized or state.status != "found"
                or state.channel in self.localized):
            return
        sites = sorted({tuple(o["位置"]) for o in state.region.observations if o["结果"] == "direction"})
        if len(sites) < 2:
            return
        center, radius = minimum_circle(state.region.vertices)
        if radius > 19.5:
            return
        certificate = {"频道": state.channel, "确认步骤": len(self.events) + 1,
                       "不同方向站数": len(sites), "方向站点": [list(p) for p in sites],
                       "覆盖圆心": center.tolist(), "外包覆盖圆半径_米": radius,
                       "外包顶点": state.region.vertices.tolist()}
        self.localized[state.channel] = certificate
        self._log({"动作": "定位完成", "扫描状态": "localized", **certificate})

    def _measure(self, state, point, reason):
        result = super()._measure(state, point, reason)
        if self.scanning:
            self._mark_localized(state)
        return result

    def _finish_scan(self):
        channels = []
        for state in self.channels.values():
            sites = {tuple(o["位置"]) for o in state.region.observations if o["结果"] == "direction"}
            row = {"频道": state.channel, "状态": state.status,
                   "扫描状态": self.scan_status(state), "不同方向站数": len(sites)}
            if state.status == "unknown":
                raise RuntimeError("固定扫描结束仍有未知频道，缺少覆盖排除证据")
            if state.status == "found":
                if len(sites) < 2:
                    raise RuntimeError("未清除目标不足两个不同方向站")
                _, radius = minimum_circle(state.region.vertices)
                row.update({"外包覆盖圆半径_米": radius, "可安全单点清除": radius <= 19.5,
                            "外包顶点": state.region.vertices.tolist()})
                if self.scan_status(state) == "localized":
                    cert = self.localized[state.channel]
                    if max_distance(state.region.vertices, cert["覆盖圆心"]) > 19.5:
                        raise RuntimeError("已定位待清除频道的保留证书失效")
            channels.append(row)
        robot = self.client.state
        self.scan_summary = {"实际虚拟时间_秒": robot.virtual_time, "实际路程_米": robot.distance,
            "检测次数": robot.measurements, "切换次数": robot.switches, "清除数": robot.clear_success,
            "已完成固定站": list(self.completed_search_stations), "频道快照": channels,
            "待清除数": sum(s.status == "found" for s in self.channels.values()),
            "需继续缩小外包数": sum(s.get("可安全单点清除") is False for s in channels),
            "已定位待清除数": sum(s["扫描状态"] == "localized" for s in channels),
            "跳过已定位频道检测次数": self.skipped_measurements,
            "扫描结束原因": self.scan_stop_reason or "完成固定站序列"}
        self._log({"动作": "固定扫描完成", "扫描摘要": self.scan_summary})

    def _after_fixed_station(self, index, point):
        """扩展点：完成本站检测后，可由路线调度器安排有限的顺路服务。"""

    def run(self):
        if not self.config.skip_localized and not self.station_hook_enabled:
            # 原版的几何、调度与事件顺序原样保留，便于同场景单项对照。
            result = super().run()
            result.update({"停止已定位频道扫描": False, "跳过已定位频道检测次数": 0, "定位完成证书": []})
            return result
        self.started = time.monotonic()
        try:
            self.client.enter()
            self.client.state.deadline = min(self.client.state.deadline, self.client.clock() + self.config.runtime_limit)
            self.scanning = True
            for index, point in enumerate(np.asarray(self.layout["站点"])):
                if self.complete():
                    break
                active = [s for s in self.channels.values() if s.status in {"unknown", "found"}]
                # 只有全部剩余目标已定位、其他频道已有排除证据，才能结束扫描。
                # 不为到达一个没有待测频道的站点额外制造检测或move请求。
                if all(self.scan_status(s) == "localized" for s in active):
                    self.scan_stop_reason = "剩余目标均已定位，其他频道均已清除或排除"
                    break
                self._log({"动作": "开始固定站", "站序": index, "位置": point.tolist()})
                active.sort(key=lambda s: (s.channel != self.client.state.channel, s.channel))
                for state in active:
                    if state.status not in {"unknown", "found"}:
                        continue
                    if self.scan_status(state) == "localized":
                        self._guard()
                        self.skipped_measurements += 1
                        self._log({"动作": "跳过固定检测", "频道": state.channel, "站序": index,
                                   "计划站位置": point.tolist(), "原因": "已具双站与19.5米清除证书",
                                   "证书步骤": self.localized[state.channel]["确认步骤"]})
                        continue
                    self._measure(state, point, f"固定扫描站{index}")
                    # 只有实际检测到达该站后，才尝试不绕行的原地清除。
                    self._station_clear(point)
                    if self.complete():
                        break
                self.completed_search_stations.append(index)
                if not self.complete():
                    self._after_fixed_station(index, point)
            self.scanning = False
            if sum(s.status == "cleared" for s in self.channels.values()) == 16:
                self.scan_summary = {"提前结束": "已清除16个目标", "实际虚拟时间_秒": self.client.state.virtual_time,
                    "已完成固定站": list(self.completed_search_stations), "清除数": 16,
                    "跳过已定位频道检测次数": self.skipped_measurements}
                self._log({"动作": "固定扫描完成", "扫描摘要": self.scan_summary})
            else:
                self._finish_scan()
            self._service_known()
            if not self.complete():
                raise RuntimeError("停止扫描后的定位清除流程仍缺少全部完成证据")
        except (BudgetExceeded, ProtocolError, RuntimeError, ValueError) as error:
            self.error = f"{type(error).__name__}: {error}"
            self._log({"动作": "异常停止", "说明": self.error})
        finally:
            self.scanning = False
            if self.client.state.entered and not self.client.state.exited and self.client.pending is None:
                try:
                    self.client.exit()
                except (BudgetExceeded, ProtocolError) as error:
                    self.error = (self.error + "；" if self.error else "") + f"退出未确认：{error}"
        robot = self.client.state
        cleared = [c for c, s in self.channels.items() if s.status == "cleared"]
        description = "方案二：固定双覆盖扫描"
        if self.config.skip_localized:
            description += "，已定位频道停止检测"
        return {"方案": description + "，后续精定位清除", "配置": asdict(self.config),
            "布局": self.layout, "停止已定位频道扫描": self.config.skip_localized,
            "运行成功": self.complete() and robot.exited and self.error is None,
            "全部完成证据": self.complete(), "正常退出": robot.exited, "异常": self.error,
            "清除数": len(cleared), "已清除频道": cleared,
            "已证明不存在频道": [c for c, s in self.channels.items() if s.status == "absent"],
            "终止依据": "达到16个目标上界" if len(cleared) == 16 else "逐频道清除或覆盖排除" if self.complete() else "未完成",
            "虚拟总时间_秒": robot.virtual_time, "平均定位清除时间_秒": robot.virtual_time / len(cleared) if cleared else None,
            "程序运行时间_秒": time.monotonic() - self.started,
            "总路程_米": robot.distance, "检测次数": robot.measurements, "频道切换次数": robot.switches,
            "清除成功次数": robot.clear_success, "清除失败次数": robot.clear_failure, "时间分项_秒": robot.time_parts(),
            "已完成搜索站": self.completed_search_stations, "扫描阶段": self.scan_summary,
            "跳过已定位频道检测次数": self.skipped_measurements, "定位完成证书": list(self.localized.values()),
            "频道记录": [{"频道": c, "状态": s.status, "扫描状态": self.scan_status(s),
                           "测向记录": s.region.observations, "排除约束": s.region.exclusions,
                           "不存在证书": s.absence_certificate, "追加测向次数": s.localization_count}
                          for c, s in self.channels.items()], "动作记录": self.events}
