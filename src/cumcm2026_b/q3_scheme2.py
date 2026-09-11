"""方案二：固定双覆盖扫描结束后，集中精定位清除；复用已验证的动作层。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
import numpy as np

from .q3_geometry import max_distance, minimum_circle
from .q3_protocol import BudgetExceeded, ProtocolError
from .q3_strategy import SchemeOne, StrategyConfig


LAYOUT_NAMES = {"main14": "14站主布局", "margin14": "14站10米余量布局", "compact15": "15站小余量布局"}


def layout_spec(name="main14"):
    """由连续分区证明生成坐标及覆盖上界，不用网格采样充当证明。"""
    if name not in LAYOUT_NAMES:
        raise ValueError("未知方案二布局")
    if name == "compact15":
        n = 14
        rho = 900/math.cos(2*math.pi/n)
        inner = None
        split = 0.0
    else:
        n = 12
        rho = 1150.0 if name == "main14" else 1800*math.cos(math.pi/6)-math.sqrt(990**2-900**2)
        inner = 500.0 if name == "main14" else 400.0
        split = 400.0 if name == "main14" else 500.0
    ring = rho*np.column_stack((np.cos(np.arange(n)*2*np.pi/n), np.sin(np.arange(n)*2*np.pi/n)))
    stations = np.vstack(([[0., 0.]] if inner is None else [[0., 0.], [inner, 0.]], ring))
    def edge_bound(r):
        return math.sqrt(r*r+rho*rho-2*r*rho*math.cos(2*math.pi/n))
    bound = max(edge_bound(split), edge_bound(1800), 0 if inner is None else split+inner)
    if not bound < 1000:
        raise ValueError("布局未通过全域连续双重覆盖上界")
    length = float(np.linalg.norm(np.diff(stations, axis=0), axis=1).sum())
    return {"标识": name, "名称": LAYOUT_NAMES[name], "站点": stations.tolist(), "外围半径_米": rho,
            "站数": len(stations), "中央分区半径_米": split, "第二近距离连续上界_米": bound,
            "保证接收余量_米": 1000-bound, "全频道固定扫描路程_米": length,
            "全频道固定扫描时间_秒": length/5+len(stations)*(20*5+19)}


@dataclass
class SchemeTwoConfig(StrategyConfig):
    layout: str = "main14"
    scan_clear: bool = True

    def __post_init__(self):
        super().__post_init__()
        layout_spec(self.layout)
        if not isinstance(self.scan_clear, bool):
            raise ValueError("scan_clear须为布尔值")


class SchemeTwo(SchemeOne):
    """仅复用单源服务/动作/几何；固定扫描调度与验收独立实现。"""
    def __init__(self, client, config=None):
        super().__init__(client, config or SchemeTwoConfig())
        self.layout = layout_spec(self.config.layout)
        self.scan_summary = None

    def _measure(self, state, point, reason):
        code = super()._measure(state, point, reason)
        # 四叉树只能给充分证书；全布局无信号时还可直接使用连续覆盖证明。
        if state.status == "unknown" and code == "no_signal":
            negatives = np.asarray(state.negative_sites)
            if all(np.linalg.norm(negatives-p, axis=1).min() <= 1e-6 for p in np.asarray(self.layout["站点"])):
                state.status = "absent"
                state.absence_certificate = {"方式": "完整固定布局连续双覆盖", "布局": self.config.layout,
                    "第二近距离连续上界_米": self.layout["第二近距离连续上界_米"]}
                self._log({"动作": "不存在证书", "频道": state.channel, "证书": state.absence_certificate})
        return code

    def _station_clear(self, point):
        if not self.config.scan_clear:
            return
        for state in self.channels.values():
            if state.status == "found" and max_distance(state.region.vertices, point) <= 19.5:
                self._clear(state, point, "固定扫描站原地证书清除", certified=True)

    def _finish_scan(self):
        channels = []
        for state in self.channels.values():
            sites = {tuple(o["位置"]) for o in state.region.observations if o["结果"] == "direction"}
            row = {"频道": state.channel, "状态": state.status, "不同方向站数": len(sites)}
            if state.status == "unknown":
                raise RuntimeError("固定扫描结束仍有未知频道，缺少覆盖排除证据")
            if state.status == "found":
                if len(sites) < 2:
                    raise RuntimeError("固定双覆盖扫描结束，未清除目标不足两个不同方向站")
                _, radius = minimum_circle(state.region.vertices)
                row.update({"外包覆盖圆半径_米": radius, "可安全单点清除": radius <= 19.5,
                            "外包顶点": state.region.vertices.tolist()})
            channels.append(row)
        state = self.client.state
        self.scan_summary = {"实际虚拟时间_秒": state.virtual_time, "实际路程_米": state.distance,
            "检测次数": state.measurements, "切换次数": state.switches, "清除数": state.clear_success,
            "已完成固定站": list(self.completed_search_stations), "频道快照": channels,
            "待清除数": sum(s.status == "found" for s in self.channels.values()),
            "需继续缩小外包数": sum(s.get("可安全单点清除") is False for s in channels)}
        self._log({"动作": "固定扫描完成", "扫描摘要": self.scan_summary})

    def run(self):
        self.started = time.monotonic()
        try:
            self.client.enter()
            self.client.state.deadline = min(self.client.state.deadline, self.client.clock()+self.config.runtime_limit)
            for index, point in enumerate(np.asarray(self.layout["站点"])):
                if self.complete(): break
                self._log({"动作": "开始固定站", "站序": index, "位置": point.tolist()})
                pending = [s for s in self.channels.values() if s.status in {"unknown", "found"}]
                pending.sort(key=lambda s: (s.channel != self.client.state.channel, s.channel))
                for state in pending:
                    if state.status not in {"unknown", "found"}: continue
                    self._measure(state, point, f"固定扫描站{index}")
                    self._station_clear(point)
                    if self.complete(): break
                self.completed_search_stations.append(index)
            # 已清除16个时可合法提前终止，未知的其余频道无需强作不存在结论。
            if sum(s.status == "cleared" for s in self.channels.values()) == 16:
                self.scan_summary = {"提前结束": "已清除16个目标", "实际虚拟时间_秒": self.client.state.virtual_time,
                    "已完成固定站": list(self.completed_search_stations), "清除数": 16}
                self._log({"动作": "固定扫描完成", "扫描摘要": self.scan_summary})
            else:
                self._finish_scan()
            self._service_known()
            if not self.complete(): raise RuntimeError("双覆盖与精定位流程结束但缺少全部完成证据")
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
        return {"方案": "方案二：固定双覆盖扫描与后续精定位清除", "配置": asdict(self.config), "布局": self.layout,
            "运行成功": self.complete() and state.exited and self.error is None,
            "全部完成证据": self.complete(), "正常退出": state.exited, "异常": self.error,
            "清除数": len(cleared), "已清除频道": cleared,
            "已证明不存在频道": [c for c, s in self.channels.items() if s.status == "absent"],
            "终止依据": "达到16个目标上界" if len(cleared) == 16 else "逐频道清除或覆盖排除" if self.complete() else "未完成",
            "虚拟总时间_秒": state.virtual_time, "平均定位清除时间_秒": state.virtual_time/len(cleared) if cleared else None,
            "程序运行时间_秒": time.monotonic()-self.started, "总路程_米": state.distance,
            "检测次数": state.measurements, "频道切换次数": state.switches,
            "清除成功次数": state.clear_success, "清除失败次数": state.clear_failure,
            "时间分项_秒": state.time_parts(), "已完成搜索站": self.completed_search_stations,
            "扫描阶段": self.scan_summary,
            "频道记录": [{"频道": c, "状态": s.status, "测向记录": s.region.observations,
                "排除约束": s.region.exclusions, "不存在证书": s.absence_certificate,
                "追加测向次数": s.localization_count} for c, s in self.channels.items()],
            "动作记录": self.events}
