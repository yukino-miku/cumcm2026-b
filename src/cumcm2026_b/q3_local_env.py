"""可复现的本地构造环境；并非官方模拟器，不向策略提供目标真值。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
import numpy as np


@dataclass(frozen=True)
class Source:
    channel: int
    position: tuple[float, float]
    radius: float = 1000.0


class LocalEnvironment:
    def __init__(self, sources, *, robot_id="local-robot", seed=0, error_mode="hash", remaining=1200):
        self._sources = {s.channel: s for s in sources}
        if len(self._sources) != len(sources): raise ValueError("频道重复")
        for s in sources:
            if not 1 <= s.channel <= 20 or math.hypot(*s.position) > 1800+1e-6 or not 1000 <= s.radius <= 1500:
                raise ValueError("构造源参数超出第三问范围")
        if error_mode not in {"hash", "zero", "plus", "minus", "alternating"}: raise ValueError("未知误差模式")
        self.robot_id, self.seed, self.error_mode, self.remaining = robot_id, seed, error_mode, remaining
        self.position, self.channel, self.virtual = (0., 0.), 1, 0.
        self.started = self.ended = False
        self.cleared = set()
        self.cache = {}
        self.history = []

    def _error(self, channel, position):
        if self.error_mode == "zero": return 0.
        if self.error_mode == "plus": return 1.
        if self.error_mode == "minus": return -1.
        if self.error_mode == "alternating": return 1. if channel % 2 else -1.
        raw = json.dumps([self.seed, channel, list(position)], separators=(",", ":")).encode()
        return 2*int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")/(2**64-1)-1

    def post(self, path, payload, timeout=3):
        rejected = {"accepted": False, "real_timestamp_ms": int(time.time()*1000), "virtual_time_s": 0}
        expected = {"arena_id", "robot_id", "request_id"} | ({"position", "channel"} if path in {"/measure", "/clear"} else set())
        if path not in {"/enter", "/measure", "/clear", "/exit"}: return 404, rejected
        if set(payload) != expected or payload.get("arena_id") != "default" or payload.get("robot_id") != self.robot_id:
            return 200, rejected
        key = payload["request_id"]
        signature = json.dumps([path, payload], sort_keys=True, allow_nan=False)
        if key in self.cache:
            saved_sig, response = self.cache[key]
            return (200, dict(response)) if saved_sig == signature else (409, rejected)
        if self.ended or (path == "/enter" and self.started) or (path != "/enter" and not self.started): return 200, rejected
        response = {"accepted": True, "real_timestamp_ms": int(time.time()*1000)}
        if path == "/enter":
            self.started = True
            response.update(max_virtual_duration_s=360000, max_real_duration_s=1200, remaining_real_duration_s=self.remaining)
        elif path == "/exit":
            self.ended = True
            response["exit_reason"] = "user_exit"
        else:
            if set(payload["position"]) != {"x", "y"}: return 200, rejected
            point = tuple(payload["position"][k] for k in ("x", "y"))
            channel = payload["channel"]
            if (isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= 2e6 for v in point)):
                return 400, rejected
            self.virtual += round(math.dist(self.position, point)/5, 6)
            self.position = point
            source = self._sources.get(channel) if channel not in self.cleared else None
            distance = math.dist(point, source.position) if source is not None else math.inf
            if path == "/measure":
                self.virtual += 5+int(channel != self.channel)
                self.channel = channel
                if source is None or distance > source.radius:
                    response["measure_result"] = "no_signal"
                elif distance <= 5:
                    response["measure_result"] = "near"
                else:
                    bearing = math.degrees(math.atan2(source.position[1]-point[1], source.position[0]-point[0]))
                    response.update(measure_result="direction", svd_deg=round((bearing+self._error(channel, point)) % 360, 2) % 360)
            else:
                if distance <= 20:
                    self.cleared.add(channel)
                    self.virtual += 5
                    response["clear_result"] = "success"
                else:
                    self.virtual += 3
                    response["clear_result"] = "no_target_in_range"
        response["virtual_time_s"] = round(self.virtual, 6)
        self.cache[key] = signature, dict(response)
        self.history.append({"path": path, "payload": json.loads(json.dumps(payload)), "response": dict(response)})
        return 200, response

    def truth_for_evaluation(self):
        """仅供运行结束后的评估器使用；策略不调用此方法。"""
        return {"目标": [{"频道": s.channel, "位置": list(s.position), "接收半径": s.radius} for s in self._sources.values()],
                "目标数": len(self._sources), "清除数": len(self.cleared), "全部清除": len(self.cleared) == len(self._sources)}


def make_case(seed: int, count: int, layout="uniform", radius_mode="min", error_mode="hash"):
    rng = np.random.default_rng(seed)
    channels = rng.choice(np.arange(1, 21), count, replace=False)
    if layout == "boundary":
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)+rng.uniform(0, .1)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))*rng.uniform(1790, 1800, (count, 1))
    elif layout == "cluster":
        positions = np.clip(rng.normal([1150, -350], [40, 40], (count, 2)), -1250, 1250)
    elif layout == "center":
        angles = rng.uniform(0, 2*np.pi, count)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))*rng.uniform(6, 180, (count, 1))
        positions[0] = [0, 0]
    elif layout == "uniform":
        angles = rng.uniform(0, 2*np.pi, count)
        positions = np.column_stack((np.cos(angles), np.sin(angles)))*1800*np.sqrt(rng.uniform(size=(count, 1)))
    else: raise ValueError("未知构造布局")
    radii = np.full(count, 1000.0 if radius_mode == "min" else 1500.0) if radius_mode != "mixed" else rng.uniform(1000, 1500, count)
    sources = [Source(int(c), tuple(p), float(r)) for c, p, r in zip(channels, positions, radii)]
    return LocalEnvironment(sources, seed=seed, error_mode=error_mode)
