"""第三问串行协议客户端；所有定位策略只能通过此接口获取反馈。"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Protocol
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from uuid import uuid4


class ProtocolError(RuntimeError):
    """拒绝、格式异常或未决动作；不可将其当成无信号。"""


class BudgetExceeded(RuntimeError):
    """为正常退出预留时间，停止发起新检测或清除。"""


class Transport(Protocol):
    def post(self, path: str, payload: dict, timeout: float) -> tuple[int, dict]: ...


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTransport:
    """仅连接题目规定的本机接口；不自动跟随重定向或使用代理。"""
    def __init__(self, base_url: str = "http://127.0.0.1:2026"):
        url = urlsplit(base_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.path not in {"", "/"} or url.query or url.fragment or url.username or url.password):
            raise ValueError("接口必须是本机回环HTTP地址，例如 http://127.0.0.1:2026")
        self.base_url = base_url.rstrip("/")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def post(self, path: str, payload: dict, timeout: float) -> tuple[int, dict]:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        request = Request(self.base_url+path, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raw = error.read().decode("utf-8")
            try:
                result = json.loads(raw)
            except ValueError:
                result = {"accepted": False, "http_error_body": raw[:1000]}
            return error.code, result


def identifier(value: str, limit: int) -> str:
    if (not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= limit
            or any(unicodedata.category(c) in {"Cc", "Cf"} for c in value)):
        raise ValueError("标识符为空、过长或含控制/不可见格式字符")
    return value


def coordinate(value) -> tuple[float, float]:
    if len(value) != 2 or any(isinstance(v, bool) for v in value):
        raise ValueError("位置须包含两个有限数值")
    point = tuple(float(v) for v in value)
    if not all(math.isfinite(v) and abs(v) <= 2_000_000 for v in point):
        raise ValueError("坐标非有限或超出协议范围")
    return point


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"响应字段{name}不是有限数值")
    return float(value)


@dataclass
class RobotState:
    position: tuple[float, float] = (0.0, 0.0)
    channel: int = 1
    virtual_time: float = 0.0
    distance: float = 0.0
    measurements: int = 0
    switches: int = 0
    clear_success: int = 0
    clear_failure: int = 0
    entered: bool = False
    exited: bool = False
    deadline: float = math.inf
    virtual_limit: float = 360000.0

    def time_parts(self) -> dict:
        return {"移动": self.distance/5, "检测": self.measurements*5,
                "切换": self.switches, "清除成功": self.clear_success*5,
                "清除失败": self.clear_failure*3}


class RobotClient:
    def __init__(self, transport: Transport, robot_id: str, log_path: Path | None = None,
                 *, retries: int = 3, timeout: float = 3.0, exit_reserve: float = 5.0,
                 clock=time.monotonic, sleeper=time.sleep):
        self.transport = transport
        self.robot_id = identifier(robot_id, 64)
        if retries < 0 or timeout <= 0 or exit_reserve < 0:
            raise ValueError("重试次数、超时或退出预留参数不合法")
        self.retries, self.timeout, self.exit_reserve = retries, timeout, exit_reserve
        self.clock, self.sleeper = clock, sleeper
        self.state = RobotState()
        self.session_prefix = uuid4().hex
        self.sequence = 0
        self.pending: tuple[str, dict] | None = None
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("x", encoding="utf-8", newline="\n")

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None

    def record(self, event: dict):
        event = {"记录序号": len(self.records)+1, **event}
        self.records.append(event)
        if self._log is not None:
            self._log.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
            self._log.flush()

    def check_budget(self, position=None, action_cost=0.0, *, exiting=False):
        if self.state.entered:
            reserve = 0.0 if exiting else self.exit_reserve
            if self.clock() >= self.state.deadline-reserve:
                raise BudgetExceeded("实际剩余现实时间不足")
            travel = 0.0 if position is None else math.dist(self.state.position, position)/5
            if not exiting and self.state.virtual_time+travel+action_cost >= self.state.virtual_limit-.01:
                raise BudgetExceeded("下一动作可能达到虚拟时间上限")

    def _action(self, path: str, position=None, channel=None) -> dict:
        if not self._lock.acquire(blocking=False):
            raise ProtocolError("禁止并发发送不同动作")
        try:
            if self.pending is not None:
                raise ProtocolError("上一个动作结果未决，禁止使用新ID继续")
            if path not in {"/enter", "/measure", "/clear", "/exit"}:
                raise ValueError("未知动作")
            if self.state.exited or (path == "/enter" and self.state.entered) or (path != "/enter" and not self.state.entered):
                raise ProtocolError("当前客户端状态不允许该动作")
            point = None
            if path in {"/measure", "/clear"}:
                point = coordinate(position)
                if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
                    raise ValueError("频道必须是1至20的整数")
            cost = 5.0 + int(path == "/measure" and channel != self.state.channel)
            self.check_budget(point, cost if point is not None else 0, exiting=path == "/exit")
            self.sequence += 1
            payload = {"arena_id": "default", "robot_id": self.robot_id,
                       "request_id": f"{self.session_prefix}-{self.sequence}"}
            if point is not None:
                payload.update(position={"x": point[0], "y": point[1]}, channel=channel)
            self.pending = (path, payload)
            started = self.clock()
            for attempt in range(self.retries+1):
                self.record({"类型": "请求", "path": path, "payload": payload, "尝试": attempt+1})
                remaining = self.state.deadline-self.clock() if self.state.entered else self.timeout
                if remaining <= 0:
                    raise ProtocolError("动作未决时达到现实截止")
                try:
                    status, response = self.transport.post(path, payload, min(self.timeout, remaining))
                    self.record({"类型": "响应", "request_id": payload["request_id"], "HTTP": status, "response": response})
                    if status >= 500:
                        raise OSError(f"HTTP {status}，执行状态待确认")
                except (OSError, URLError, TimeoutError, ValueError) as error:
                    self.record({"类型": "通信异常", "request_id": payload["request_id"], "说明": str(error), "尝试": attempt+1})
                    if attempt == self.retries:
                        raise ProtocolError("相同ID重试耗尽，保留未决动作并停止") from error
                    self.sleeper(min(.05*2**attempt, .2))
                    continue
                if status != 200 or not isinstance(response, dict) or response.get("accepted") is not True:
                    self.pending = None  # 明确拒绝：不更新位置、频道或时钟。
                    raise ProtocolError(f"请求被拒绝：HTTP {status}，accepted={response.get('accepted') if isinstance(response, dict) else None}")
                self._accept(path, response, point, channel, started)
                self.pending = None
                return response
            raise AssertionError("不可达")
        finally:
            self._lock.release()

    def _accept(self, path, response, point, channel, started):
        now = finite_number(response.get("virtual_time_s"), "virtual_time_s")
        if now < self.state.virtual_time-1e-6:
            raise ProtocolError("已接受响应的虚拟时间倒退，停止以核对协议")
        if path == "/enter":
            remaining = finite_number(response.get("remaining_real_duration_s"), "remaining_real_duration_s")
            limit = finite_number(response.get("max_virtual_duration_s"), "max_virtual_duration_s")
            if not 0 <= remaining <= 1200 or limit <= 0 or abs(now) > 1e-6:
                raise ProtocolError("进入响应的时间范围异常")
            self.state.entered = True
            self.state.deadline = started+remaining  # 将重试及网络延迟也计入预算。
            self.state.virtual_limit = limit
        elif path == "/exit":
            if response.get("exit_reason") != "user_exit":
                raise ProtocolError("退出响应原因异常")
            self.state.exited = True
        else:
            result_key = "measure_result" if path == "/measure" else "clear_result"
            result = response.get(result_key)
            allowed = {"direction", "near", "no_signal"} if path == "/measure" else {"success", "no_target_in_range"}
            if result not in allowed:
                raise ProtocolError("已接受响应缺少合法动作结果")
            if result == "direction":
                angle = finite_number(response.get("svd_deg"), "svd_deg")
                if not 0 <= angle < 360:
                    raise ProtocolError("示向度超出[0,360)")
            distance = math.dist(self.state.position, point)
            switch = int(path == "/measure" and channel != self.state.channel)
            action_time = 3 if result == "no_target_in_range" else 5
            expected = self.state.virtual_time+distance/5+switch+action_time
            if abs(now-expected) > .001:
                raise ProtocolError(f"响应计时与附件规则不符：预期{expected}，收到{now}")
            self.state.distance += distance
            self.state.position = point
            if path == "/measure":
                self.state.measurements += 1
                self.state.switches += switch
                self.state.channel = channel
            elif result == "success":
                self.state.clear_success += 1
            else:
                self.state.clear_failure += 1
        self.state.virtual_time = now

    def enter(self): return self._action("/enter")
    def measure(self, position, channel): return self._action("/measure", position, channel)
    def clear(self, position, channel): return self._action("/clear", position, channel)
    def exit(self): return self._action("/exit")
