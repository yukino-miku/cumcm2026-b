"""第四问本地构造环境：固定半径的全向/180度定向源；非官方模拟器。"""
from dataclasses import dataclass
import json
import math
import time
import numpy as np

from .q3_local_env import LocalEnvironment as OmniEnvironment, Source as OmniSource


@dataclass(frozen=True)
class Source(OmniSource):
    axis_deg: float | None = None  # None表示全向，角度从正东逆时针计算。


def visible(source, point):
    if math.dist(source.position, point) > source.radius:
        return False
    if source.axis_deg is None:
        return True
    angle = math.radians(source.axis_deg)
    dot = (point[0]-source.position[0])*math.cos(angle) + (point[1]-source.position[1])*math.sin(angle)
    return dot >= -1e-9  # 只对浮点计算的半平面闭边界作微小容差。


class LocalEnvironment(OmniEnvironment):
    def __init__(self, sources, **kwargs):
        sources = list(sources)
        for s in sources:
            if (isinstance(s.channel, bool) or not isinstance(s.channel, int)
                    or len(s.position) != 2 or not all(math.isfinite(v) for v in s.position)
                    or not math.isfinite(s.radius)
                    or (s.axis_deg is not None and (isinstance(s.axis_deg, bool) or not math.isfinite(s.axis_deg)))):
                raise ValueError('第四问构造源参数不合法')
        super().__init__(sources, **kwargs)

    def post(self, path, payload, timeout=3):
        # 协议和费用与第三问一致，独立实现物理反馈，保持第三问源码及历史结果不变。
        rejected = {'accepted': False, 'real_timestamp_ms': int(time.time()*1000), 'virtual_time_s': 0}
        expected = {'arena_id', 'robot_id', 'request_id'} | ({'position', 'channel'} if path in {'/measure', '/clear'} else set())
        if path not in {'/enter', '/measure', '/clear', '/exit'}:
            return 404, rejected
        if set(payload) != expected or payload.get('arena_id') != 'default' or payload.get('robot_id') != self.robot_id:
            return 200, rejected
        key = payload['request_id']
        signature = json.dumps([path, payload], sort_keys=True, allow_nan=False)
        if key in self.cache:
            saved_sig, response = self.cache[key]
            return (200, dict(response)) if saved_sig == signature else (409, rejected)
        if self.ended or (path == '/enter' and self.started) or (path != '/enter' and not self.started):
            return 200, rejected
        response = {'accepted': True, 'real_timestamp_ms': int(time.time()*1000)}
        if path == '/enter':
            self.started = True
            response.update(max_virtual_duration_s=360000, max_real_duration_s=1200, remaining_real_duration_s=self.remaining)
        elif path == '/exit':
            self.ended = True
            response['exit_reason'] = 'user_exit'
        else:
            if not isinstance(payload['position'], dict) or set(payload['position']) != {'x', 'y'}:
                return 200, rejected
            point = tuple(payload['position'][k] for k in ('x', 'y'))
            channel = payload['channel']
            if (isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= 2e6 for v in point)):
                return 400, rejected
            self.virtual += round(math.dist(self.position, point)/5, 6)
            self.position = point
            source = self._sources.get(channel) if channel not in self.cleared else None
            distance = math.dist(point, source.position) if source else math.inf
            if path == '/measure':
                self.virtual += 5 + int(channel != self.channel)
                self.channel = channel
                if source is None or not visible(source, point):
                    response['measure_result'] = 'no_signal'
                elif distance <= 5:
                    response['measure_result'] = 'near'
                else:
                    bearing = math.degrees(math.atan2(source.position[1]-point[1], source.position[0]-point[0]))
                    response.update(measure_result='direction', svd_deg=round((bearing+self._error(channel, point)) % 360, 2) % 360)
            elif distance <= 20:
                self.cleared.add(channel)
                self.virtual += 5
                response['clear_result'] = 'success'
            else:
                self.virtual += 3
                response['clear_result'] = 'no_target_in_range'
        response['virtual_time_s'] = round(self.virtual, 6)
        self.cache[key] = signature, dict(response)
        self.history.append({'path': path, 'payload': json.loads(json.dumps(payload)), 'response': dict(response)})
        return 200, response

    def truth_for_evaluation(self):
        if not self.ended:
            raise RuntimeError('第四问真值只允许退出后离线评估')
        result = super().truth_for_evaluation()
        for item, source in zip(result['目标'], self._sources.values()):
            item.update(类型='全向' if source.axis_deg is None else '定向', 发射轴_度=source.axis_deg)
        result['遗漏频道'] = sorted(set(self._sources) - self.cleared)
        return result


def make_case(seed, count=13, layout='uniform', radius_mode='min', error_mode='hash', directional_fraction=.5, orientation='random'):
    if not 1 <= count <= 16 or not 0 <= directional_fraction <= 1:
        raise ValueError('构造目标数或定向比例不合法')
    if radius_mode not in {'min', 'max', 'mixed'} or orientation not in {'random', 'outward', 'inward', 'tangent'}:
        raise ValueError('半径或发射朝向模式不合法')
    rng = np.random.default_rng(seed)
    channels = rng.choice(np.arange(1, 21), count, replace=False)
    angles = rng.uniform(0, 2*np.pi, count)
    if layout == 'boundary':
        angles = np.arange(count)*2*np.pi/count + rng.uniform(0, .1)
        positions = np.column_stack((np.cos(angles), np.sin(angles))) * rng.uniform(1790, 1800, (count, 1))
    elif layout == 'center':
        positions = np.column_stack((np.cos(angles), np.sin(angles))) * rng.uniform(6, 180, (count, 1))
        positions[0] = [0, 0]
    elif layout == 'cluster':
        positions = np.clip(rng.normal([1150, -350], [40, 40], (count, 2)), -1250, 1250)
    elif layout == 'uniform':
        positions = np.column_stack((np.cos(angles), np.sin(angles))) * 1800*np.sqrt(rng.uniform(size=(count, 1)))
    else:
        raise ValueError('未知构造布局')
    radii = rng.uniform(1000, 1500, count) if radius_mode == 'mixed' else np.full(count, 1000. if radius_mode == 'min' else 1500.)
    directional = set(rng.permutation(count)[:int(math.floor(count*directional_fraction+.5))])
    axes = rng.uniform(0, 360, count)
    if orientation != 'random':
        axes = np.degrees(np.arctan2(positions[:, 1], positions[:, 0])) + {'outward': 0, 'inward': 180, 'tangent': 90}[orientation]
    sources = [Source(int(c), tuple(p), float(r), float(axes[i] % 360) if i in directional else None)
               for i, (c, p, r) in enumerate(zip(channels, positions, radii))]
    return LocalEnvironment(sources, seed=seed, error_mode=error_mode)
