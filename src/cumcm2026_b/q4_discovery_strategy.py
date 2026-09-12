"""补齐真实顺路停点的未知频道扫描，避免前四个候选挤掉未发现源。"""
from dataclasses import dataclass

import numpy as np

from .q4_feedback_strategy import FeedbackConfig, FeedbackFour


@dataclass
class DiscoveryConfig(FeedbackConfig):
    scan_all_unknown: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.scan_all_unknown, bool):
            raise ValueError('scan_all_unknown必须为布尔值')


class DiscoveryFour(FeedbackFour):
    def __init__(self, client, config=None):
        super().__init__(client, config or DiscoveryConfig())

    def _opportunistic(self, point, primary):
        # 先保留原本最多4个频道的顺路处理，包括已发现但未定位的频道。
        super()._opportunistic(point, primary)
        if not self.config.opportunistic or not self.config.scan_all_unknown or self.complete():
            return
        pending = [s for s in self.channels.values() if s.status == 'unknown' and s.channel != primary
                   and (not s.measured_sites or min(np.linalg.norm(np.asarray(p) - point)
                                                   for p in s.measured_sites) >= 600)]
        pending.sort(key=lambda s: (s.channel != self.client.state.channel, len(s.measured_sites), s.channel))
        if pending:
            self._log({'动作': '补齐顺路未知频道', '位置': np.asarray(point).tolist(),
                       '候选频道': [s.channel for s in pending], '同频道站距门槛_米': 600,
                       '说明': '只复用已到达停点；补齐原4频道上限之外的合格未知频道'})
        for state in pending:
            self._measure(state, point, '顺路检测')
            if self.complete():
                break

    def run(self):
        result = super().run()
        result['方案'] += '；合格未知频道顺路扫描补全'
        return result
