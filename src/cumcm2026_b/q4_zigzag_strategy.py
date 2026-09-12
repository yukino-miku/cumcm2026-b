"""第四问折线补测试验：有效测向后左右交替；成对失去信号时缩短步长。"""
from dataclasses import dataclass
import math
import numpy as np

from .q4_strategy import Q4Config, SchemeFour
from .q3_geometry import max_distance


@dataclass
class ZigzagConfig(Q4Config):
    zigzag_forward_m: float = 200.
    zigzag_lateral_m: float = 80.

    def __post_init__(self):
        super().__post_init__()
        for name in ('zigzag_forward_m', 'zigzag_lateral_m'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}须为正的有限数')


class ZigzagFour(SchemeFour):
    """只替换主补测点选择，沿用13站、路线、外包、光学覆盖及结束规则。"""
    def __init__(self, client, config=None):
        super().__init__(client, config or ZigzagConfig())

    def _observation_plan(self, state, current):
        if state.localization_count >= self.config.max_localization_measurements:
            return {'动作': '覆盖清除', '原因': '达到单频道主补测预算'}
        positives = [e for e in self.events if e.get('频道') == state.channel
                     and e['动作'] == '检测' and e.get('结果') == 'direction']
        if not positives:
            return {'动作': '覆盖清除', '原因': '没有可用的有效方向锚点'}
        last = positives[-1]
        anchor = np.asarray(last['位置'], dtype=float)
        theta = math.radians(last['示向度'])
        forward = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-forward[1], forward[0]])
        projection = (state.region.vertices-anchor) @ forward
        estimated = max(5., float((projection.min()+projection.max())/2))
        step = min(self.config.zigzag_forward_m, max(20., .4*estimated))
        width = min(self.config.zigzag_lateral_m, max(15., .25*estimated))
        attempts = sum(e.get('频道') == state.channel and e.get('用途') == '追加测向'
                       and e['步骤'] > last['步骤'] for e in self.events)
        # 新正观测成为新锚点；无新正观测则左右试探一对后缩短，防止已越过近处源还向前盲走。
        contraction = .4 ** (attempts // 2)
        advance = step * contraction * (1.15 if attempts % 2 else 1.)
        offset = width * contraction
        side = 1 if state.localization_count % 2 == 0 else -1
        point = anchor + advance*forward + side*offset*normal
        if any(np.linalg.norm(point-np.asarray(p)) <= .01 for p in state.measured_sites):
            # 不发出重复的同频道同坐标观测；有限覆盖保留完整位置外包。
            return {'动作': '覆盖清除', '原因': '折线候选与该频道已测位置重复'}
        return {'动作': '追加测向', '位置': point.tolist(), '方式': '有效方向锚定的交替折线',
                '锚点': anchor.tolist(), '锚点步骤': last['步骤'], '锚点示向度': last['示向度'],
                '锚点后主补测序号': attempts+1, '前进距离_米': advance, '横向偏移_米': side*offset,
                '步长缩放': contraction, '预计目标纵向距离_米': estimated,
                '最远可能目标距离_米': max_distance(state.region.vertices, point),
                '接收保证': False,
                '评分性质': '按折线序列生成，未估计发射朝向或接收概率；不作必然接收或最优性声明'}

    def run(self):
        result = super().run()
        result['方案'] = '第四问13站折线补测试验版：左右交替，失去信号成对缩步；保留顺路路线'
        return result
