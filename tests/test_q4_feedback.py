"""小范围联合覆盖、方向负反馈、有限回退及原版关闭开关的行为核验。"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cumcm2026_b.q4_feedback_strategy import (FeedbackFour, FeedbackConfig, orientation_arcs,
                                              joint_hypotheses, reception_support, ordered_small_sweep)
from cumcm2026_b.q4_spacing_strategy import SpacedFour, SpacingConfig
from cumcm2026_b.q4_local_env import LocalEnvironment, Source, make_case, visible
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_geometry import max_distance, min_distance, circle_polygon

ROOT = Path(__file__).resolve().parents[1]


def contains_angle(arcs, degrees):
    angle = math.radians(degrees) % (2 * math.pi)
    return any(a - 1e-8 <= angle <= b + 1e-8 for a, b in arcs)


def test_orientation_intervals_keep_true_axis_across_wrap_and_ignore_far_negative():
    position = np.array([300., 0.])
    observations = [{'位置': [0., 0.], '结果': 'direction'},
                    {'位置': [700., 200.], '结果': 'no_signal'}]
    arcs = orientation_arcs(position, observations)
    assert contains_angle(arcs, 180) and not contains_angle(arcs, 0)
    distant = observations + [{'位置': [-1800., 0.], '结果': 'no_signal'}]
    assert orientation_arcs(position, distant) == arcs
    east = orientation_arcs(np.zeros(2), [{'位置': [100, 0], '结果': 'direction'}])
    assert contains_angle(east, 359) and contains_angle(east, 1) and not contains_angle(east, 180)


def test_known_positive_radius_lower_bound_can_explain_negative_as_backside():
    observations = [{'位置': [1200., 0.], '结果': 'direction'},
                    {'位置': [0., -1100.], '结果': 'no_signal'}]
    arcs = orientation_arcs(np.zeros(2), observations)
    assert contains_angle(arcs, 45) and not contains_angle(arcs, 315)


def test_feedback_moves_to_receiving_side_without_shrinking_outer_region():
    source = Source(1, (300., 0.), 1000., 180.)
    client = RobotClient(LocalEnvironment([source], error_mode='zero'), 'local-robot')
    solver = FeedbackFour(client, FeedbackConfig(opportunistic=False))
    client.enter()
    state = solver.channels[1]
    solver._measure(state, np.array([0., 0.]), '测试发现')
    solver._measure(state, np.array([700., 200.]), '测试无信号')
    outer = state.region.vertices.copy()
    model = joint_hypotheses(state)
    assert reception_support(model, np.array([250., -150.])) > reception_support(model, np.array([700., 200.]))
    plan = solver._observation_plan(state, np.array([700., 200.]))
    assert plan['方向反馈已启用'] and not plan['接收保证']
    assert visible(source, plan['位置'])
    # 该近源反例需要允许候选离部分尚未排除的位置超过1000米。
    assert not plan['距离保证'] and plan['最远可能目标距离_米'] > 1000
    assert np.array_equal(outer, state.region.vertices) and not state.region.exclusions
    assert state.localization_count == 0
    solver._service_known()
    assert state.status == 'cleared' and state.localization_count <= 8
    client.exit()
    client.close()


def test_no_confirmed_backside_keeps_original_candidate_method():
    solver = FeedbackFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[1]
    state.status = 'found'
    state.region.observe([0., 0.], 'direction', 0.)
    state.measured_sites.append([0., 0.])
    state.region.observe([-1600., 0.], 'no_signal')
    assert not joint_hypotheses(state)
    plan = solver._observation_plan(state, np.zeros(2))
    assert not plan.get('方向反馈已启用', False)


def test_sample_degeneracy_falls_back_without_losing_continuous_region(monkeypatch):
    from cumcm2026_b import q4_feedback_strategy as module
    solver = FeedbackFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[1]
    state.region.observe([0, 0], 'direction', 0.)
    state.measured_sites.append([0, 0])
    outer = state.region.vertices.copy()
    monkeypatch.setattr(module, 'joint_hypotheses', lambda *args: [])
    assert solver._observation_plan(state, np.zeros(2))['动作'] in ('追加测向', '覆盖清除')
    assert np.array_equal(outer, state.region.vertices)


def test_two_point_task_is_inserted_before_leaving_northwest_using_only_then_known_geometry():
    r = json.loads((ROOT / 'results/models/第四问/实测100米间距/N4-5.json').read_text(encoding='utf8'))
    obs = next(e for e in r['结果']['动作记录'] if e['步骤'] == 156)
    vertices, current = np.array(obs['外包顶点']), np.array(obs['位置'])
    destination = np.array([-1425., 950 * math.sqrt(3) / 2])
    plan = ordered_small_sweep(vertices, current, destination, 4)
    assert plan['覆盖点数'] == 2 and plan['完整绕行_米'] < 211 and plan['增加费用上界_秒'] < 51
    grid = np.asarray(plan['位置序列'])
    # 所有边界及内部凸组合均被联合覆盖；每一点单独不冒充完整证书。
    center = vertices.mean(axis=0)
    samples = np.concatenate([center + t * (vertices - center) for t in np.linspace(0, 1, 31)])
    assert np.linalg.norm(samples[:, None, :] - grid[None, :, :], axis=2).min(axis=1).max() < 20
    assert all(max_distance(vertices, p) > 19.5 for p in grid)
    solver = FeedbackFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[15]
    state.status = 'found'
    state.region.vertices = vertices
    remaining = solver.layout[6:]
    choice = solver._small_task([state], current, remaining, 24)
    assert choice is not None and choice[1] == 15


def test_small_region_still_obeys_complete_detour_and_action_allowance():
    r = json.loads((ROOT / 'results/models/第四问/实测100米间距/N4-6.json').read_text(encoding='utf8'))
    obs = next(e for e in r['结果']['动作记录'] if e['步骤'] == 78)
    solver = FeedbackFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[11]
    state.status = 'found'
    state.region.vertices = np.array(obs['外包顶点'])
    p = ordered_small_sweep(state.region.vertices, np.array(obs['位置']), solver.layout[3], 4)
    assert p['覆盖点数'] == 3 and p['完整绕行_米'] > 1400
    assert solver._small_task([state], np.array(obs['位置']), solver.layout[3:], 24) is None
    assert ordered_small_sweep(state.region.vertices, np.array(obs['位置']), solver.layout[3], 2) is None
    assert ordered_small_sweep(circle_polygon(), np.zeros(2), solver.layout[1], 4) is None


def test_all_extensions_off_reproduces_spacing_baseline_actions():
    def run(cls, cfg):
        env = make_case(100201, 10, 'uniform', 'min', 'hash', 1., 'random')
        client = RobotClient(env, 'local-robot')
        try:
            return cls(client, cfg).run()
        finally:
            client.close()
    baseline = run(SpacedFour, SpacingConfig(probe_spacing_m=100))
    disabled = run(FeedbackFour, FeedbackConfig(small_sweep_limit=0, direction_feedback=False, task_routing=False))
    fields = ('动作', '用途', '频道', '位置', '结果')
    project = lambda r: [{k: e.get(k) for k in fields} for e in r['动作记录'] if e['动作'] in ('检测', '清除')]
    assert project(baseline) == project(disabled)
    assert abs(baseline['虚拟总时间_秒'] - disabled['虚拟总时间_秒']) < .001


@pytest.mark.parametrize('parameters', [{'small_sweep_limit': True}, {'small_sweep_limit': -1},
                                       {'small_sweep_limit': 7}, {'direction_feedback': 1},
                                       {'task_routing': 'yes'}, {'position_samples': 3}])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        FeedbackConfig(**parameters)


@pytest.mark.parametrize('layout,orientation', [('uniform', 'random'), ('boundary', 'outward'), ('cluster', 'tangent')])
def test_complete_loop_preserves_truth_clear_certificates_and_finite_service(layout, orientation):
    env = make_case(100310, 12, layout, 'mixed', 'hash', 1., orientation)
    client = RobotClient(env, 'local-robot')
    try:
        result = FeedbackFour(client).run()
    finally:
        client.close()
    assert result['流程正常完成'] and not result['已发现未清除频道']
    truth = {s['频道']: s['位置'] for s in env.truth_for_evaluation()['目标']}
    for e in result['动作记录']:
        if '外包顶点' in e:
            assert min_distance(e['外包顶点'], truth[e['频道']]) < 1e-5
        if e['动作'] == '清除' and e['具有覆盖证书']:
            assert e['结果'] == 'success' and math.dist(e['位置'], truth[e['频道']]) <= 20
    assert all(s['追加测向次数'] <= 8 for s in result['频道记录'])
    assert result['末尾增站评估']['自动新增未知源搜索站数'] == 0
