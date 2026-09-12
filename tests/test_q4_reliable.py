"""清除优先版的漏源反例、连续证书、路线候选和异常终止回归。"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from cumcm2026_b.q4_coverage import discovery_mesh, coverage_status, open_route
from cumcm2026_b.q4_reliable_strategy import ReliableFour, ReliableConfig
from cumcm2026_b.q4_strategy import search_stations
from cumcm2026_b.q4_local_env import LocalEnvironment, Source, make_case, visible
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_geometry import circle_polygon, min_distance

ROOT = Path(__file__).resolve().parents[1]


def test_mesh_covers_closed_domain_with_reliable_triangles_and_rejects_inner13_certificate():
    stations, cells = discovery_mesh()
    assert len(stations) == 31 and len(cells) == 42
    assert coverage_status(stations)['已覆盖单元数'] == 42
    assert coverage_status(search_stations())['未覆盖单元']
    assert coverage_status([])['已覆盖单元数'] == 0
    angles = np.linspace(0, 2 * math.pi, 721)
    for p in 1800 * np.column_stack((np.cos(angles), np.sin(angles))):
        assert min(min_distance(stations[ids], p) for ids in cells) < 1e-6
        vectors = stations - p
        good = vectors[np.linalg.norm(vectors, axis=1) <= 999.5]
        bearings = np.sort(np.arctan2(good[:, 1], good[:, 0]))
        assert np.diff(np.r_[bearings, bearings[0] + 2 * math.pi]).max() < math.pi


def test_compact_mesh_encloses_entire_disk_and_all_edges_have_distance_margin():
    from scipy.spatial import ConvexHull
    stations, cells = discovery_mesh('compact')
    assert len(stations) == 25 and len(cells) == 36
    assert -ConvexHull(stations).equations[:, 2].max() > 1835
    triangles = stations[cells]
    assert np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max() < 984
    assert coverage_status(stations, 'compact')['已覆盖单元数'] == 36
    assert coverage_status(search_stations(), 'compact')['未覆盖单元']


def test_strict_convex_substitution_is_checked_in_distance_and_position():
    stations, triangles = discovery_mesh()
    vertices = stations[triangles[20]]
    center = vertices.mean(axis=0)
    expanded = center + 1.01 * (vertices - center)
    assert coverage_status(expanded)['单元证据'][20]['类型'] == '严格凸包替代'
    assert coverage_status(center + .99 * (vertices - center))['单元证据'][20] is None
    assert coverage_status(center + 4 * (vertices - center))['单元证据'][20] is None


@pytest.mark.parametrize('i,channel,step,next_index', [(3, 10, 197, 7), (4, 8, 264, 8)])
def test_recorded_long_tail_has_an_admissible_probe_before_deferral(i, channel, step, next_index):
    archive = json.loads((ROOT / f'results/models/第四问/实测方向反馈_新建文件夹2/F4-{i}.json').read_text(encoding='utf8'))
    solver = ReliableFour(RobotClient(LocalEnvironment([]), 'local-robot'), ReliableConfig(planning_seconds=10))
    state = solver.channels[channel]
    current = np.zeros(2)
    for e in archive['结果']['动作记录']:
        if e['步骤'] >= step:
            break
        if e['动作'] in {'检测', '清除'}:
            current = np.array(e['位置'])
        if e.get('频道') != channel:
            continue
        if e['动作'] == '检测':
            state.region.observe(e['位置'], e['结果'], e.get('示向度'))
            state.measured_sites.append(e['位置'])
            if e['结果'] != 'no_signal':
                state.status = 'found'
            state.localization_count += e['用途'] == '追加测向'
        elif e['动作'] == '清除' and e['结果'] != 'success':
            state.region.failed_clear(e['位置'])
    solver._remaining = solver.layout[next_index:]
    outer = state.region.vertices.copy()
    plan = solver._observation_plan(state, current)
    assert plan['动作'] == '追加测向'
    assert 0 < plan['路线可行候选数'] < plan['路线筛选前候选数']
    p, nxt = plan['位置'], solver._remaining[0]
    assert math.dist(current, p) + math.dist(p, nxt) - math.dist(current, nxt) <= 800
    assert np.array_equal(outer, state.region.vertices)


def test_close_stop_can_receive_a_directional_source_missed_at_previous_stop():
    source = Source(8, (1400, -600), 1000, 20)
    assert not visible(source, (950, 0)) and visible(source, (1200, 0))
    client = RobotClient(LocalEnvironment([source], error_mode='zero'), 'local-robot')
    solver = ReliableFour(client)
    client.enter()
    solver._measure(solver.channels[8], np.array([950., 0.]), '测试旧站')
    solver._measure(solver.channels[1], np.array([1200., 0.]), '测试当前停点')
    distance = client.state.distance
    solver._opportunistic(np.array([1200., 0.]), 1)
    assert solver.channels[8].status == 'found' and client.state.distance == distance
    count = client.state.measurements
    solver._opportunistic(np.array([1200., 0.]), 1)
    assert client.state.measurements == count
    client.exit()
    client.close()


def test_distance_skip_uses_whole_polygon_and_does_not_invent_measurements():
    solver = ReliableFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[1]
    state.status = 'found'
    state.region.vertices = circle_polygon((0, 0), 80)
    assert solver._skip_measure(state, np.array([1600., 0.]))
    assert not solver._skip_measure(state, np.array([1510., 0.]))
    assert not state.measured_sites and not state.region.observations
    solver._refresh_absence()
    assert state.status == 'found' and not any(s.status == 'absent' for s in solver.channels.values())


@pytest.mark.parametrize('count,error', [(10, 'plus'), (16, 'minus')])
def test_outward_boundary_sources_are_all_cleared_with_truth_hidden_until_exit(count, error):
    env = make_case(106100 + count, count, 'boundary', 'min', error, 1., 'outward')
    client = RobotClient(env, 'local-robot')
    solver = ReliableFour(client)
    with pytest.raises(RuntimeError):
        env.truth_for_evaluation()
    result = solver.run()
    truth = env.truth_for_evaluation()
    client.close()
    assert result['运行成功'] and result['全部完成证据'], result['异常']
    assert truth['全部清除'] and result['清除数'] == count
    assert 0 < len(result['发现保障站坐标']) <= 12
    assert not result['已发现未清除频道']
    targets = {s['频道']: s['位置'] for s in truth['目标']}
    for row in result['频道记录']:
        assert row['追加测向次数'] <= 8
        if row['不存在证书'] is not None:
            assert row['频道'] not in targets
            assert not coverage_status(row['不存在证书']['实际负观测站'], row['不存在证书']['网格'])['未覆盖单元']
    for e in result['动作记录']:
        if '外包顶点' in e:
            assert min_distance(e['外包顶点'], targets[e['频道']]) < 1e-5
        if e['动作'] == '清除' and e['具有覆盖证书']:
            assert e['结果'] == 'success' and math.dist(e['位置'], targets[e['频道']]) <= 20


def test_budget_exhaustion_never_reports_all_cleared_or_absent():
    client = RobotClient(make_case(106000, 10), 'local-robot')
    result = ReliableFour(client, ReliableConfig(max_actions=1)).run()
    client.close()
    assert not result['运行成功'] and not result['全部完成证据']
    assert result['异常'] and result['正常退出'] and not result['已证明不存在频道']


def test_already_cleared_sixteen_does_not_run_exterior_discovery():
    client = RobotClient(LocalEnvironment([Source(c, (0, 0), 1000, 0) for c in range(1, 17)]), 'local-robot')
    result = ReliableFour(client).run()
    client.close()
    assert result['运行成功'] and result['清除数'] == 16 and not result['发现保障站坐标']


def test_sixteen_found_sources_are_serviced_before_any_unknown_discovery():
    solver = ReliableFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    for c in range(1, 17):
        solver.channels[c].status = 'found'
    assert not solver.complete() and not len(solver._needed_stations())
    assert solver._skip_measure(solver.channels[20], np.zeros(2))


def test_no_sample_gain_never_certifies_absence_or_deletes_discovery_stations():
    solver = ReliableFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[1]
    state.negative_sites = [[0., 0.]]
    assert not solver._has_sample_gain(state, np.zeros(2))
    solver._refresh_absence()
    assert state.status == 'unknown' and state.absence_certificate is None
    assert len(solver._needed_stations()) > 0


def test_open_route_preserves_every_station_without_requiring_return():
    points = np.array([[1, 0], [3, 0], [2, 0]], float)
    assert np.array_equal(open_route([0, 0], points), [[1, 0], [2, 0], [3, 0]])


@pytest.mark.parametrize('kwargs', [{'discovery_safeguard': 1}, {'skip_out_of_range': 'yes'},
                                   {'unknown_scan_spacing_m': -1}, {'unknown_scan_spacing_m': float('nan')}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ReliableConfig(**kwargs)
