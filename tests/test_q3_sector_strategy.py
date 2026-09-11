"""扇形推进证据、逐频道覆盖、完整区域安全性及实际协议闭环。"""
import json
import math
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import circle_polygon, min_distance
from cumcm2026_b.q3_sector_geometry import intersect_sector, sector_polygon, sector_midpoint, cover_polygon
from cumcm2026_b.q3_sector_strategy import SchemeOneSector, SectorConfig, sector_route
from cumcm2026_b.q3_strategy import ChannelState
from cumcm2026_b.q3_local_env import LocalEnvironment, Source, make_case
from cumcm2026_b.q3_protocol import RobotClient


@pytest.mark.parametrize('sector', range(6))
def test_center_and_rotated_midpoint_cover_complete_sector(sector):
    result = cover_polygon(sector_polygon(sector), [[0, 0], sector_midpoint(sector)])
    assert result['covered'] and not len(result['holes'])


def test_current_fixed_station_and_center_do_not_cover_first_sector():
    assert not cover_polygon(sector_polygon(0), [[0, 0], [1200, 0]])['covered']
    assert not cover_polygon(sector_polygon(0), [[0, 0], [1200, 0], [300, 0]])['covered']


def test_union_of_disks_cannot_pass_just_because_each_vertex_is_covered():
    triangle = np.array([[0., 0.], [2000., 0.], [1000., 1700.]])
    result = cover_polygon(triangle, triangle, radius=600., cell_m=5.)
    assert not result['covered']
    assert not cover_polygon(triangle, triangle, radius=600., max_nodes=1)['covered']


def test_radial_boundaries_origin_and_wrap_are_closed_without_mutating_region():
    p = np.array([[1000., 0.]])
    assert len(intersect_sector(p, 0)) and len(intersect_sector(p, 5))
    assert not len(intersect_sector(p, 2))
    assert all(len(intersect_sector([[0., 0.]], i)) for i in range(6))
    p = circle_polygon(); old = p.copy()
    intersect_sector(p, 3)
    assert np.array_equal(p, old)


def test_responsibility_does_not_crop_full_region_to_fake_safe_clear():
    state = ChannelState(1, status='found')
    state.region.vertices = np.array([[800, -300], [1000, -300], [900, 2.]])
    solver = SchemeOneSector(RobotClient(LocalEnvironment([]), 'local-robot'))
    solver.sector = 0
    solver.channels[1] = state
    before = state.region.vertices.copy()
    assert solver._responsibility(state) is None
    route = sector_route([state], [1200, 0], [600, 1039.23])
    assert not route['计划路径'][1]['可安全清除']
    assert np.array_equal(state.region.vertices, before)


def test_coverage_cannot_be_borrowed_from_another_channel():
    solver = SchemeOneSector(RobotClient(LocalEnvironment([]), 'local-robot'))
    solver.sector = 0
    solver.channels[1].negative_sites = [[0, 0], sector_midpoint(0).tolist()]
    solver.channels[2].negative_sites = [[0, 0]]
    assert solver._responsibility(solver.channels[1]) is not None
    assert solver._responsibility(solver.channels[2]) is None


def test_all_responsible_tasks_precede_terminal_including_open_final_sector():
    states = []
    for channel, p in enumerate([[1300, 200], [650, 1100], [1200, -10]], 1):
        s = ChannelState(channel, status='found')
        s.region.vertices = np.asarray(p)+np.array([[-1., -1.], [1., -1.], [0., 1.]])
        states.append(s)
    for terminal in ([600., 1039.23], None):
        nodes = sector_route(states, [1200., 0.], terminal)['计划路径']
        assert sorted(n['频道'] for n in nodes if n['类型'] == '目标') == [1, 2, 3]
        if terminal is not None: assert nodes[-1]['类型'] == '下一固定站'


def verify_truth_gates(result, truth):
    sources = {s['频道']: np.array(s['位置']) for s in truth['目标']}
    cleared, gates = set(), set()
    for e in result['动作记录']:
        if '外包顶点' in e:
            assert min_distance(e['外包顶点'], sources[e['频道']]) <= 1e-5
        if e['动作'] == '清除' and e['结果'] == 'success':
            assert math.dist(e['位置'], sources[e['频道']]) <= 20+1e-6
            cleared.add(e['频道'])
        if e['动作'] == '扇形清空证书':
            sector = e['扇形']; gates.add(sector)
            for c, point in sources.items():
                if len(intersect_sector([point], sector)):
                    assert c in cleared, (sector, c, point)
        if e['动作'] == '检测' and e['用途'].startswith('搜索站'):
            index = int(e['用途'][3:])
            if index >= 2: assert index-2 in gates
    assert cleared == set(sources)
    assert result['已验收扇形'] == list(range(len(result['已验收扇形'])))
    assert sum(result['时间分项_秒'].values()) == pytest.approx(result['虚拟总时间_秒'], abs=.001)
    json.dumps(result, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize('seed,count,layout,radius,error', [
    (31, 10, 'uniform', 'min', 'hash'), (32, 13, 'boundary', 'min', 'plus'),
    (33, 16, 'boundary', 'max', 'minus'), (34, 10, 'center', 'mixed', 'alternating'),
    (35, 13, 'cluster', 'mixed', 'zero'), (36, 16, 'uniform', 'min', 'hash')])
def test_complete_sector_loop_and_each_departure_has_cleared_real_sector(seed,count,layout,radius,error):
    env = make_case(seed, count, layout, radius, error)
    result = SchemeOneSector(RobotClient(env, 'local-robot')).run()
    assert result['运行成功'], result['异常']
    assert env.truth_for_evaluation()['全部清除']
    verify_truth_gates(result, env.truth_for_evaluation())


def test_exact_sector_boundaries_and_late_first_detection_are_not_left_behind():
    positions = [1800*np.array([math.cos(i*math.pi/3), math.sin(i*math.pi/3)]) for i in range(6)]
    positions.append(1800*np.array([math.cos(math.radians(50)), math.sin(math.radians(50))]))
    env = LocalEnvironment([Source(i+1, tuple(p)) for i, p in enumerate(positions)], error_mode='plus')
    result = SchemeOneSector(RobotClient(env, 'local-robot')).run()
    assert result['运行成功'], result['异常']
    verify_truth_gates(result, env.truth_for_evaluation())


def test_budget_stop_is_not_mislabelled_as_sector_clear():
    result = SchemeOneSector(RobotClient(make_case(48, 10), 'local-robot'), SectorConfig(max_actions=24)).run()
    assert not result['运行成功'] and result['正常退出']
    assert not result['已验收扇形']


def test_finite_sweep_fallback_uses_full_region_and_completes():
    env = LocalEnvironment([Source(1, (800., 30.))], error_mode='plus')
    result = SchemeOneSector(RobotClient(env, 'local-robot'), SectorConfig(max_localization_measurements=0, opportunistic=False)).run()
    assert result['运行成功'], result['异常']
    verify_truth_gates(result, env.truth_for_evaluation())
    assert any(e['动作'] == '有限覆盖' for e in result['动作记录'])


@pytest.mark.parametrize('params', [{'coverage_radius_m':1000}, {'coverage_max_nodes':True},
    {'coverage_cell_m':0}, {'reuse_min_gain':float('nan')}, {'nearby_clear_detour_m':-1}])
def test_invalid_sector_config_is_rejected(params):
    with pytest.raises(ValueError): SectorConfig(**params)
