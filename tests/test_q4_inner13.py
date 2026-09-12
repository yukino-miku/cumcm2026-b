"""第四问13站物理、反馈分支、终止证据和路线约束的回归验证。"""
import math
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q4_local_env import Source, LocalEnvironment, make_case, visible
from cumcm2026_b.q4_strategy import Q4Config, SchemeFour, DirectionalRegion, search_stations


def assert_events_consistent(result, truth):
    targets = {s['频道']: s for s in truth['目标']}
    for event in result['动作记录']:
        if '外包顶点' in event:
            assert min_distance(event['外包顶点'], targets[event['频道']]['位置']) <= 1e-5
        if event['动作'] == '清除' and event['具有覆盖证书']:
            assert event['结果'] == 'success'
            assert math.dist(event['位置'], targets[event['频道']]['位置']) <= 20
    assert not result['已证明不存在频道']
    assert abs(sum(result['时间分项_秒'].values())-result['虚拟总时间_秒']) < .001


def test_exact_thirteen_station_order_and_optimal_length():
    points = search_stations()
    assert points.shape == (13, 2)
    assert np.allclose(points[0], [0, 0])
    assert np.max(np.linalg.norm(points, axis=1)) < 1800
    assert np.allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), 950)
    pairwise = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    assert pairwise.min() == pytest.approx(950)


@pytest.mark.parametrize('angle,expected', [(0, True), (89.999, True), (90, True),
                                            (90.001, False), (180, False), (270, True), (269.999, False)])
def test_fixed_half_plane_including_boundary(angle, expected):
    source = Source(1, (0, 0), 1000, 0)
    point = (500*math.cos(math.radians(angle)), 500*math.sin(math.radians(angle)))
    assert visible(source, point) == expected


def test_range_near_and_clear_are_distinct_and_clear_does_not_switch():
    env = LocalEnvironment([Source(2, (0, 0), 1000, 0)], error_mode='zero')
    client = RobotClient(env, 'local-robot')
    client.enter()
    assert client.measure((1000, 0), 2)['measure_result'] == 'direction'
    assert client.measure((1000.001, 0), 2)['measure_result'] == 'no_signal'
    assert client.measure((5, 0), 2)['measure_result'] == 'near'
    assert client.measure((-1, 0), 2)['measure_result'] == 'no_signal'
    client.measure((-1, 0), 1)
    assert client.clear((-20.001, 0), 2)['clear_result'] == 'no_target_in_range'
    assert client.clear((-20, 0), 2)['clear_result'] == 'success'
    assert client.state.channel == 1
    client.exit()
    assert env.truth_for_evaluation()['全部清除']


def test_repeated_request_is_idempotent_even_after_source_is_cleared():
    env = LocalEnvironment([Source(1, (100, 0), 1000, 180)], error_mode='plus')
    client = RobotClient(env, 'local-robot')
    client.enter()
    first = client.measure((0, 0), 1)
    saved = env.history[-1]
    client.clear((100, 0), 1)
    time_after_clear = env.virtual
    status, replay = env.post(saved['path'], saved['payload'])
    assert status == 200 and replay == first and env.virtual == time_after_clear
    client.exit()


def test_direction_then_backside_does_not_erase_real_target_or_mark_absent():
    env = LocalEnvironment([Source(1, (500, 0), 1000, 180)], error_mode='zero')
    client = RobotClient(env, 'local-robot')
    solver = SchemeFour(client)
    client.enter()
    state = solver.channels[1]
    assert solver._measure(state, [0, 0], '测试接收') == 'direction'
    before = state.region.vertices.copy()
    assert solver._measure(state, [700, 0], '测试背侧') == 'no_signal'
    assert state.status == 'found' and state.region.maybe_contains([500, 0])
    assert np.array_equal(before, state.region.vertices) and not state.region.exclusions
    assert state.absence_certificate is None
    client.exit()


def test_negative_observation_only_records_ambiguity_and_optical_failure_remains_valid():
    region = DirectionalRegion()
    region.observe([0, 0], 'no_signal')
    assert not region.exclusions and region.maybe_contains([200, 0])
    region.failed_clear([200, 0])
    assert not region.maybe_contains([200, 0])
    assert region.exclusions[0]['半径'] == 20


def test_all_fixed_stations_finish_without_silent_peripheral_search_or_false_completion():
    # 未发现源不触发定位动作，固定13站应严格保留，总移动应正好11.4公里。
    env = LocalEnvironment([Source(1, (1700, 0), 1000, 0)])
    result = SchemeFour(RobotClient(env, 'local-robot')).run()
    assert result['流程正常完成'] and not result['全部完成证据'] and not result['运行成功']
    assert result['清除数'] == 0 and result['未发现频道'] == list(range(1, 21))
    assert result['检测次数'] == 260 and result['总路程_米'] == pytest.approx(11400)
    assert result['已完成搜索站'] == list(range(13))
    assert result['末尾增站评估']['自动新增未知源搜索站数'] == 0
    assert not env.truth_for_evaluation()['全部清除']
    assert all(e['用途'].startswith('固定搜索站') for e in result['动作记录'] if e['动作'] == '检测')
    assert_events_consistent(result, env.truth_for_evaluation())


def test_no_signal_during_extra_measurement_is_normal_and_finite_clear_fallback_works():
    class ControlledProbe(SchemeFour):
        def _observation_plan(self, state, current):
            if state.localization_count == 0:
                return {'动作': '追加测向', '位置': [700, 0], '接收保证': False}
            return {'动作': '覆盖清除'}

    env = LocalEnvironment([Source(1, (500, 0), 1000, 180)], error_mode='zero')
    client = RobotClient(env, 'local-robot')
    solver = ControlledProbe(client, Q4Config(opportunistic=False))
    client.enter()
    solver._measure(solver.channels[1], [0, 0], '测试发现')
    solver.phase = '固定扫描后处理已发现源'
    solver._service_known()
    assert solver.channels[1].status == 'cleared'
    assert any(e['动作'] == '补测无信号' for e in solver.events)
    assert any(e['动作'] == '有限覆盖' for e in solver.events)
    client.exit()
    assert env.truth_for_evaluation()['全部清除']


@pytest.mark.parametrize('layout,count,orientation', [('uniform', 10, 'random'), ('boundary', 13, 'inward'), ('center', 16, 'tangent')])
def test_mixed_closed_loop_truth_inclusion_clear_certificates_and_route_order(layout, count, orientation):
    env = make_case(20260912, count, layout, 'mixed', 'hash', .5, orientation)
    result = SchemeFour(RobotClient(env, 'local-robot')).run()
    assert result['流程正常完成'], result['异常']
    truth = env.truth_for_evaluation()
    assert result['已完成搜索站'] == list(range(len(result['已完成搜索站'])))
    assert all(e['阶段'] != '末尾增站评估' for e in result['动作记录'] if e['动作'] == '检测')
    assert not result['已发现未清除频道']
    assert_events_consistent(result, truth)
    for event in result['动作记录']:
        if event['动作'] == '路线规划':
            fixed = [n['剩余站序'] for n in event['计划路径'] if n['类型'] == '固定站']
            assert fixed == sorted(fixed)


def test_known_targets_are_serviced_before_last_station():
    env = LocalEnvironment([Source(1, (480, 0), 1000, 180)], error_mode='zero')
    result = SchemeFour(RobotClient(env, 'local-robot')).run()
    assert result['流程正常完成'] and env.truth_for_evaluation()['全部清除']
    end = next(e['步骤'] for e in result['动作记录'] if e['动作'] == '固定扫描完成')
    assert any(e['动作'] == '清除' and e['步骤'] < end for e in result['动作记录'])


def test_sixteen_distinct_clears_prove_completion_without_reading_truth():
    env = LocalEnvironment([Source(c, (0, 0), 1000, 0 if c % 2 else None) for c in range(1, 17)])
    result = SchemeFour(RobotClient(env, 'local-robot')).run()
    assert result['运行成功'] and result['全部完成证据'] and result['清除数'] == 16
    assert not result['末尾增站评估']['仍需评估增站']


def test_action_budget_stops_cleanly_and_is_not_a_finished_experiment():
    env = make_case(6, 10)
    result = SchemeFour(RobotClient(env, 'local-robot'), Q4Config(max_actions=3)).run()
    assert not result['流程正常完成'] and not result['运行成功'] and result['正常退出']
    assert result['异常'].startswith('BudgetExceeded')


def test_truth_is_unavailable_to_online_strategy():
    env = make_case(4, 10)
    with pytest.raises(RuntimeError):
        env.truth_for_evaluation()


def test_http_entry_with_local_protocol_stub_and_saved_provenance(tmp_path):
    # 仅本机临时HTTP服务，不连接官方模拟器。通过真实入口验证协议和结果文件。
    env = LocalEnvironment([Source(c, (0, 0), 1000, 0 if c % 2 else None) for c in range(1, 17)])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            status, response = env.post(self.path, payload)
            raw = json.dumps(response).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            root = Path(__file__).resolve().parents[1]
            output = tmp_path/'http-result'
            process = subprocess.run([sys.executable, '-X', 'utf8', str(root/'scripts/run_q4.py'),
                '--mode', 'http', '--robot-id', 'local-robot', '--base-url', f'http://127.0.0.1:{server.server_port}',
                '--output', str(output)], cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=40)
            assert process.returncode == 0, process.stdout+process.stderr
            result = json.loads((output/'运行结果.json').read_text(encoding='utf-8'))
            assert result['流程正常完成'] and result['全部完成证据']
            assert '离线真值核验' not in result and result['运行来源']['源码_SHA256']
            assert (output/'请求响应日志.jsonl').exists()
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize('parameters', [{'candidate_limit': True}, {'planning_seconds': float('nan')},
                                      {'consecutive_no_signal_limit': 0}, {'skip_localized': 'yes'}])
def test_invalid_configuration(parameters):
    with pytest.raises(ValueError):
        Q4Config(**parameters)
