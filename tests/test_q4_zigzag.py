from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import math
import pytest

from cumcm2026_b.q4_zigzag_strategy import ZigzagConfig, ZigzagFour
from cumcm2026_b.q4_local_env import LocalEnvironment, Source, make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_geometry import min_distance


def solver_with_anchor(bearing=90.):
    solver = ZigzagFour(RobotClient(LocalEnvironment([]), 'local-robot'))
    state = solver.channels[1]
    state.status = 'found'
    state.region.observe([0, 0], 'direction', bearing)
    state.measured_sites.append([0, 0])
    solver.events.append({'步骤': 1, '动作': '检测', '用途': '固定搜索站1', '频道': 1,
                          '位置': [0, 0], '结果': 'direction', '示向度': bearing})
    return solver, state


def append_probe(solver, state, plan, result='no_signal'):
    state.localization_count += 1
    state.measured_sites.append(plan['位置'])
    solver.events.append({'步骤': len(solver.events)+1, '动作': '检测', '用途': '追加测向',
                          '频道': 1, '位置': plan['位置'], '结果': result, '示向度': 80. if result == 'direction' else None})


def test_alternation_contraction_and_deferred_plans_do_not_consume_attempts():
    solver, state = solver_with_anchor()
    p1 = solver._observation_plan(state, [0, 0])
    assert solver._observation_plan(state, [0, 0]) == p1
    append_probe(solver, state, p1)
    p2 = solver._observation_plan(state, p1['位置'])
    assert p1['横向偏移_米'] * p2['横向偏移_米'] < 0
    assert p2['前进距离_米'] > p1['前进距离_米']
    append_probe(solver, state, p2)
    p3 = solver._observation_plan(state, p2['位置'])
    assert p3['步长缩放'] == .4 and p3['锚点步骤'] == 1
    assert p3['前进距离_米'] < p1['前进距离_米']


def test_new_positive_anchors_and_budget_remains_global():
    solver, state = solver_with_anchor()
    p = solver._observation_plan(state, [0, 0]); append_probe(solver, state, p, 'direction')
    new = solver._observation_plan(state, p['位置'])
    assert new['锚点'] == p['位置'] and new['锚点后主补测序号'] == 1
    state.localization_count = 8
    assert solver._observation_plan(state, p['位置'])['动作'] == '覆盖清除'


@pytest.mark.parametrize('parameters', [{'zigzag_forward_m': 0}, {'zigzag_lateral_m': True},
                                       {'zigzag_forward_m': math.inf}, {'zigzag_lateral_m': math.nan}])
def test_bad_configuration(parameters):
    with pytest.raises(ValueError): ZigzagConfig(**parameters)


@pytest.mark.parametrize('seed,layout,orientation', [(74001, 'uniform', 'random'), (74002, 'boundary', 'tangent'), (74003, 'cluster', 'inward')])
def test_closed_loop_uses_feedback_only_and_preserves_clear_safety(seed, layout, orientation):
    env = make_case(seed, 10, layout, 'min', 'hash', 1., orientation)
    result = ZigzagFour(RobotClient(env, 'local-robot')).run()
    assert result['流程正常完成'], result['异常']
    truth = env.truth_for_evaluation()
    targets = {s['频道']: s['位置'] for s in truth['目标']}
    assert result['末尾增站评估']['自动新增未知源搜索站数'] == 0
    assert not result['已发现未清除频道']
    assert result['已完成搜索站'] == list(range(13))
    for e in result['动作记录']:
        if '外包顶点' in e: assert min_distance(e['外包顶点'], targets[e['频道']]) < 1e-5
        if e['动作'] == '清除' and e['具有覆盖证书']:
            assert e['结果'] == 'success' and math.dist(e['位置'], targets[e['频道']]) <= 20
        if e['动作'] == '路线规划':
            fixed = [n['剩余站序'] for n in e['计划路径'] if n['类型'] == '固定站']
            assert fixed == sorted(fixed)


def test_directional_undiscovered_sources_are_not_silently_searched_outside():
    sources = [Source(c, (1700, 0), 1000, 0) for c in range(1, 11)]
    env = LocalEnvironment(sources)
    result = ZigzagFour(RobotClient(env, 'local-robot')).run()
    assert result['流程正常完成'] and result['清除数'] == 0 and not result['全部完成证据']
    assert result['总路程_米'] == pytest.approx(11400)
    assert all(e.get('用途') != '追加测向' for e in result['动作记录'])


@pytest.mark.parametrize('spacing', [None, 0, 200])
def test_withdrawn_zigzag_http_entry_uses_original_strategy(tmp_path, spacing):
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
            process = subprocess.run([sys.executable, '-X', 'utf8', str(root/'scripts/run_q4_zigzag.py'),
                '--mode', 'http', '--robot-id', 'local-robot', '--base-url', f'http://127.0.0.1:{server.server_port}',
                *(['--probe-spacing', str(spacing)] if spacing is not None else []),
                '--output', str(output)], cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=40)
            assert process.returncode == 0, process.stdout+process.stderr
            result = json.loads((output/'运行结果.json').read_text(encoding='utf-8'))
            assert result['流程正常完成'] and result['全部完成证据']
            assert '离线真值核验' not in result and result['运行来源']['源码_SHA256']
            assert result['配置']['localization_detour_limit_m'] == 400
            assert result['配置']['probe_spacing_m'] == (100 if spacing is None else spacing)
            assert 'zigzag_forward_m' not in result['配置']
            assert '折线' not in result['方案']
            assert '折线方案已按用户要求撤回' in process.stdout
            assert (output/'请求响应日志.jsonl').exists()
        finally:
            server.shutdown()
            thread.join(timeout=5)


