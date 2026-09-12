"""19站最短路、机制复用、完整扫描与真实入口的本机协议回归。"""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q4_local_env import LocalEnvironment, Source, visible
from cumcm2026_b.q4_reliable_strategy import ReliableFour
from cumcm2026_b.q4_scheme2 import SchemeTwo, SchemeTwoConfig
from cumcm2026_b.q4_scheme2.route import station_pool, shortest_route
from cumcm2026_b.q4_strategy import search_stations

ROOT = Path(__file__).resolve().parents[1]


def test_route_attains_independent_euclidean_lower_bound():
    pool = station_pool()
    route, certificate = shortest_route()
    assert np.array_equal(pool[:13], search_stations())
    assert len(pool) == len(set(map(tuple, pool))) == 19
    assert set(map(tuple, pool)) == set(map(tuple, route))
    assert np.array_equal(route[0], [0, 0])
    assert np.allclose(np.linalg.norm(pool[13:], axis=1), 1900)
    minimum = min(math.dist(a, b) for i, a in enumerate(pool) for b in pool[i+1:])
    cost = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
    assert minimum == pytest.approx(950)
    assert cost == pytest.approx(18 * minimum) == pytest.approx(17100)
    assert certificate['达到的总长度_米'] == pytest.approx(cost)


def test_default_mechanisms_and_config_are_exactly_shared():
    old = json.loads((ROOT / 'configs/q4_reliable.json').read_text(encoding='utf-8'))
    new = json.loads((ROOT / 'configs/q4_scheme2.json').read_text(encoding='utf-8'))
    assert old == new == asdict(SchemeTwoConfig())
    for name in ['_measure', '_clear', '_service_known', '_observation_plan',
                 '_opportunistic', '_skip_measure', 'complete']:
        assert getattr(SchemeTwo, name) is getattr(ReliableFour, name)
    with pytest.raises(ValueError, match='自动补漏'):
        SchemeTwoConfig(discovery_safeguard=True)


def test_all_19_stations_execute_without_false_all_clear_or_extra_search():
    # 凸包外、朝外的源说明19站仍不是任意方向发现证明；不得以流程完成伪造全清。
    p = (1790 * math.cos(math.pi / 6), 1790 * math.sin(math.pi / 6))
    sources = [Source(c, p, 1000, 30) for c in range(1, 11)]
    assert not any(visible(sources[0], q) for q in station_pool())
    env = LocalEnvironment(sources)
    client = RobotClient(env, 'local-robot')
    try:
        result = SchemeTwo(client).run()
        assert result['流程正常完成'] and not result['全部完成证据']
        assert result['清除数'] == 0 and result['已完成搜索站'] == list(range(19))
        assert result['末尾增站评估']['固定十九站是否全部完成']
        assert not result['发现保障站坐标'] and not result['已证明不存在频道']
        assert result['总路程_米'] == pytest.approx(17100)
        assert result['检测次数'] == 380
        # 每站先复用当前频道，20次检测只切换19次。
        assert result['虚拟总时间_秒'] == pytest.approx(17100/5 + 380*5 + 19*19)
        actual = [e['位置'] for e in result['动作记录'] if e['动作'] == '开始固定站']
        assert np.array_equal(actual, shortest_route()[0])
    finally:
        client.close()


@pytest.mark.parametrize('count', [10, 16])
def test_http_entry_scans_all_or_stops_on_16_proof(tmp_path, count):
    # 本机临时HTTP服务器模拟协议，不连接官方软件，也不返回目标总数。
    env = LocalEnvironment([Source(c, (0, 0), 1000, None) for c in range(1, count+1)])

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
            output = tmp_path / 'http-result'
            process = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/run_q4_scheme2.py'),
                '--mode', 'http', '--robot-id', 'local-robot', '--base-url',
                f'http://127.0.0.1:{server.server_port}', '--output', str(output)],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=40)
            assert process.returncode == 0, process.stdout + process.stderr
            result = json.loads((output / '运行结果.json').read_text(encoding='utf-8'))
            assert result['流程正常完成'] and result['清除数'] == count
            assert len(result['固定站坐标']) == 19
            assert result['全部完成证据'] == (count == 16)
            assert len(result['已完成搜索站']) == (1 if count == 16 else 19)
            assert '离线真值核验' not in result and not result['发现保障站坐标']
            assert result['运行来源']['策略入口'] == 'q4_scheme2'
        finally:
            server.shutdown()
            thread.join(timeout=5)
