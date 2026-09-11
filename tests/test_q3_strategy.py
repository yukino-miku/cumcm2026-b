"""策略仅从响应得到信息；运行后用独立真值核查完整性。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q3_local_env import make_case, LocalEnvironment, Source
from cumcm2026_b.q3_protocol import RobotClient, HttpTransport
from cumcm2026_b.q3_strategy import SchemeOne, StrategyConfig, ChannelState, observation_plan


@pytest.mark.parametrize("seed,count,layout,radius,error", [
    (1,10,"uniform","min","hash"), (2,13,"boundary","min","plus"),
    (3,16,"boundary","max","minus"), (4,10,"center","mixed","alternating"),
    (5,13,"cluster","mixed","zero"), (6,16,"uniform","min","hash")])
def test_complete_closed_loop_and_truth_stays_in_all_outer_regions(seed,count,layout,radius,error):
    env = make_case(seed,count,layout,radius,error)
    client = RobotClient(env,"local-robot")
    strategy = SchemeOne(client)
    result = strategy.run()
    assert result["运行成功"], result["异常"]
    truth = env.truth_for_evaluation()
    assert truth["全部清除"] and result["清除数"] == count
    sources = {s["频道"]:s["位置"] for s in truth["目标"]}
    for event in result["动作记录"]:
        if "外包顶点" in event:
            assert min_distance(event["外包顶点"],sources[event["频道"]]) <= 1e-5
        if event["动作"] == "清除" and event["具有覆盖证书"]:
            assert event["结果"] == "success"
            assert event["最远可能目标距离_米"] < 20
    assert set(result["已证明不存在频道"]).isdisjoint(sources)
    assert sum(result["时间分项_秒"].values()) == pytest.approx(result["虚拟总时间_秒"],abs=.001)


def test_finite_optical_fallback_really_completes_when_measurement_planning_disabled():
    env = LocalEnvironment([Source(1,(925.,0.))],error_mode="plus")
    result = SchemeOne(RobotClient(env,"local-robot"),StrategyConfig(max_localization_measurements=0,opportunistic=False)).run()
    assert result["运行成功"],result["异常"]
    assert env.truth_for_evaluation()["全部清除"]
    assert any(e["动作"] == "有限覆盖" for e in result["动作记录"])
    assert result["清除失败次数"] > 0
    assert result["虚拟总时间_秒"] < 360000


def test_count_ten_does_not_skip_missing_channel_certificates():
    env = make_case(7,10,"cluster","min","zero")
    result = SchemeOne(RobotClient(env,"local-robot"),StrategyConfig(opportunistic=False)).run()
    assert result["运行成功"]
    assert len(result["已证明不存在频道"]) == 10
    assert result["已完成搜索站"] == list(range(7))


def test_action_limit_reports_incomplete_instead_of_fake_completion():
    env = make_case(8,10)
    result = SchemeOne(RobotClient(env,"local-robot"),StrategyConfig(max_actions=3)).run()
    assert not result["运行成功"] and not result["全部完成证据"]
    assert result["正常退出"] and "BudgetExceeded" in result["异常"]


def test_clipped_wedge_with_center_ray_outside_circle_has_alternative_candidates():
    state=ChannelState(1,status="found")
    state.region.observe([1800,1000],"direction",271.)
    plan=observation_plan(state,np.array([1800.,1000.]),StrategyConfig())
    assert plan["动作"] in {"追加测向","覆盖清除"}


def test_real_http_transport_matches_documented_wire_protocol():
    env = LocalEnvironment([])
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.headers["Content-Type"] == "application/json"
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            assert not raw.startswith(b'\xef\xbb\xbf')
            payload = json.loads(raw)
            seen.append((self.path,payload))
            status,response = env.post(self.path,payload)
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self,*args): pass
    server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        client=RobotClient(HttpTransport(f"http://127.0.0.1:{server.server_port}"),"local-robot")
        client.enter()
        client.measure([300,400],2)
        client.clear([300,0],3)
        client.exit()
        assert client.state.virtual_time == 189
        assert [p for p,_ in seen] == ["/enter","/measure","/clear","/exit"]
        assert len({p['request_id'] for _,p in seen}) == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
