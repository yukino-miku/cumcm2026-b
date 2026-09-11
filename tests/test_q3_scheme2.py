"""双覆盖扫描的独立几何、阶段顺序、同反馈反例与闭环验证。"""
import math
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q3_local_env import LocalEnvironment, Source, make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_scheme2 import SchemeTwo, SchemeTwoConfig, layout_spec


@pytest.mark.parametrize("layout", ["main14", "margin14", "compact15"])
def test_continuous_partition_bound_and_independent_polar_grid(layout):
    spec = layout_spec(layout)
    stations = np.array(spec["站点"])
    angles = np.linspace(0, 2*np.pi, 721)
    points = (np.linspace(0, 1800, 73)[:, None, None]*np.stack((np.cos(angles), np.sin(angles)), axis=1)).reshape(-1, 2)
    distances = np.linalg.norm(points[:, None]-stations[None], axis=2)
    second = np.partition(distances, 1, axis=1)[:, 1]
    assert second.max() <= spec["第二近距离连续上界_米"]+1e-8
    assert np.min(np.sum(distances <= 1000, axis=1)) >= 2
    if layout == "main14":
        assert second.max() == pytest.approx(988.511420, abs=1e-6)
        assert spec["全频道固定扫描时间_秒"] == pytest.approx(3205.624368, abs=1e-6)


@pytest.mark.parametrize("layout", ["main14", "margin14", "compact15"])
@pytest.mark.parametrize("count,error", [(10,"plus"),(13,"minus"),(16,"alternating")])
def test_closed_loop_and_unfinished_sources_have_two_distinct_scan_stations(layout,count,error):
    env = make_case(300+count,count,"boundary","min",error)
    result = SchemeTwo(RobotClient(env,"local-robot"),SchemeTwoConfig(layout=layout)).run()
    assert result["运行成功"], result["异常"]
    truth = env.truth_for_evaluation()
    positions = {s["频道"]:s["位置"] for s in truth["目标"]}
    assert truth["全部清除"] and result["清除数"] == count
    scan = result["扫描阶段"]
    for row in scan["频道快照"]:
        if row["状态"] == "found": assert row["不同方向站数"] >= 2
        assert row["状态"] != "unknown"
    events = result["动作记录"]
    end = next(e["步骤"] for e in events if e["动作"] == "固定扫描完成")
    fixed = np.asarray(result["布局"]["站点"])
    for event in events:
        if event["步骤"] < end and event["动作"] in {"检测","清除"}:
            assert np.linalg.norm(fixed-event["位置"],axis=1).min() < 1e-7
        if event.get("用途") in {"追加测向", "顺路检测", "有限方格覆盖"}:
            assert event["步骤"] > end
        if "外包顶点" in event:
            assert min_distance(event["外包顶点"],positions[event["频道"]]) < 1e-5
        if event["动作"] == "清除" and event["具有覆盖证书"]:
            assert event["结果"] == "success"
    assert set(result["已证明不存在频道"]).isdisjoint(positions)
    assert sum(result["时间分项_秒"].values()) == pytest.approx(result["虚拟总时间_秒"],abs=.001)


def test_full_scan_accounting_without_skips_matches_analytic_baseline():
    class FullScan(SchemeTwo):
        def _measure(self, state, point, reason):
            result = super()._measure(state,point,reason)
            # 仅计费单元测试禁止提前排除，强制保留全频道扫描基准。
            if state.status == "absent": state.status = "unknown"
            return result
        def _finish_scan(self):
            assert self.client.state.measurements == 280
            assert self.client.state.switches == 266
            assert self.client.state.virtual_time == pytest.approx(3205.624368,abs=1e-5)
            for state in self.channels.values(): state.status = "absent"
            super()._finish_scan()
    result = FullScan(RobotClient(LocalEnvironment([]),"local-robot")).run()
    assert result["运行成功"],result["异常"]


def test_same_scan_feedback_counterexample_needs_further_actions():
    prefixes = []
    for x in (-149., -6.):
        env = LocalEnvironment([Source(1,(x,0.))],error_mode="zero")
        result = SchemeTwo(RobotClient(env,"local-robot"),SchemeTwoConfig(scan_clear=False,opportunistic=False)).run()
        assert result["运行成功"] and env.truth_for_evaluation()["全部清除"]
        target = result["扫描阶段"]["频道快照"][0]
        assert target["不同方向站数"] == 2
        assert not target["可安全单点清除"]
        observations = [e for e in result["动作记录"] if e.get("频道") == 1 and e.get("用途", "").startswith("固定扫描站")]
        prefixes.append([(e["位置"],e["结果"],e["示向度"]) for e in observations])
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("scan_clear", [True,False])
def test_near_always_clears_without_moving_even_if_optional_scan_clear_is_off(scan_clear):
    result = SchemeTwo(RobotClient(LocalEnvironment([Source(1,(0.,0.))]),"local-robot"),SchemeTwoConfig(scan_clear=scan_clear)).run()
    assert result["运行成功"]
    first = next(e for e in result["动作记录"] if e["动作"] == "清除")
    assert first["位置"] == [0.,0.] and first["用途"] == "near原地清除"


def test_scan_budget_interruption_is_not_reported_as_scan_or_mission_complete():
    result = SchemeTwo(RobotClient(make_case(9,10),"local-robot"),SchemeTwoConfig(max_actions=3)).run()
    assert not result["运行成功"] and not result["全部完成证据"]
    assert result["扫描阶段"] is None and result["正常退出"]
    assert "BudgetExceeded" in result["异常"]


def test_incomplete_scan_receipt_certificate_is_detected():
    solver = SchemeTwo(RobotClient(LocalEnvironment([]),"local-robot"))
    for s in solver.channels.values(): s.status = "absent"
    state = solver.channels[1]
    state.status = "found"
    state.region.observe([0,0],"direction",45)
    with pytest.raises(RuntimeError, match="不足两个"):
        solver._finish_scan()
