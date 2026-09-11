"""停止已定位频道扫描：证书、跳过行为、未知覆盖和原版可切换性。"""
import math
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q3_local_env import LocalEnvironment, Source, make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_scheme2 import SchemeTwo, SchemeTwoConfig, layout_spec
from cumcm2026_b.q3_scheme2_localized import SchemeTwoLocalized, SchemeTwoLocalizedConfig


@pytest.mark.parametrize("layout", ["main14", "margin14", "compact15"])
@pytest.mark.parametrize("count,error", [(10, "plus"), (13, "minus"), (16, "alternating")])
def test_retained_certificate_skips_scans_but_still_clears_every_target(layout, count, error):
    env = make_case(500 + count, count, "boundary", "mixed", error)
    solver = SchemeTwoLocalized(RobotClient(env, "local-robot"), SchemeTwoLocalizedConfig(layout=layout))
    result = solver.run()
    assert result["运行成功"], result["异常"]
    truth = env.truth_for_evaluation()
    positions = {s["频道"]: s["位置"] for s in truth["目标"]}
    assert truth["全部清除"] and set(result["已清除频道"]) == set(positions)
    assert set(result["已证明不存在频道"]).isdisjoint(positions)
    events = result["动作记录"]
    boundary = next(e["步骤"] for e in events if e["动作"] == "固定扫描完成")
    frozen, positive_sites = {}, {}
    count_skips = 0
    for event in events:
        channel = event.get("频道")
        if event["动作"] == "检测" and event["步骤"] < boundary:
            assert channel not in frozen, "已定位频道再次进行固定检测"
            if event["结果"] == "direction":
                positive_sites.setdefault(channel, set()).add(tuple(event["位置"]))
        if event["动作"] == "定位完成":
            assert len(positive_sites[channel]) >= 2
            assert set(map(tuple, event["方向站点"])) == positive_sites[channel]
            vertices = np.asarray(event["外包顶点"])
            assert np.linalg.norm(vertices - event["覆盖圆心"], axis=1).max() <= 19.5
            assert min_distance(vertices, positions[channel]) <= 1e-5
            frozen[channel] = event
        if event["动作"] == "跳过固定检测":
            count_skips += 1
            assert event["证书步骤"] == frozen[channel]["步骤"] < event["步骤"] < boundary
        if event["步骤"] < boundary and event["动作"] in {"检测", "清除"}:
            assert min(math.dist(event["位置"], p) for p in result["布局"]["站点"]) < 1e-6
        if event["动作"] == "清除" and event["具有覆盖证书"]:
            assert event["结果"] == "success"
        if event.get("用途") in {"追加测向", "顺路检测", "有限方格覆盖"}:
            assert event["步骤"] > boundary
    assert count_skips == result["跳过已定位频道检测次数"]
    if layout in {"main14", "margin14"}:
        assert count_skips > 0
    for row in result["扫描阶段"].get("频道快照", []):
        assert row["状态"] != "unknown"
        if row["扫描状态"] == "localized":
            assert row["状态"] == "found" and row["不同方向站数"] >= 2
            assert row["可安全单点清除"]
    assert abs(sum(result["时间分项_秒"].values()) - result["虚拟总时间_秒"]) < .001
    requests = [e for e in env.history if e["path"] in {"/measure", "/clear"}]
    assert len(requests) == result["检测次数"] + result["清除成功次数"] + result["清除失败次数"]


def test_two_receipts_without_clear_precision_do_not_stop_scanning():
    env = LocalEnvironment([Source(1, (-149., 0.))], error_mode="zero")
    result = SchemeTwoLocalized(RobotClient(env, "local-robot"), SchemeTwoLocalizedConfig(scan_clear=False)).run()
    assert result["运行成功"] and env.truth_for_evaluation()["全部清除"]
    row = result["扫描阶段"]["频道快照"][0]
    assert row["不同方向站数"] == 2 and not row["可安全单点清除"]
    assert row["扫描状态"] == "found"
    assert not result["定位完成证书"]


@pytest.mark.parametrize("sites,vertices", [
    ([(0, 0)], [[0, 0], [1, 0], [0, 1]]),
    ([(0, 0), (0, 0)], [[0, 0], [1, 0], [0, 1]]),
    # 直径38米仍不保证可由19.5米圆覆盖：等边三角形外接圆半径38/sqrt(3)。
    ([(0, 0), (100, 100)], [[0, 0], [38, 0], [19, 19 * math.sqrt(3)]]),
])
def test_one_site_repeated_site_and_diameter_only_are_insufficient(sites, vertices):
    solver = SchemeTwoLocalized(RobotClient(LocalEnvironment([]), "local-robot"))
    state = solver.channels[1]
    state.status = "found"
    state.region.vertices = np.asarray(vertices, dtype=float)
    state.region.observations = [{"位置": list(p), "结果": "direction"} for p in sites]
    solver._mark_localized(state)
    assert solver.scan_status(state) == "found"
    assert not solver.complete()


def test_disabling_feature_matches_original_actions_and_costs():
    old_env = make_case(20260911, 10)
    new_env = make_case(20260911, 10)
    old = SchemeTwo(RobotClient(old_env, "local-robot"), SchemeTwoConfig()).run()
    new = SchemeTwoLocalized(RobotClient(new_env, "local-robot"), SchemeTwoLocalizedConfig(skip_localized=False)).run()
    keys = ["运行成功", "虚拟总时间_秒", "检测次数", "总路程_米", "频道切换次数", "已清除频道"]
    assert {k: old[k] for k in keys} == {k: new[k] for k in keys}
    def actions(result):
        return [(e["动作"], e["频道"], e["位置"], e["结果"]) for e in result["动作记录"] if e["动作"] in {"检测", "清除"}]
    assert actions(old) == actions(new)
    assert not new["定位完成证书"] and new["跳过已定位频道检测次数"] == 0


@pytest.mark.parametrize("scan_clear", [False, True])
def test_near_is_still_cleared_immediately(scan_clear):
    env = LocalEnvironment([Source(1, (0., 0.))])
    result = SchemeTwoLocalized(RobotClient(env, "local-robot"), SchemeTwoLocalizedConfig(scan_clear=scan_clear)).run()
    assert result["运行成功"]
    first = next(e for e in result["动作记录"] if e["动作"] == "清除")
    assert first["用途"] == "near原地清除" and first["位置"] == [0., 0.]
    assert result["已证明不存在频道"] == list(range(2, 21))


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_scan_station_clear_remains_available_for_localized_targets(enabled):
    station = np.array(layout_spec()["站点"][3])
    env = LocalEnvironment([Source(1, tuple(station + [6., 1.]), 1500)], error_mode="zero")
    result = SchemeTwoLocalized(RobotClient(env, "local-robot"), SchemeTwoLocalizedConfig(scan_clear=enabled)).run()
    assert result["运行成功"]
    clears = [e for e in result["动作记录"] if e.get("用途") == "固定扫描站原地证书清除"]
    assert bool(clears) == enabled
    if enabled:
        assert clears[0]["位置"] == station.tolist()


def test_budget_interruption_cannot_count_located_as_cleared():
    result = SchemeTwoLocalized(RobotClient(make_case(9, 10), "local-robot"),
                                SchemeTwoLocalizedConfig(max_actions=30)).run()
    assert not result["运行成功"] and not result["全部完成证据"]
    assert result["扫描阶段"] is None and result["正常退出"]
    assert "BudgetExceeded" in result["异常"]


@pytest.mark.parametrize("value", [1, "true", None])
def test_switch_must_be_boolean(value):
    with pytest.raises(ValueError, match="skip_localized"):
        SchemeTwoLocalizedConfig(skip_localized=value)
