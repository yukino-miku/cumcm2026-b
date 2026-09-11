"""顺路清除、固定站顺序、预测位置与安全清除分离，以及原版回退验证。"""
import math
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q3_local_env import LocalEnvironment, Source, make_case
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q3_strategy import SchemeOne, ChannelState
from cumcm2026_b.q3_scheme2 import SchemeTwo
from cumcm2026_b.q3_route_planning import plan_route
from cumcm2026_b.q3_routed_strategies import SchemeOneRouted, SchemeTwoRouted, RoutedOneConfig, RoutedTwoConfig


def target(channel, xy, size=1):
    state = ChannelState(channel)
    state.status = "found"
    state.region.vertices = np.asarray(xy) + np.array([[-size, -size], [size, -size], [0, size]])
    return state


def test_a_target_on_later_leg_is_deferred_without_reordering_search_stations():
    fixed = [[1000., 0.], [1000., 1000.]]
    state = target(1, [1000, 900])
    plan = plan_route([state], [0, 0], fixed)
    assert plan["计划路径"][1]["类型"] == "固定站"
    assert [n["位置"] for n in plan["计划路径"] if n["类型"] == "固定站"] == fixed
    node = next(n for n in plan["计划路径"] if n["类型"] == "目标")
    assert np.linalg.norm(state.region.vertices - node["位置"], axis=1).max() <= 19.5 + 1e-7


def test_safe_clear_can_be_inserted_on_current_leg_without_extra_travel():
    state = target(1, [400, 10])
    plan = plan_route([state], [0, 0], [[1000, 0]])
    assert plan["计划路径"][1]["类型"] == "目标"
    assert plan["预计几何路程_米"] == pytest.approx(1000, abs=1e-6)
    point = plan["计划路径"][1]["位置"]
    assert abs(point[1]) < 1e-7
    assert np.linalg.norm(state.region.vertices - point, axis=1).max() <= 19.5 + 1e-7


def test_uncertain_target_center_is_never_marked_safe_to_clear():
    plan = plan_route([target(1, [400, 10], 100)], [0, 0], [[1000, 0]])
    assert not next(n for n in plan["计划路径"] if n["类型"] == "目标")["可安全清除"]


@pytest.mark.parametrize("solver,config", [(SchemeOneRouted, RoutedOneConfig), (SchemeTwoRouted, RoutedTwoConfig)])
@pytest.mark.parametrize("count,layout,error", [(10, "uniform", "hash"), (13, "boundary", "plus"), (16, "cluster", "minus")])
def test_routed_closed_loop_clear_certificates_and_original_fixed_order(solver, config, count, layout, error):
    env = make_case(20260911, count, layout, "mixed", error)
    result = solver(RobotClient(env, "local-robot"), config()).run()
    assert result["运行成功"], result["异常"]
    truth = env.truth_for_evaluation()
    positions = {t["频道"]: t["位置"] for t in truth["目标"]}
    assert truth["全部清除"] and set(result["已清除频道"]) == set(positions)
    assert set(result["已证明不存在频道"]).isdisjoint(positions)
    visited = result["已完成搜索站"]
    assert visited == list(range(len(visited)))
    assert result["路线调度统计"]["重规划次数"] > 0
    for e in result["动作记录"]:
        if "外包顶点" in e:
            assert min_distance(e["外包顶点"], positions[e["频道"]]) < 1e-5
        if e["动作"] == "清除" and e["具有覆盖证书"]:
            assert e["结果"] == "success" and math.dist(e["位置"], positions[e["频道"]]) <= 20
        if e["动作"] == "路线规划":
            fixed = [n["剩余站序"] for n in e["计划路径"] if n["类型"] == "固定站"]
            assert fixed == sorted(fixed)
    assert abs(sum(result["时间分项_秒"].values()) - result["虚拟总时间_秒"]) < .001


def test_scheme_two_actually_clears_on_the_way_before_last_fixed_station():
    result = SchemeTwoRouted(RobotClient(make_case(20260911, 10), "local-robot")).run()
    assert result["运行成功"]
    events = result["动作记录"]
    end = next(e["步骤"] for e in events if e["动作"] == "固定扫描完成")
    clears = [e for e in events if e.get("用途") == "顺路证书清除"]
    assert clears and all(e["步骤"] < end for e in clears)
    assert any(e["动作"] == "开始固定站" and e["步骤"] > clears[0]["步骤"] for e in events)


@pytest.mark.parametrize("original,routed,config", [
    (SchemeOne, SchemeOneRouted, RoutedOneConfig(route_planning=False)),
    (SchemeTwo, SchemeTwoRouted, RoutedTwoConfig(route_planning=False, skip_localized=False)),
])
def test_all_optimization_switches_off_preserves_original_behavior(original, routed, config):
    a = original(RobotClient(make_case(12, 10), "local-robot")).run()
    b = routed(RobotClient(make_case(12, 10), "local-robot"), config).run()
    keys = ["运行成功", "虚拟总时间_秒", "总路程_米", "检测次数", "已清除频道"]
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    assert b["路线调度统计"]["重规划次数"] == 0


def test_route_only_scheme_two_works_independently_of_measurement_skip_switch():
    result = SchemeTwoRouted(RobotClient(make_case(20260911, 10), "local-robot"),
                             RoutedTwoConfig(skip_localized=False)).run()
    assert result["运行成功"] and result["路线调度统计"]["顺路证书清除次数"] > 0
    assert not result["停止已定位频道扫描"] and not result["定位完成证书"]
    assert "已定位频道停止检测" not in result["方案"]


@pytest.mark.parametrize("solver,config", [(SchemeOneRouted, RoutedOneConfig), (SchemeTwoRouted, RoutedTwoConfig)])
def test_route_budget_failure_is_not_reported_as_complete(solver, config):
    result = solver(RobotClient(make_case(9, 10), "local-robot"), config(max_actions=30)).run()
    assert not result["运行成功"] and not result["全部完成证据"] and result["正常退出"]


@pytest.mark.parametrize("solver,config", [(SchemeOneRouted, RoutedOneConfig), (SchemeTwoRouted, RoutedTwoConfig)])
def test_deferred_uncertain_target_has_final_finite_sweep_fallback(solver, config):
    env = LocalEnvironment([Source(1, (-149., 0.))], error_mode="zero")
    result = solver(RobotClient(env, "local-robot"), config(max_localization_measurements=0)).run()
    assert result["运行成功"] and env.truth_for_evaluation()["全部清除"]
    assert any(e["动作"] == "有限覆盖" for e in result["动作记录"])
