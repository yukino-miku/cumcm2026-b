"""第二问角域鲁棒选点的解析性质与第一问独立交叉验证。"""

import json

import numpy as np
import pytest

from cumcm2026_b.q1_geometry import solve_bearings
from cumcm2026_b.q2_physical_geometry import build_physical_region, sample_region
from cumcm2026_b.q2_active_localization import (
    batch_two_station_diameters, candidate_region_statistics, evaluate_detection,
    linearized_diameter, robust_objective, sampled_robust_objective,
    search_second_station, serialize_result,
)


@pytest.fixture
def standard():
    return build_physical_region([0.0, 0.0], 0.0)


def test_batch_orthogonal_case_matches_independent_closed_formula():
    diameter, kind = batch_two_station_diameters([-100, 0], 0, [0, -100], [90])
    tangent = np.tan(np.deg2rad(1.0))
    assert kind[0] == "bounded"
    assert diameter[0] == pytest.approx(200 * np.sqrt(2) * tangent / (1 - tangent ** 2))


@pytest.mark.parametrize("rotation", [0.0, 37.0, 270.0])
def test_batch_matches_first_question_for_feasible_and_empty_observations(rotation):
    radians = np.deg2rad(rotation)
    matrix = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    stations = np.array([[0.0, 0.0], [750.0, 600.0]]) @ matrix.T + [34500, -61700]
    angles = np.array([-179, -135, -89, -45, -1, 0, 1, 45, 89, 135, 179]) + rotation
    diameters, kinds = batch_two_station_diameters(stations[0], rotation, stations[1], angles)
    for index, angle in enumerate(angles):
        checked = solve_bearings(stations, [rotation, angle])
        if checked.kind in {"empty", "unbounded"}:
            assert kinds[index] == checked.kind
        else:
            assert kinds[index] == "bounded"
            assert diameters[index] == pytest.approx(checked.diameter, rel=1e-7, abs=1e-7)


def test_batch_near_parallel_does_not_clip_distant_vertex():
    values, states = batch_two_station_diameters([0, 0], 1.00005, [0, 100], [-1.00005])
    far_x = 50 / np.tan(np.deg2rad(0.00005))
    near_x = 50 / np.tan(np.deg2rad(2.00005))
    assert states[0] == "bounded"
    assert values[0] == pytest.approx(far_x - near_x, rel=1e-8)


def test_single_scenario_calls_original_geometric_model(standard):
    result = evaluate_detection(standard, [750, 630], [1000, 0], 1.0)
    theta = np.rad2deg(np.arctan2(-630, 250)) + 1
    checked = solve_bearings([[0, 0], [750, 630]], [0, theta])
    assert result.diameter == pytest.approx(checked.diameter)
    # 定位多边形不会为了物理先验而被目标圆或最大接收圆截断。
    far_result = evaluate_detection(standard, [-200, 0], [1000, 0])
    assert far_result.kind == "unbounded"


def test_standard_mirror_symmetry_and_final_first_question_verification(standard):
    plus = robust_objective(standard, [750, 630], verify_q1=True)
    minus = robust_objective(standard, [750, -630], verify_q1=True)
    assert plus["status"] == minus["status"] == "finite"
    assert plus["diameter_m"] == pytest.approx(minus["diameter_m"], rel=1e-9)
    assert plus["reliable_bearing"] and minus["reliable_bearing"]
    assert plus["q1_verified"] and minus["q1_verified"]
    assert standard.contains(plus["witness_target"])
    assert abs(plus["witness_error_deg"]) <= 1 + 1e-10


def test_parallel_baseline_is_unbounded_but_side_station_is_finite(standard):
    parallel = robust_objective(standard, [-200, 0], verify_q1=True)
    side = robust_objective(standard, [750, 630])
    assert parallel["status"] == "unbounded"
    assert np.isinf(parallel["diameter_m"])
    assert np.isfinite(side["diameter_m"])
    assert not parallel["reliable_detection"]


def test_inside_target_domain_does_not_pretend_guaranteed_direction(standard):
    result = robust_objective(standard, [750, 0])
    assert result["reliable_detection"]
    assert not result["reliable_bearing"]
    assert result["status"] == "direction_undefined"
    assert np.isinf(result["diameter_m"])


def test_analytic_centerline_point_is_not_reliable_for_full_wedge(standard):
    height = np.sqrt(1000.0 ** 2 - 750.0 ** 2)
    point = np.array([750, height])
    target = 1500 * np.array([np.cos(np.deg2rad(1)), -np.sin(np.deg2rad(1))])
    assert np.linalg.norm(point - target) > 1000
    result = robust_objective(standard, point)
    assert not result["reliable_detection"]
    assert result["max_distance_m"] == pytest.approx(np.linalg.norm(point - target))


def test_full_target_error_grid_agrees_with_direction_reduction(standard):
    station = [750, 630]
    targets = sample_region(standard, boundary_count=128, radial_levels=4)
    discrete = sampled_robust_objective(standard, station, targets, error_count=9)
    continuous_angle = robust_objective(standard, station, angle_count=257, verify_q1=True)
    assert discrete["scenario_count"] == len(targets) * 9
    assert discrete["diameter_m"] <= continuous_angle["diameter_m"] + 1e-7
    assert discrete["diameter_m"] == pytest.approx(continuous_angle["diameter_m"], rel=1e-7)
    independently_checked = evaluate_detection(standard, station, discrete["witness_target"], discrete["witness_error_deg"])
    assert independently_checked.diameter == pytest.approx(discrete["diameter_m"])


def test_nested_observation_grids_stabilize_without_claiming_global_proof(standard):
    values = [robust_objective(standard, [750, 630], angle_count=count, refine=False)["diameter_m"]
              for count in (17, 33, 65, 129)]
    assert np.min(np.diff(values)) >= -1e-8
    assert values[-1] == pytest.approx(values[-2], rel=1e-5)


def test_wider_bounded_errors_do_not_reduce_fixed_strategy_worst_diameter():
    values = [robust_objective(build_physical_region([0, 0], 0, width), [750, 600], angle_count=129)["diameter_m"]
              for width in (0.5, 1.0, 1.5, 2.0)]
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("phi", [30.0, 60.0, 90.0, 120.0, 150.0])
def test_linearized_formula_matches_four_independently_constructed_strip_vertices(phi):
    w1, w2 = np.array([700.0, 1000.0]) * np.tan(np.deg2rad(1))
    radians = np.deg2rad(phi)
    normals = [[0, 1], [-np.sin(radians), np.cos(radians)]]
    vertices = np.array([np.linalg.solve(normals, [i * w1, j * w2]) for i in (-1, 1) for j in (-1, 1)])
    known = np.max(np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2))
    assert linearized_diameter(700, 1000, phi) == pytest.approx(known)


def test_candidate_statistics_excludes_irregular_local_points_from_area():
    records = [dict(reliable_bearing=True, diameter_m=value, movement_distance_m=move)
               for value, move in [(10, 500), (10.1, 550), (10.2, 600), (30, 700)]]
    grid = np.array([[10.0, np.inf], [10.4, 30.0]])
    stats = candidate_region_statistics(records, 10.0, grid, 25.0, (0.05,))["0.05"]
    assert stats["point_count"] == 3
    assert stats["grid_count"] == 2
    assert stats["estimated_area_m2"] == 1250.0
    assert stats["movement_range_m"] == [500, 600]


def test_small_two_level_search_produces_reliable_symmetric_candidates(standard):
    result = search_second_station(standard, grid_step=100, angle_count=33,
                                   local_steps=(20, 4), use_slsqp=False, final_angle_count=65)
    assert len(result["side_optima"]) == 2
    assert result["best"]["reliable_bearing"]
    assert result["best"]["q1_verified"]
    assert result["side_optima"][0]["diameter_m"] == pytest.approx(result["side_optima"][1]["diameter_m"], rel=1e-8)
    assert result["candidate_regions"]["0.05"]["point_count"] >= 2
    for index in result["candidate_regions"]["0.10"]["point_indices"]:
        assert result["evaluations"][index]["angle_count"] == 65
        assert result["evaluations"][index]["refine"] is True
    targets = sample_region(standard, boundary_count=512, radial_levels=3)
    for optimum in result["side_optima"]:
        assert np.max(np.linalg.norm(targets - optimum["station"], axis=1)) < 1000
        assert np.min(np.linalg.norm(targets - optimum["station"], axis=1)) > 5
    dumped = json.dumps(serialize_result(result), ensure_ascii=False, allow_nan=False)
    assert "Infinity" not in dumped and "NaN" not in dumped


def test_grazing_nonempty_region_without_centerline_intersection_remains_searchable():
    # 中心线y=1810在1800圆外，但向下1度边界擦入圆内，不能以中心线代替扇区。
    region = build_physical_region([-900, 1810], 0)
    assert not region.is_empty
    result = search_second_station(region, grid_step=200, angle_count=17,
                                   local_steps=(20,), use_slsqp=False, final_angle_count=65)
    assert result["analytic_status"] == "centerline_misses_target_domain"
    assert result["analytic_candidates"].shape == (0, 2)
    assert result["best"]["reliable_bearing"]


def test_invalid_target_error_and_sampling_inputs_fail_loudly(standard):
    with pytest.raises(ValueError):
        evaluate_detection(standard, [750, 600], [0, 1500])
    with pytest.raises(ValueError):
        evaluate_detection(standard, [750, 600], [1000, 0], 1.1)
    with pytest.raises(ValueError):
        robust_objective(standard, [750, 600], angle_count=2)
    with pytest.raises(ValueError):
        sampled_robust_objective(standard, [750, 600], [[0, 1500]])
def test_clipped_tip_does_not_admit_tolerance_only_vertex():
    """截断案例楔尖附近：用独立原函数防止接收约束外的伪顶点。"""
    from cumcm2026_b.q1_geometry import solve_bearings
    from cumcm2026_b.q2_active_localization import batch_two_station_diameters

    stations = [[1200.0, 0.0], [1671.0, -461.0]]
    angle = 134.61473901480807
    exact = solve_bearings(stations, [0.0, angle])
    values, states = batch_two_station_diameters(stations[0], 0.0, stations[1], [angle])
    assert exact.kind == "polygon" and states[0] == "bounded"
    assert abs(values[0] - exact.diameter) < 1e-7
