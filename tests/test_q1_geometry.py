"""第一问关键数学性质与边界情况验证；不依赖附件模拟器。"""

import json
from pathlib import Path

import numpy as np
import pytest

from cumcm2026_b.q1_geometry import bearing_halfplanes, solve_bearings, solve_halfplanes


def equilateral_case(side=20.0):
    triangle = np.array([[0.0, 0.0], [side, 0.0], [side / 2.0, side * np.sqrt(3) / 2.0]])
    edges = np.roll(triangle, -1, axis=0) - triangle
    stations = triangle - 25.0 * edges
    bearings = np.rad2deg(np.arctan2(edges[:, 1], edges[:, 0])) + 1.0
    return triangle, stations, bearings


def assert_same_vertex_set(actual, expected, atol=1e-7):
    assert len(actual) == len(expected)
    for vertex in expected:
        assert np.min(np.linalg.norm(actual - vertex, axis=1)) <= atol


def test_equilateral_counterexample_is_realizable_with_one_degree_error():
    triangle, stations, bearings = equilateral_case()
    result = solve_bearings(stations, bearings)
    assert result.kind == "polygon"
    assert_same_vertex_set(result.vertices, triangle)
    assert result.diameter == pytest.approx(20.0, abs=1e-8)
    assert result.cover_radius == pytest.approx(10.0, abs=1e-8)
    assert result.max_center_distance == pytest.approx(10.0 * np.sqrt(3), abs=1e-8)
    assert result.coverage_excess == pytest.approx(10.0 * (np.sqrt(3) - 1.0), abs=1e-8)
    assert result.covered is False
    assert np.all(result.A @ triangle.T <= result.b[:, None] + 1e-9)
    # 三角形最小覆盖圆半径大于半直径，验证反例不是中心选择错误。
    assert 20.0 / np.sqrt(3) > result.cover_radius


def test_orthogonal_two_stations_matches_analytic_vertices_and_diameter():
    result = solve_bearings([[-100, 0], [0, -100]], [0, 90])
    t = np.tan(np.deg2rad(1.0))
    expected = []
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            # x = sign_x*t*(y+100), y = sign_y*t*(x+100)。
            matrix = [[1.0, -sign_x * t], [-sign_y * t, 1.0]]
            rhs = [sign_x * t * 100.0, sign_y * t * 100.0]
            expected.append(np.linalg.solve(matrix, rhs))
    assert result.kind == "polygon"
    assert_same_vertex_set(result.vertices, np.asarray(expected))
    assert result.diameter == pytest.approx(200 * np.sqrt(2) * t / (1 - t * t))
    assert result.covered is True


def test_parallel_bearings_remain_unbounded_without_artificial_box():
    result = solve_bearings([[0, 0], [0, 10]], [0, 0])
    assert result.kind == "unbounded"
    assert result.diameter is None
    assert result.covered is None
    assert len(result.vertices) == 0


def test_nearly_parallel_boundaries_keep_far_but_finite_vertex():
    # 两条外侧边界只差 0.0001 度，解析远顶点约为 5729 万米。
    # 这一人为极端案例检验有界性判定与求交，不能用固定大框替代。
    result = solve_bearings([[0, 0], [0, 100]], [1.00005, -1.00005])
    far_x = 50.0 / np.tan(np.deg2rad(0.00005))
    near_x = 50.0 / np.tan(np.deg2rad(2.00005))
    assert result.kind == "polygon"
    assert float(np.max(result.vertices[:, 0])) == pytest.approx(far_x, rel=1e-8)
    assert result.diameter == pytest.approx(far_x - near_x, rel=1e-8)
    assert np.max(result.A @ result.vertices.T - result.b[:, None]) <= result.absolute_tolerance


def test_opposite_outward_bearings_are_empty():
    assert solve_bearings([[0, 0], [-10, 0]], [0, 180]).kind == "empty"


@pytest.mark.parametrize("angles", [[360, 450], [-360, -270], [720, 810]])
def test_angle_wraparound(angles):
    baseline = solve_bearings([[-100, 0], [0, -100]], [0, 90])
    actual = solve_bearings([[-100, 0], [0, -100]], angles)
    assert_same_vertex_set(actual.vertices, baseline.vertices)
    assert actual.diameter == pytest.approx(baseline.diameter)


@pytest.mark.parametrize("factor", [0.001, 1.0, 10000.0])
@pytest.mark.parametrize("angle_deg", [0.0, 37.0, 153.0])
def test_translation_rotation_and_scale_invariance(factor, angle_deg):
    triangle, stations, bearings = equilateral_case()
    radians = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    shift = np.array([125000.0, -87000.0])
    transformed_stations = factor * stations @ rotation.T + shift
    result = solve_bearings(transformed_stations, bearings + angle_deg)
    expected = factor * triangle @ rotation.T + shift
    assert result.kind == "polygon"
    assert_same_vertex_set(result.vertices, expected, atol=max(1e-7, factor * 1e-7))
    assert result.diameter == pytest.approx(20.0 * factor, rel=1e-7, abs=1e-8)
    assert result.covered is False


def test_segment_and_point_are_supported_without_qhull_failure():
    A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    segment = solve_halfplanes(A, [3, -1, 2, -2])
    assert segment.kind == "segment"
    assert_same_vertex_set(segment.vertices, np.array([[1.0, 2.0], [3.0, 2.0]]))
    assert segment.diameter == pytest.approx(2.0)
    assert segment.covered is True
    point = solve_halfplanes(A, [1, -1, 2, -2])
    assert point.kind == "point"
    assert_same_vertex_set(point.vertices, np.array([[1.0, 2.0]]))
    assert point.diameter == 0.0
    assert point.cover_radius == 0.0
    assert point.covered is True


def test_negative_coordinate_region_uses_free_lp_bounds():
    result = solve_halfplanes([[1, 0], [-1, 0], [0, 1], [0, -1]], [-10, 12, -20, 23], origin=[0, 0], scale=1)
    assert result.kind == "polygon"
    assert result.diameter == pytest.approx(np.sqrt(13))
    assert_same_vertex_set(result.vertices, np.array([[-12, -23], [-10, -23], [-10, -20], [-12, -20]]))
    assert result.covered is True


def test_unbounded_line_and_no_observations():
    assert solve_halfplanes([[1, 0], [-1, 0]], [2, -2]).kind == "unbounded"
    assert solve_halfplanes([], []).kind == "unbounded"
    assert solve_bearings([], []).kind == "unbounded"


def test_zero_constraint_row_and_row_scaling():
    assert solve_halfplanes([[0, 0]], [-1]).kind == "empty"
    assert solve_halfplanes([[0, 0]], [0]).kind == "unbounded"
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    b = np.array([3, -1, 4, -2], dtype=float)
    baseline = solve_halfplanes(A, b)
    factors = np.array([1e-6, 10.0, 3.0, 1e6])
    scaled = solve_halfplanes(A * factors[:, None], b * factors)
    assert_same_vertex_set(baseline.vertices, scaled.vertices)
    assert scaled.diameter == pytest.approx(baseline.diameter)


def test_redundant_constraints_do_not_change_vertices():
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 0], [0, -1]], dtype=float)
    b = np.array([1, 1, 1, 1, 2, 1], dtype=float)
    result = solve_halfplanes(A, b)
    assert result.kind == "polygon"
    assert len(result.vertices) == 4
    assert result.diameter == pytest.approx(2.0 * np.sqrt(2))


@pytest.mark.parametrize("half_width", [0, -1, 90, 100, float("nan")])
def test_invalid_half_width_is_rejected(half_width):
    with pytest.raises(ValueError):
        solve_bearings([[0, 0]], [0], half_width)


def test_input_dimensions_and_nonfinite_values_are_rejected():
    with pytest.raises(ValueError):
        bearing_halfplanes([[0, 0]], [0, 90])
    with pytest.raises(ValueError):
        solve_bearings([[float("inf"), 0]], [0])
    with pytest.raises(ValueError):
        solve_halfplanes([[1, 0]], [])


def test_registered_cases_and_json_serialization():
    cases = json.loads((Path(__file__).resolve().parents[1] / "configs/q1_cases.json").read_text(encoding="utf-8"))
    expected = ["polygon", "polygon", "unbounded", "empty"]
    for case, kind in zip(cases["cases"], expected, strict=True):
        result = solve_bearings(case["stations"], case["bearings_deg"], case["half_width_deg"])
        assert result.kind == kind
        serialized = json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
        assert result.kind in serialized
