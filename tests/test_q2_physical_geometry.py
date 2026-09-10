"""第二问物理层的独立解析几何验证；不把待测函数输出作为真值。"""

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from cumcm2026_b.q2_physical_geometry import (
    analytic_candidates, build_physical_region, centerline_interval,
    reliable_bbox, reliable_boundary, reliable_membership, sample_region,
)


def test_standard_sector_exact_area_and_membership():
    region = build_physical_region([0, 0], 0)
    assert region.area == pytest.approx(1500**2 * math.pi / 180, abs=1e-7)
    assert centerline_interval(region) == pytest.approx((0, 1500))
    assert region.contains([100, 0])
    assert region.contains([0, 0])  # 首次五米区域明确保守保留。
    assert not region.contains([-1, 0])
    assert not region.contains([100, 10])
    assert not region.contains([1501, 0])


def test_segment_lens_analytic_intersections():
    region = build_physical_region([0, 0], 0)
    candidates = analytic_candidates(region)
    expected_height = math.sqrt(1000**2 - 750**2)
    np.testing.assert_allclose(candidates, [[750, expected_height], [750, -expected_height]], atol=1e-10)
    for candidate in candidates:
        assert np.linalg.norm(candidate) == pytest.approx(1000)
        assert np.linalg.norm(candidate - [1500, 0]) == pytest.approx(1000)
        assert not reliable_membership(region, candidate)
        far_endpoint = np.array([1500 * math.cos(math.pi / 180), -np.sign(candidate[1]) * 1500 * math.sin(math.pi / 180)])
        assert region.max_distance(candidate) == pytest.approx(np.linalg.norm(candidate - far_endpoint), abs=1e-8)


@pytest.mark.parametrize("rotation_deg", [23, 137, 291])
def test_analytic_and_continuous_geometry_rigid_transform(rotation_deg):
    angle = math.radians(rotation_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    shift = np.array([1234.0, -987.0])
    reference = build_physical_region([0, 0], 0)
    moved = build_physical_region(shift, rotation_deg, target_center=shift)
    np.testing.assert_allclose(analytic_candidates(moved), analytic_candidates(reference) @ rotation.T + shift, atol=1e-8)
    query = np.array([740.0, 620.0])
    transformed = query @ rotation.T + shift
    assert moved.max_distance(transformed) == pytest.approx(reference.max_distance(query), abs=1e-8)
    assert moved.min_distance(transformed) == pytest.approx(reference.min_distance(query), abs=1e-8)
    assert moved.area == pytest.approx(reference.area, abs=1e-7)


def test_arc_interior_distance_extrema_are_not_replaced_by_vertices():
    angle = math.radians(20)
    unit = np.array([math.cos(angle), math.sin(angle)])
    region = build_physical_region([0, 0], 20)
    # 20度圆弧中点不是边界连接点，检查连续径向驻点。
    assert np.min(np.linalg.norm(region.vertices - 1500 * unit, axis=1)) > 1
    assert region.max_distance(-500 * unit) == pytest.approx(2000, abs=1e-8)
    assert region.min_distance(2000 * unit) == pytest.approx(500, abs=1e-8)
    assert np.max(np.linalg.norm(region.vertices + 500 * unit, axis=1)) < 2000 - 0.01


def test_arc_interior_tangencies_determine_angular_extrema():
    region = build_physical_region([0, 0], 0)
    query = np.array([1500.1, 0.0])
    result = region.angular_interval(query)
    expected_half = math.asin(1500 / 1500.1)
    assert result["upper_rad"] - result["lower_rad"] == pytest.approx(2 * expected_half, abs=1e-10)
    for key in ("lower_point", "upper_point"):
        point = result[key]
        assert np.linalg.norm(point) == pytest.approx(1500, abs=1e-8)
        assert np.dot(point, point - query) == pytest.approx(0, abs=1e-6)
        assert region.contains(point)


def test_angular_interval_contains_independent_polar_sector_grid():
    region = build_physical_region([0, 0], 0)
    query = np.array([760.0, 640.0])
    result = region.angular_interval(query)
    radii = np.linspace(0, 1500, 37)
    angles = np.linspace(-math.pi / 180, math.pi / 180, 101)
    points = (radii[:, None, None] * np.column_stack([np.cos(angles), np.sin(angles)])[None, :, :]).reshape(-1, 2)
    toward = points - query
    observed = np.arctan2(toward[:, 1], toward[:, 0])
    assert observed.min() >= result["lower_rad"] - 1e-12
    assert observed.max() <= result["upper_rad"] + 1e-12
    for angle in np.linspace(result["lower_rad"], result["upper_rad"], 19):
        point = region.ray_intersection_point(query, [math.cos(angle), math.sin(angle)])
        assert region.contains(point, tol=1e-6)


@pytest.mark.parametrize("station,bearing", [([1200, 0], 0), ([1790, 0], 89), ([-700, 350], 20)])
def test_circle_clipped_area_against_independent_polar_integration(station, bearing):
    region = build_physical_region(station, bearing)
    s = np.asarray(station, dtype=float)

    def radial_square(angle):
        direction = np.array([math.cos(angle), math.sin(angle)])
        projection = float(s @ direction)
        far = -projection + math.sqrt(projection**2 + 1800**2 - float(s @ s))
        return min(1500.0, far)**2 / 2

    result, _ = quad(radial_square, math.radians(bearing - 1), math.radians(bearing + 1), epsabs=1e-6)
    assert region.area == pytest.approx(result, rel=1e-9, abs=1e-6)
    sampled = sample_region(region, boundary_count=80, radial_levels=3)
    assert region.contains(sampled, tol=1e-6).all()
    assert np.linalg.norm(sampled, axis=1).max() <= 1800 + 1e-6


def test_reliable_region_independent_three_disk_characterization():
    region = build_physical_region([0, 0], 0)
    delta = math.pi / 180
    centers = np.array([[0, 0], [1500 * math.cos(delta), 1500 * math.sin(delta)], [1500 * math.cos(delta), -1500 * math.sin(delta)]])
    # 对该标准窄扇形，K内查询点必在前半平面；弧上远距驻点在反向，
    # 因而只需原点与两个远端三个圆。本真值不依赖连续边界实现。
    points = np.array([[x, y] for x in np.linspace(400, 1100, 17) for y in np.linspace(-800, 800, 19)])
    expected = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2).max(axis=1) <= 1000 - 1e-6
    actual = reliable_membership(region, points)
    np.testing.assert_array_equal(actual, expected)
    assert not reliable_membership(region, [0, 0])
    assert region.max_distance([0, 0]) == pytest.approx(1500)


def test_reliable_boundary_points_pass_independent_dense_sector_grid():
    region = build_physical_region([0, 0], 0)
    boundary = reliable_boundary(region, count=24)
    angles = np.linspace(-math.pi / 180, math.pi / 180, 2001)
    targets = np.vstack([[0, 0], 1500 * np.column_stack([np.cos(angles), np.sin(angles)])])
    maximum = np.linalg.norm(boundary[:, None, :] - targets[None, :, :], axis=2).max(axis=1)
    assert maximum.max() <= 1000 - 0.5e-6
    np.testing.assert_allclose(maximum, 1000 - 1e-6, atol=2e-7)
    bbox = reliable_bbox(region)
    assert np.all(boundary >= bbox[0] - 1e-6)
    assert np.all(boundary <= bbox[1] + 1e-6)


def test_reception_and_bearing_guarantees_are_distinct():
    region = build_physical_region([0, 0], 0)
    assert reliable_membership(region, [750, 0])
    assert not reliable_membership(region, [750, 0], require_bearing=True)
    assert region.min_distance([750, 0]) == 0
    assert reliable_membership(region, [750, 600], require_bearing=True)
    with pytest.raises(ValueError, match="严格位于"):
        region.angular_interval([750, 0])


def test_empty_and_tangent_physical_regions_are_explicit():
    empty = build_physical_region([5000, 0], 0)
    assert empty.is_empty
    assert empty.area == 0
    assert centerline_interval(empty) is None
    with pytest.raises(ValueError, match="为空"):
        empty.max_distance([0, 0])
    tangent = build_physical_region([3300, 0], 180)
    assert not tangent.is_empty
    assert tangent.area == pytest.approx(0, abs=1e-7)
    np.testing.assert_allclose(tangent.representative_point, [1800, 0], atol=1e-7)
    assert tangent.max_distance([0, 0]) == pytest.approx(1800, abs=1e-7)


def test_strict_json_round_trip_and_no_nonfinite_numbers():
    region = build_physical_region([1790, 0], 89)
    restored = json.loads(json.dumps(region.to_dict(), ensure_ascii=False, allow_nan=False))
    assert restored["target_radius"] == 1800
    assert restored["reception_upper"] == 1500
    assert len(restored["arcs"]) >= 1
    assert restored["area_m2"] > 0


@pytest.mark.parametrize("kwargs", [{"half_width_deg": 0}, {"target_radius": -1}, {"reception_upper": float("inf")}, {"tolerance": 0}])
def test_invalid_physical_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_physical_region([0, 0], 0, **kwargs)
