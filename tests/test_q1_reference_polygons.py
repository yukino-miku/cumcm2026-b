"""用具有解析答案的正多边形独立检验角域求交结果。"""

import numpy as np
import pytest

from cumcm2026_b.q1_geometry import solve_bearings


@pytest.mark.parametrize("count", [3, 4, 5, 8])
@pytest.mark.parametrize("rotation_deg", [0.0, 23.0, 137.0])
def test_regular_polygon_has_analytic_diameter_and_coverage(count, rotation_deg):
    """支撑边角域生成已知多边形，真值来自解析几何而非待测算法。"""
    radius = 2.0
    center = np.array([130.0, -80.0])
    angles = np.deg2rad(rotation_deg) + np.arange(count) * 2.0 * np.pi / count
    vertices = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    edges = np.roll(vertices, -1, axis=0) - vertices
    stations = vertices - 100.0 * edges
    bearings = np.rad2deg(np.arctan2(edges[:, 1], edges[:, 0])) + 1.0

    # 独立核实角域另一侧不截掉任何真值顶点，保证构造前提成立。
    for station, edge in zip(stations, edges):
        toward = vertices - station
        signed_angles = np.arctan2(
            edge[0] * toward[:, 1] - edge[1] * toward[:, 0], toward @ edge
        )
        assert signed_angles.min() >= -1e-12
        assert signed_angles.max() <= np.deg2rad(2.0) + 1e-12

    result = solve_bearings(stations, bearings)
    assert result.kind == "polygon"
    assert len(result.vertices) == count
    distances = np.linalg.norm(result.vertices[:, None, :] - vertices[None, :, :], axis=2)
    assert np.max(np.min(distances, axis=0)) < 1e-6
    assert np.max(np.min(distances, axis=1)) < 1e-6
    expected = 2.0 * radius if count % 2 == 0 else 2.0 * radius * np.cos(np.pi / (2.0 * count))
    assert result.diameter == pytest.approx(expected, rel=1e-7, abs=1e-7)
    assert bool(result.covered) is (count % 2 == 0)
