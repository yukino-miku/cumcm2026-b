"""第三问协议、覆盖证书与保守几何的独立边界验证。"""
import math
import numpy as np
import pytest

from cumcm2026_b.q3_geometry import (TargetRegion, minimum_circle, clear_point, circle_polygon,
    min_distance, max_distance, coverage_certificate, search_stations, sweep_path)
from cumcm2026_b.q3_local_env import LocalEnvironment, Source
from cumcm2026_b.q3_protocol import RobotClient, ProtocolError, BudgetExceeded


def test_attachment_time_sequence_and_clear_channel():
    env = LocalEnvironment([])
    c = RobotClient(env, "local-robot")
    c.enter()
    assert c.measure([300, 400], 1)["virtual_time_s"] == 105
    assert c.measure([300, 400], 2)["virtual_time_s"] == 111
    assert c.clear([300, 0], 3)["virtual_time_s"] == 194
    assert c.state.channel == 2
    assert c.measure([300, 0], 2)["virtual_time_s"] == 199
    assert c.exit()["virtual_time_s"] == 199
    assert sum(c.state.time_parts().values()) == 199


def test_lost_response_reuses_same_id_and_charges_once():
    env = LocalEnvironment([])
    class LoseOnce:
        lost = False
        def post(self, path, payload, timeout):
            result = env.post(path, payload, timeout)
            if path == "/measure" and not self.lost:
                self.lost = True
                raise TimeoutError("已执行后丢失响应")
            return result
    c = RobotClient(LoseOnce(), "local-robot", sleeper=lambda _: None)
    c.enter()
    c.measure([300, 400], 1)
    requests = [r for r in c.records if r.get("path") == "/measure"]
    assert len(requests) == 2 and requests[0]["payload"] == requests[1]["payload"]
    assert c.state.virtual_time == env.virtual == 105
    assert c.state.measurements == 1


def test_rejected_response_does_not_reset_state():
    env = LocalEnvironment([])
    c = RobotClient(env, "local-robot")
    c.enter()
    c.measure([300, 400], 2)
    original = c.state.position, c.state.channel, c.state.virtual_time
    c.robot_id = "wrong"
    with pytest.raises(ProtocolError): c.measure([0, 0], 3)
    assert (c.state.position, c.state.channel, c.state.virtual_time) == original


def test_uncertain_action_prevents_new_request():
    class Broken:
        def post(self, *args): raise TimeoutError("无响应")
    c = RobotClient(Broken(), "local-robot", retries=0)
    with pytest.raises(ProtocolError): c.enter()
    assert c.pending is not None
    with pytest.raises(ProtocolError, match="未决"): c.enter()


def test_deadline_uses_actual_enter_remaining_and_reserves_exit():
    now = [50.]
    c = RobotClient(LocalEnvironment([], remaining=9), "local-robot", clock=lambda: now[0])
    c.enter()
    assert c.state.deadline == 59
    now[0] = 55
    with pytest.raises(BudgetExceeded): c.measure([0, 0], 1)
    c.exit()
    assert c.state.exited


def test_near_exact_threshold_and_clear_success():
    c = RobotClient(LocalEnvironment([Source(1, (3, 4))]), "local-robot")
    c.enter()
    assert c.measure([0, 0], 1)["measure_result"] == "near"
    assert c.clear([0, 0], 1)["clear_result"] == "success"
    assert c.state.virtual_time == 10


@pytest.mark.parametrize("position,channel", [([math.nan, 0], 1), ([2e6+1, 0], 1), ([0, 0], True), ([0, 0], 21)])
def test_invalid_action_not_sent(position, channel):
    env = LocalEnvironment([])
    c = RobotClient(env, "local-robot")
    c.enter()
    with pytest.raises(ValueError): c.measure(position, channel)
    assert len(env.history) == 1 and c.pending is None


def test_fixed_location_error_and_rounding_wrap():
    env = LocalEnvironment([Source(1, (900, -0.01))], error_mode="hash")
    c = RobotClient(env, "local-robot")
    c.enter()
    a = c.measure([0, 0], 1)
    b = c.measure([0, 0], 1)
    assert a["svd_deg"] == b["svd_deg"]
    assert 0 <= a["svd_deg"] < 360


def test_equilateral_diameter_40_is_not_safe_clear():
    vertices = np.array([[0., 0.], [40., 0.], [20., 20*math.sqrt(3)]])
    _, radius = minimum_circle(vertices)
    assert radius == pytest.approx(40/math.sqrt(3))
    assert clear_point(vertices, [0, 0]) is None


def test_clear_certificate_and_move_saving():
    p = np.array([[0., -3.], [10., -3.], [10., 3.], [0., 3.]])
    target = clear_point(p, [-100., 0.])
    assert target[0] < 0 and max_distance(p, target) <= 19.5+1e-8


def test_outer_circle_contains_true_boundary_and_error_is_outward():
    p = circle_polygon(radius=1500)
    assert np.linalg.norm(p, axis=1).min() > 1500
    for angle in np.linspace(0, 2*np.pi, 259):
        assert min_distance(p, [1500*math.cos(angle), 1500*math.sin(angle)]) <= 1e-7


@pytest.mark.parametrize("error", [-1., 1.])
def test_full_history_keeps_truth_with_rounded_extreme_errors(error):
    truth = np.array([1799.5, .1])
    region = TargetRegion()
    for station in [np.array([1200., 0]), np.array([1500., 400]), np.array([1700., -300])]:
        angle = math.degrees(math.atan2(*(truth-station)[::-1]))
        reported = round((angle+error) % 360, 2) % 360
        region.observe(station, "direction", reported)
        assert min_distance(region.vertices, truth) < 1e-6
    assert len(region.observations) == 3


def test_negative_is_retained_without_unsafe_convex_cut():
    region = TargetRegion()
    before = region.vertices.copy()
    region.observe([0, 0], "no_signal")
    np.testing.assert_array_equal(region.vertices, before)
    assert not region.maybe_contains([0, 0])
    assert region.maybe_contains([1200, 0])


def test_seven_sites_are_certificate_but_partial_sites_are_not():
    assert coverage_certificate(search_stations()) is not None
    assert coverage_certificate(search_stations()[:3]) is None
    assert coverage_certificate(np.tile([0., 0.], (8, 1))) is None


def test_finite_sweep_covers_boundary_not_just_interior_centers():
    p = np.array([[0., 0.], [81., 0.], [81., 13.], [0., 13.]])
    path = sweep_path(p, [0, 0])
    for x in np.linspace(0, 81, 37):
        for y in np.linspace(0, 13, 9):
            assert np.linalg.norm(path-[x, y], axis=1).min() <= 10*math.sqrt(2)+1e-7


def test_collinear_vertices_do_not_claim_external_point_is_inside():
    assert min_distance([[0., 0.], [5., 0.], [10., 0.]], [20., 0.]) == pytest.approx(10)
