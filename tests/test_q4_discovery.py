"""未知频道补全必须复用当前站、遵守600米间隔及全清停止。"""
import numpy as np
import pytest

from cumcm2026_b.q4_discovery_strategy import DiscoveryFour, DiscoveryConfig
from cumcm2026_b.q4_local_env import LocalEnvironment, Source
from cumcm2026_b.q3_protocol import RobotClient


@pytest.mark.parametrize('enabled', [False, True])
def test_fifth_and_later_unknown_channel_can_be_discovered_without_extra_movement(enabled):
    env = LocalEnvironment([Source(8, (900, 700), 1000, None)], error_mode='zero')
    client = RobotClient(env, 'local-robot')
    solver = DiscoveryFour(client, DiscoveryConfig(scan_all_unknown=enabled))
    client.enter()
    point = np.array([700., 700.])
    solver._measure(solver.channels[1], point, '测试当前站')
    before = client.state.distance
    solver._opportunistic(point, 1)
    assert client.state.distance == before
    assert solver.channels[8].status == ('found' if enabled else 'unknown')
    if enabled:
        assert any(e['动作'] == '补齐顺路未知频道' and 8 in e['候选频道'] for e in solver.events)
    count = client.state.measurements
    solver._opportunistic(point, 1)
    if enabled:
        assert client.state.measurements == count
    else:
        # 旧限额只扫前4个；在同一停点再次调用仍可处理尚未测过的下一批。
        assert client.state.measurements == count + 4
    measured = [e['频道'] for e in solver.events if e['动作'] == '检测']
    assert len(measured) == len(set(measured))
    client.exit()
    client.close()


def test_recent_unknown_measurements_still_use_600_meter_spacing():
    client = RobotClient(LocalEnvironment([]), 'local-robot')
    solver = DiscoveryFour(client)
    client.enter()
    for state in solver.channels.values():
        state.measured_sites = [[0., 0.]]
    before = client.state.measurements
    solver._opportunistic(np.array([599., 0.]), 1)
    assert client.state.measurements == before
    client.exit()
    client.close()


def test_opportunistic_off_disables_both_parent_and_unknown_completion():
    client = RobotClient(LocalEnvironment([]), 'local-robot')
    solver = DiscoveryFour(client, DiscoveryConfig(opportunistic=False))
    client.enter()
    solver._opportunistic(np.zeros(2), 1)
    assert client.state.measurements == 0 and not solver.events
    client.exit()
    client.close()


def test_invalid_unknown_scan_flag():
    with pytest.raises(ValueError):
        DiscoveryConfig(scan_all_unknown='true')
