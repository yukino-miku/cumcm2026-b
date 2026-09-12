"""几何补测间距规则与原版回退的行为检查。"""
import math
import numpy as np
import pytest
from cumcm2026_b.q4_spacing_strategy import SpacedFour, SpacingConfig
from cumcm2026_b.q4_strategy import SchemeFour
from cumcm2026_b.q3_protocol import RobotClient
from cumcm2026_b.q4_local_env import LocalEnvironment, Source, make_case
from cumcm2026_b.q3_geometry import min_distance, max_distance


def anchor(gap):
    solver=SpacedFour(RobotClient(LocalEnvironment([]),'local-robot'),SpacingConfig(probe_spacing_m=gap))
    state=solver.channels[1]
    state.status='found'
    state.region.observe([0,0],'direction',0.)
    state.measured_sites.append([0.,0.])
    return solver,state


def test_separation_retains_range_envelope_and_does_not_consume_measurement_budget():
    solver,state=anchor(300.)
    plan=solver._observation_plan(state,np.zeros(2))
    assert plan['动作']=='追加测向' and not plan['间距约束已放宽']
    assert plan['距当前机器人_米']>=300-1e-6
    assert plan['距该频道上次观测_米']>=300-1e-6
    assert max_distance(state.region.vertices,plan['位置'])<=999.5+1e-6
    assert not plan['接收保证'] and state.localization_count==0


def test_impossible_separation_relaxes_without_forcing_optical_sweep():
    solver,state=anchor(4000.)
    plan=solver._observation_plan(state,np.zeros(2))
    assert plan['动作']=='追加测向' and plan['间距约束已放宽']
    assert plan['距当前机器人_米']<4000
    assert plan['间距筛选前候选数']==plan['间距筛选后候选数']


def test_zero_spacing_calls_original_geometry(monkeypatch):
    solver,state=anchor(0.)
    sentinel={'动作':'测试原几何分支'}
    monkeypatch.setattr(SchemeFour,'_observation_plan',lambda self,s,c:sentinel)
    assert solver._observation_plan(state,np.zeros(2)) is sentinel


@pytest.mark.parametrize('value',[-1,True,float('nan'),float('inf'),'100'])
def test_spacing_validation(value):
    with pytest.raises(ValueError):SpacingConfig(probe_spacing_m=value)


def test_known_source_full_loop_preserves_truth_and_clear_safety():
    env=make_case(80101,10,'uniform','min','hash',1.,'random')
    r=SpacedFour(RobotClient(env,'local-robot'),SpacingConfig(probe_spacing_m=200)).run()
    assert r['流程正常完成'] and not r['已发现未清除频道']
    truth={s['频道']:s['位置'] for s in env.truth_for_evaluation()['目标']}
    for e in r['动作记录']:
        if '外包顶点' in e:assert min_distance(e['外包顶点'],truth[e['频道']])<1e-5
        if e['动作']=='清除' and e['具有覆盖证书']:
            assert e['结果']=='success' and math.dist(e['位置'],truth[e['频道']])<=20
    assert r['末尾增站评估']['自动新增未知源搜索站数']==0
