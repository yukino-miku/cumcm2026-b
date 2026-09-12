"""清除优先版独立动作审计。物理费用核验源自旧审计器，另核查实际负观测的连续证书。"""
import math
import numpy as np
from scipy.spatial import ConvexHull
from cumcm2026_b.q3_geometry import min_distance
from cumcm2026_b.q4_strategy import DirectionalChannel
from cumcm2026_b.q4_coverage import discovery_mesh

def verify_run(result):
    assert result['流程正常完成'] and result['正常退出'] and result['异常'] is None
    truth = result['离线真值核验']
    sources = {s['频道']: s for s in truth['目标']}
    cleared = set()
    current = np.zeros(2)
    channel = 1
    virtual = distance_sum = 0.
    counts = dict(measure=0, switch=0, success=0, fail=0, envelope=0)
    for event in result['动作记录']:
        if '外包顶点' in event:
            assert min_distance(event['外包顶点'], sources[event['频道']]['位置']) < 1e-5
            counts['envelope'] += 1
        if event['动作'] not in {'检测', '清除'}:
            continue
        point = np.asarray(event['位置'])
        travel = float(np.linalg.norm(point-current))
        if event.get('用途') == '顺路检测':
            assert travel < 1e-6  # 顺路扫描只能复用当前真实停点。
        current = point
        distance_sum += travel
        virtual += travel/5
        c = event['频道']
        target = sources.get(c) if c not in cleared else None
        distance = math.dist(point, target['位置']) if target else math.inf
        if event['动作'] == '检测':
            assert event['阶段'] != '末尾增站评估'
            assert event['用途'].startswith('固定搜索站') or event['用途'] in {'顺路检测', '追加测向', '发现保障补扫'}
            counts['measure'] += 1
            switch = int(c != channel)
            counts['switch'] += switch
            virtual += 5+switch
            channel = c
            reception = target is not None and distance <= target['接收半径']
            if reception and target['发射轴_度'] is not None:
                angle = math.radians(target['发射轴_度'])
                dx, dy = point-np.asarray(target['位置'])
                reception = dx*math.cos(angle)+dy*math.sin(angle) >= -1e-9
            expected = 'no_signal' if not reception else 'near' if distance <= 5 else 'direction'
            assert expected == event['结果']
            if expected == 'direction':
                delta = np.asarray(target['位置'])-point
                bearing = math.degrees(math.atan2(delta[1], delta[0]))
                error = (event['示向度']-bearing+180) % 360-180
                assert abs(error) <= 1.005+1e-8
        else:
            success = distance <= 20
            assert event['结果'] == ('success' if success else 'no_target_in_range')
            if event['具有覆盖证书']:
                assert success and event['最远可能目标距离_米'] < 20
            virtual += 5 if success else 3
            counts['success' if success else 'fail'] += 1
            if success:
                cleared.add(c)
        assert abs(virtual-event['虚拟时间_秒']) < .001
    assert cleared == set(result['已清除频道'])
    assert sources.keys()-cleared == set(truth['遗漏频道'])
    assert bool(len(sources) == len(cleared)) == truth['全部清除']
    assert abs(distance_sum-result['总路程_米']) < 1e-5
    assert abs(virtual-result['虚拟总时间_秒']) < .001
    assert abs(sum(result['时间分项_秒'].values())-virtual) < .001
    assert [counts[k] for k in ['measure', 'switch', 'success', 'fail']] == [
        result[k] for k in ['检测次数', '频道切换次数', '清除成功次数', '清除失败次数']]
    assert result['已完成搜索站'] == list(range(len(result['已完成搜索站'])))
    assert len(result['已完成搜索站']) == 13 or result['清除数'] == 16
    absent = set(result['已证明不存在频道'])
    assert not absent.intersection(sources) and not result['已发现未清除频道']
    proven = len(cleared) == 16 or len(cleared | absent) == 20
    assert result['全部完成证据'] == proven
    assert not proven or truth['全部清除']
    states = {c: DirectionalChannel(c) for c in range(1, 21)}
    for e in result['动作记录']:
        c = e.get('频道')
        if e['动作'] == '检测':
            s = states[c]
            assert s.status in {'unknown', 'found'}
            s.region.observe(e['位置'], e['结果'], e.get('示向度'))
            s.measured_sites.append(e['位置'])
            if e['结果'] == 'no_signal':
                s.negative_sites.append(e['位置'])
            else:
                s.status = 'found'
        elif e['动作'] == '清除':
            assert states[c].status == 'found'
            if e['结果'] == 'success':
                states[c].status = 'cleared'
            else:
                states[c].region.failed_clear(e['位置'])
        elif e['动作'] == '频道不存在证书':
            s = states[c]
            cert = e['证书']
            assert s.status == 'unknown' and cert['实际负观测站'] == s.negative_sites
            mesh, triangles = discovery_mesh(cert.get('网格', 'lattice'))
            assert cert['未覆盖单元'] == [] and cert['已覆盖单元数'] == cert['总单元数'] == len(triangles)
            sites = np.asarray(s.negative_sites)
            assert len(cert['单元证据']) == len(triangles)
            for ids, witness in zip(triangles, cert['单元证据']):
                vertices = mesh[ids]
                points = sites[witness['负观测序号']]
                assert np.linalg.norm(points[:, None, :] - vertices, axis=2).max() <= 999.5
                if witness['类型'] == '原三角形顶点':
                    assert np.array_equal(points, vertices)
                else:
                    assert witness['类型'] == '严格凸包替代'
                    hull = ConvexHull(points)
                    assert np.max(vertices @ hull.equations[:, :2].T + hull.equations[:, 2]) < -1e-6
            s.status = 'absent'
        elif e['动作'] == '跳过检测' and '距离下界_米' in e:
            assert states[c].status == 'found'
            assert np.allclose(states[c].region.vertices, e['外包顶点'], atol=1e-7, rtol=0)
            assert min_distance(states[c].region.vertices, e['位置']) > 1500 + 1e-5
    for row in result['频道记录']:
        s = states[row['频道']]
        assert row['状态'] == s.status
        assert all(e['来源'] == '清除失败' and e['半径'] == 20 for e in row['排除约束'])
        assert row['追加测向次数'] <= 8
        assert (row['不存在证书'] is not None) == (s.status == 'absent')
    return counts
