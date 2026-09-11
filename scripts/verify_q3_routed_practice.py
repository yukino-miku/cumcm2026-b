#!/usr/bin/env python
"""核验新版演练脱敏记录、完整费用、统计、图表哈希和链接；可选重新读取私有来源。"""
import argparse
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from PIL import Image
from evaluate_q3_routed_practice import ROOT, BASE, REPORT, TRACKS, OLD, gather, digest, statistics, comparison
from plot_q3_routed_practice import MANIFEST, verify_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-source', action='store_true', help='核对全部私有原始输入并重新关联HTTP动作')
    args = parser.parse_args()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    data = json.loads(TRACKS.read_text(encoding='utf-8'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows = report['逐局记录']
    verify_tracks(data, report)
    assert statistics(rows) == report['统计']
    assert comparison([r for r in rows if r['组'] == 's11'], [r for r in rows if r['组'] == 's22']) == {
        k: v for k, v in report['两方案比较'].items() if k not in ['第一组', '第二组']}
    old = json.loads(OLD.read_text(encoding='utf-8'))
    assert old['统计'] == report['原版历史统计']
    for new_group, old_group in [('s11', 's1'), ('s22', 's2')]:
        assert comparison([r for r in rows if r['组'] == new_group], [r for r in old['逐局记录'] if r['组'] == old_group]) == {
            k: v for k, v in report['历史对照'][new_group].items() if k not in ['第一组', '第二组']}
    total = 0
    for run, row in zip(data['逐局轨迹'], rows):
        assert run['编号'] == row['显示编号'] and run['全清'] and row['正常退出']
        current, channel, virtual, distance, switches = [0., 0.], 1, 0., 0., 0
        cleared = set(); before = 0
        parts = {'移动': 0., '检测': 0., '切换': 0., '清除成功': 0., '清除失败': 0.}
        phases = {'末次固定检测及之前': 0., '末次固定检测之后': 0.}
        for e in run['实际动作']:
            length = math.dist(current, e['位置']); current = e['位置']; distance += length
            virtual += length / 5; parts['移动'] += length / 5
            phases['末次固定检测之后' if e['步骤'] > run['末次固定检测步骤'] else '末次固定检测及之前'] += length
            if e['动作'] == '检测':
                switches += e['频道'] != channel
                virtual += 5 + (e['频道'] != channel)
                parts['检测'] += 5; parts['切换'] += e['频道'] != channel; channel = e['频道']
            elif e['结果'] == 'success':
                assert e['频道'] not in cleared
                cleared.add(e['频道']); virtual += 5; parts['清除成功'] += 5
                before += e['步骤'] < run['末次固定检测步骤']
            else:
                virtual += 3; parts['清除失败'] += 3
            assert abs(virtual - e['虚拟时间_秒']) < .001
        assert len(cleared) == row['官方结果文件目标总数'] == row['成功清除数']
        assert switches == row['切换次数'] and before == row['末次固定检测前成功清除数']
        assert abs(virtual - row['总虚拟时间_秒']) < .001 and abs(distance - row['总路程_米']) < 1e-5
        assert abs(row['每目标虚拟时间_秒'] - virtual / len(cleared)) < .001
        for key in parts: assert abs(parts[key] - row['时间分项_秒'][key]) < .001
        for key in phases: assert abs(phases[key] - row['按阶段路程_米'][key]) < 1e-5
        total += len(run['实际动作'])
    assert total == report['核验合计']['实际检测清除坐标核对数'] == 2340
    assert len(manifest['图表SHA256']) == 34
    for name, value in manifest['绘图来源SHA256'].items(): assert digest(ROOT/name) == value, name
    for name, value in manifest['图表SHA256'].items():
        path = ROOT/name
        assert digest(path) == value, name
        if path.suffix == '.png':
            with Image.open(path) as im: im.verify()
        else: ET.parse(path)
    # 仓库内来源无论是否有私有文件都应验证；私有原始输入只在显式模式下读取。
    for name, value in report['来源SHA256'].items():
        if args.with_source or not name.startswith(('materials/', 'local-only/')):
            assert digest(ROOT/name) == value, name
    if args.with_source:
        fresh_rows, fresh_tracks, sources = gather(BASE)
        # JSON对象的键必为字符串，HTTP状态Counter中的整数200写出后成为"200"。
        fresh_rows = json.loads(json.dumps(fresh_rows, ensure_ascii=False))
        assert fresh_rows == rows and fresh_tracks == data['逐局轨迹']
        assert all(report['来源SHA256'][name] == value for name, value in sources.items())
        assert all(digest(ROOT/name) == value for name, value in old['来源SHA256'].items())
    # 核对中文报告手写的十二行数值，避免图表正确但论文表格转录错误。
    document = (ROOT/'docs/第三问/新版演练评估_s11_s22.md').read_text(encoding='utf-8')
    for row in rows:
        line = next(s for s in document.splitlines() if s.startswith(f"| {row['显示编号']} |"))
        cells = [s.strip() for s in line.split('|')[1:-1]]
        assert cells == [row['显示编号'], row['案例码'], str(row['官方结果文件目标总数']),
                         f"{row['总虚拟时间_秒']:.3f}", f"{row['每目标虚拟时间_秒']:.3f}",
                         f"{row['总路程_米']/1000:.3f}", str(row['检测次数']), str(row['路线调度统计']['顺路证书清除次数'])]
    links = 0
    paths = list((ROOT/'docs/第三问').glob('*.md')) + list((ROOT/'paper/sections').glob('第三问*.md'))
    for path in paths:
        text = path.read_text(encoding='utf-8')
        assert '\ufffd' not in text and text.count('```') % 2 == 0, path
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent/target.split('#')[0]).resolve().exists(), (path, target)
                links += 1
    print(f'ROUTED_PRACTICE_VERIFY_OK：12局、{total}条实际检测清除坐标、完整费用和统计、34个PNG/SVG、{links}个中文链接；私有来源复核={args.with_source}。')


if __name__ == '__main__': main()
