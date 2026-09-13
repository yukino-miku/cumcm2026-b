"""归档q3form/q4form正式包及配对HTTP原始记录，逐动作审计并绘制中文轨迹。

--verify仅从已上传ZIP复核；--source-check另外核对本机原始文件。
不调用模拟器，不解密正式包，不将运行成功标志等同官方目标数或成绩。
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from analyze_q4_practice_4 import matching_logs
from analyze_q3_practice_log import audit_actions
from evaluate_q3_routed_practice import audit_trace
from cumcm2026_b.q3_geometry import TargetRegion, search_stations as q3_stations, max_distance
from cumcm2026_b.q4_strategy import DirectionalRegion, search_stations as q4_stations

SOURCE = ROOT / 'materials/original/simulator/CUMCM2026B/Jammers-simulator-win64/Jammers-simulator/JammersSimulatorData/behavior-logs'
ARCHIVE = ROOT / 'materials/original/formal_tests/2026-09-13'
TABLES = ROOT / 'results/tables/正式测试/2026-09-13'
MODELS = ROOT / 'results/models/正式测试/2026-09-13'
FIG = ROOT / 'results/figures/正式测试/2026-09-13'
REPORT = ROOT / 'docs/正式测试/2026-09-13_第三四问正式测试分析.md'
PAPER = ROOT / 'paper/sections/第三四问_正式测试_论文备用说明.md'
INVENTORY = ARCHIVE / '归档清单.json'
SUMMARY = TABLES / '汇总.json'
MANIFEST = TABLES / '图文及分析来源.json'


def load(raw):
    return json.loads(raw.decode('utf-8-sig') if isinstance(raw, bytes) else raw)


def read(path):
    raw = path.read_bytes()
    return load(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded(value)
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def digest(raw):
    return sha256(raw).hexdigest()


def millis(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp() * 1000


def beijing(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00')).astimezone(
        timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')


def formal_header(raw):
    assert len(raw) >= 14 and raw[:8] == b'JMBFLOG1', '正式日志魔数不符'
    assert int.from_bytes(raw[8:10], 'big') == 1
    size = int.from_bytes(raw[10:14], 'big')
    assert 0 < size < len(raw) - 14
    header = load(raw[14:14+size])
    assert header['package_type'] == 'formal_behavior_log'
    assert header['formal_index'] in (1, 2, 3) and header['problem_no'] in (3, 4)
    assert header['practice_run_no'] is None
    return header


def create_archives():
    """只复制指定的18个正式文件及12份关联原始记录；已存在归档必须逐字节相同。"""
    index = matching_logs()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    entries, run_links = [], []
    for number in (3, 4):
        group = f'q{number}form'
        folder = SOURCE / group
        paths = sorted(p for p in folder.iterdir() if p.is_file())
        assert len(paths) == 9 and len(list(folder.glob('*.jlog'))) == 3
        content = {f'{group}/{p.name}': (p, p.read_bytes()) for p in paths}
        for package in sorted(folder.glob('*.jlog')):
            raw = package.read_bytes()
            h = formal_header(raw)
            assert h['problem_no'] == number
            label = f'F{number}-{h["formal_index"]}'
            matches = [x for x in index if x[0] == number and x[1] == h['team_no']
                       and abs(millis(h['created_at_utc']) - x[2]) <= 2000]
            assert len(matches) == 1, (label, '不能唯一匹配HTTP日志')
            _, _, exited, http_folder, _ = matches[0]
            for filename in ['请求响应日志.jsonl', '运行结果.json']:
                p = http_folder / filename
                content[f'http/{label}/{filename}'] = (p, p.read_bytes())
            run_links.append({'编号': label, 'ZIP': f'{group}.zip', '正式包内路径': f'{group}/{package.name}',
                              'HTTP内目录': f'http/{label}', '关联本地目录': http_folder.relative_to(ROOT).as_posix(),
                              '生成减退出响应_毫秒': millis(h['created_at_utc']) - exited})
        target = ARCHIVE / f'{group}.zip'
        if target.exists():
            with zipfile.ZipFile(target) as z:
                assert set(z.namelist()) == set(content)
                assert all(z.read(name) == raw for name, (_, raw) in content.items())
        else:
            with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                for name, (_, raw) in sorted(content.items()):
                    info = zipfile.ZipInfo(name, date_time=(2026, 9, 13, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, raw)
        for name, (p, raw) in sorted(content.items()):
            entries.append({'ZIP': target.name, '包内路径': name, '原始路径': p.relative_to(ROOT).as_posix(),
                            '字节数': len(raw), 'SHA256': digest(raw)})
    value = {'说明': '18个正式文件加12份原始HTTP/策略结果，逐字节原样归档；ZIP避免Git换行转换。',
             '文件': entries, '运行关联': run_links,
             'ZIP_SHA256': {f'q{n}form.zip': digest((ARCHIVE / f'q{n}form.zip').read_bytes()) for n in (3, 4)}}
    if INVENTORY.exists():
        assert read(INVENTORY) == value
    else:
        write(INVENTORY, value)


def verify_archives(source_check=False):
    inventory = read(INVENTORY)
    assert len(inventory['文件']) == 30 and len(inventory['运行关联']) == 6
    for filename, expected in inventory['ZIP_SHA256'].items():
        path = ARCHIVE / filename
        assert digest(path.read_bytes()) == expected
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            rows = [r for r in inventory['文件'] if r['ZIP'] == filename]
            assert len(rows) == 15 and set(z.namelist()) == {r['包内路径'] for r in rows}
            for row in rows:
                raw = z.read(row['包内路径'])
                assert digest(raw) == row['SHA256'] and len(raw) == row['字节数']
                if source_check:
                    assert raw == (ROOT / row['原始路径']).read_bytes()
    return inventory


def pair_protocol(items, result):
    """保留请求与响应全部原始字段，并明确关联JSONL记录序号及策略动作步骤。"""
    assert [x['记录序号'] for x in items] == list(range(1, len(items)+1)), '原日志记录序号缺失或乱序'
    requests = [x for x in items if x['类型'] == '请求']
    responses = [x for x in items if x['类型'] == '响应']
    assert len({x['payload']['request_id'] for x in requests}) == len(requests), '本批应没有重试或重复请求'
    assert len({x['request_id'] for x in responses}) == len(responses)
    lookup = {x['request_id']: x for x in responses}
    assert set(lookup) == {x['payload']['request_id'] for x in requests}
    assert requests[0]['path'] == '/enter' and requests[-1]['path'] == '/exit'
    actions = audit_trace(items, result)
    logged_strategy = [{k:v for k,v in item.items() if k not in ('类型', '记录序号')}
                       for item in items if item['类型'] == '策略']
    assert logged_strategy == result['动作记录'], '原HTTP策略轨迹与结果文件不一致'
    output, current, tuned, virtual, action_index = [], [0., 0.], 1, 0., 0
    for index, request in enumerate(requests):
        response = lookup[request['payload']['request_id']]
        assert request['记录序号'] < response['记录序号']
        if index + 1 < len(requests):
            assert response['记录序号'] < requests[index+1]['记录序号']
        assert request['尝试'] == 1 and response['HTTP'] == 200 and response['response']['accepted']
        p, body, endpoint = request['payload'], response['response'], request['path']
        travel = movement = operating = switch = 0.
        old = list(current)
        event = None
        if endpoint in ('/measure', '/clear'):
            current = [p['position']['x'], p['position']['y']]
            travel = math.dist(old, current)
            movement = travel / 5
            event = actions[action_index]
            action_index += 1
            if endpoint == '/measure':
                operating, switch = 5., float(tuned != p['channel'])
                tuned = p['channel']
                assert event.get('示向度') == body.get('svd_deg')
            else:
                operating = 5. if body['clear_result'] == 'success' else 3.
        cost = movement + operating + switch
        virtual += cost
        assert abs(virtual - body['virtual_time_s']) < .001
        output.append({'指令序号': index+1, '请求记录序号': request['记录序号'],
                       '响应记录序号': response['记录序号'], '策略步骤': event['步骤'] if event else None,
                       '用途': event['用途'] if event else '进入任务' if endpoint == '/enter' else '退出任务',
                       '请求': request, '响应': response, '移动起点': old, '移动终点': list(current),
                       '移动距离_米': travel, '移动时间_秒': movement, '操作时间_秒': operating,
                       '切换时间_秒': switch, '本次费用_秒': cost, '重算累计虚拟时间_秒': virtual})
    assert action_index == len(actions)
    return output


def audit_geometry(result, number):
    """从真实反馈重建位置外包；Q3逐个检查实际负观测站的七站连续覆盖证据。"""
    regions = {c: (TargetRegion() if number == 3 else DirectionalRegion()) for c in range(1, 21)}
    negatives, discovered, cleared = defaultdict(list), set(), set()
    certificates = 0
    for event in result['动作记录']:
        if event['动作'] not in ('检测', '清除'):
            continue
        c, p = event['频道'], event['位置']
        assert c not in cleared
        region = regions[c]
        if event['动作'] == '检测':
            region.observe(p, event['结果'], event.get('示向度'))
            if event['结果'] == 'no_signal':
                negatives[c].append(p)
            else:
                discovered.add(c)
            if '外包顶点' in event:
                assert np.allclose(region.vertices, event['外包顶点'], rtol=0, atol=1e-7)
        else:
            assert c in discovered
            bound = max_distance(region.vertices, p)
            assert abs(bound - event['最远可能目标距离_米']) < 1e-5
            if event['具有覆盖证书']:
                assert bound < 20 and event['结果'] == 'success'
                certificates += 1
            if event['结果'] == 'success':
                cleared.add(c)
            else:
                region.failed_clear(p)
    assert sorted(cleared) == result['已清除频道'] and discovered == cleared
    absent = set(result['已证明不存在频道'])
    geometric_bound = math.sqrt(1800**2 + 1200**2 - 2*1800*1200*math.cos(math.pi/6))
    for row in result['频道记录']:
        c = row['频道']
        assert row['排除约束'] == regions[c].exclusions
        if row['状态'] == 'absent':
            assert number == 3 and c not in discovered and c in absent
            assert row['不存在证书']['方式'] == '七站连续覆盖'
            assert geometric_bound < 1000
            assert abs(row['不存在证书']['最大最近距离_米'] - geometric_bound) < 1e-8
            assert all(any(math.dist(s, p) <= 1e-6 for p in negatives[c]) for s in q3_stations())
        elif row['状态'] == 'cleared':
            assert c in cleared
        else:
            assert row['状态'] == 'unknown' and c not in discovered
    if number == 3:
        assert cleared | absent == set(range(1, 21)) and not (cleared & absent)
        proof = True
    else:
        assert not absent and not result['已发现未清除频道']
        assert not result['配置']['discovery_safeguard'] and result['发现保障站坐标'] == []
        assert np.allclose(result['固定站坐标'], q4_stations(), rtol=0, atol=1e-8)
        proof = len(cleared) == 16
    assert result['全部完成证据'] == proof
    return {'连续外包及清除证书': '通过', '复核的安全清除次数': certificates,
            '逐频道不存在证书数': len(absent), '模型全清证据': proof,
            '证据性质': '题设数量上界' if len(cleared) == 16 else '全向七站逐频道连续覆盖排除' if number == 3 else '全清待核实',
            '官方目标总数': None, '根据证据可确定目标数': len(cleared) if proof else None,
            '可能遗漏数范围': [0, 0] if proof else [0, 16-len(cleared)],
            '限制': '正式包正文未解密，未认证包签名；证据来自关联HTTP反馈及题设模型，不是官方评分回执。'}


def diagnose(result, number):
    actions = [e for e in result['动作记录'] if e['动作'] in ('检测', '清除')]
    fixed_actions = [e for e in actions if e['用途'].startswith(('搜索站', '固定搜索站'))]
    markers = [e for e in result['动作记录'] if e['动作'] == '固定扫描完成']
    cutoff = markers[0] if markers else fixed_actions[-1]
    fixed = q3_stations() if number == 3 else q4_stations()
    seen = []
    for event in fixed_actions:
        i = int(np.argmin(np.linalg.norm(fixed-np.asarray(event['位置']), axis=1)))
        assert math.dist(event['位置'], fixed[i]) < 1e-7
        if not seen or seen[-1] != i:
            seen.append(i)
    assert seen == result['已完成搜索站']
    assert seen == list(range(len(seen)))
    assert len(seen) == len(fixed) or result['清除数'] == 16
    position, groups, feedback, purposes = [0., 0.], defaultdict(lambda: {'路程_米': 0., '清除数': 0}), Counter(), Counter()
    channels, clear_failures, legs = {}, [], []
    for e in actions:
        c = e['频道']
        info = channels.setdefault(c, {'频道': c, '首次发现': None, '成功清除': None,
                                      '追加测向次数': 0, '追加无信号次数': 0, '清除失败次数': 0})
        tail = e['步骤'] > cutoff['步骤']
        category = ('固定扫描后收尾' if tail else '前往固定站' if e['用途'].startswith(('搜索站', '固定搜索站'))
                    else '前往清除点' if e['动作'] == '清除' else '前往补测点')
        d = math.dist(position, e['位置'])
        groups[category]['路程_米'] += d
        if d > 1e-7:
            legs.append({'起点': position, '终点': e['位置'], '类别': category, '距离_米': d, '步骤': e['步骤']})
        position = e['位置']
        if e['动作'] == '检测':
            feedback[e['结果']] += 1
            purposes['固定站检测' if e['用途'].startswith(('搜索站', '固定搜索站')) else e['用途']] += 1
            if e['结果'] != 'no_signal' and info['首次发现'] is None:
                info['首次发现'] = {k:e[k] for k in ['步骤', '用途', '位置', '虚拟时间_秒']}
            if e['用途'] == '追加测向':
                info['追加测向次数'] += 1
                info['追加无信号次数'] += int(e['结果'] == 'no_signal')
        elif e['结果'] == 'success':
            info['成功清除'] = {k:e[k] for k in ['步骤', '用途', '位置', '虚拟时间_秒']}
            groups[category]['清除数'] += 1
        else:
            info['清除失败次数'] += 1
            clear_failures.append({k:e[k] for k in ['步骤', '用途', '频道', '位置', '虚拟时间_秒', '具有覆盖证书']})
    clear_events = [e for e in actions if e['动作'] == '清除' and e['结果'] == 'success']
    discovered = [x for x in channels.values() if x['首次发现']]
    assert all(x['成功清除'] for x in discovered)
    tail = groups['固定扫描后收尾']
    assert abs(sum(x['路程_米'] for x in groups.values()) - result['总路程_米']) < 1e-5
    return {'固定站': fixed.tolist(), '固定站完成数': len(seen), '固定扫描截止步骤': cutoff['步骤'],
            '固定扫描截止口径': '固定扫描完成事件' if markers else '最后一次固定站检测',
            '固定扫描截止时间_秒': cutoff['虚拟时间_秒'], '尾程时间_秒': result['虚拟总时间_秒']-cutoff['虚拟时间_秒'],
            '尾程路程_米': tail['路程_米'], '尾程清除数': tail['清除数'],
            '扫描中清除数': result['清除数']-tail['清除数'], '按移动目的分类': dict(groups),
            '检测用途次数': dict(purposes), '测量反馈次数': dict(feedback),
            '主动补测次数': sum(x['追加测向次数'] for x in channels.values()),
            '补测无信号次数': sum(x['追加无信号次数'] for x in channels.values()),
            '首次发现用途': dict(Counter('固定站检测' if x['首次发现']['用途'].startswith(('搜索站', '固定搜索站')) else x['首次发现']['用途'] for x in discovered)),
            '已发现源均已清除': True, '成功清除用途': dict(Counter(e['用途'] for e in clear_events)),
            '失败清除记录': clear_failures, '逐频道': list(channels.values()), '移动线段': legs,
            '最长5段移动': sorted(legs, key=lambda x:x['距离_米'], reverse=True)[:5],
            '最后成功清除后剩余时间_秒': result['虚拟总时间_秒']-clear_events[-1]['虚拟时间_秒'],
            '移动时间占比': result['时间分项_秒']['移动']/result['虚拟总时间_秒']}


def command_markdown(label, commands):
    lines = [f'# {label} 正式测试指令序列', '',
             '由原始HTTP日志导出，按实际发送顺序排列。原始请求、响应完整字段与记录序号保存在同名JSONL；原始日志原样保存在ZIP。', '',
             '本次费用=移动距离/5+检测或清除费用+检测频道切换费用；示向度为空表示该响应未返回示向。', '',
             '|序号|接口|频道|到达位置x,y（米）|用途|HTTP|响应结果|示向度/°|虚拟时刻/秒|本段距离/米|本次费用/秒|',
             '|---:|---|---:|---|---|---:|---|---:|---:|---:|---:|']
    for row in commands:
        request, response = row['请求'], row['响应']
        p, body = request['payload'], response['response']
        point = p.get('position')
        xy = f'{point["x"]:.3f},{point["y"]:.3f}' if point else '—'
        outcome = body.get('measure_result', body.get('clear_result', body.get('exit_reason', 'accepted')))
        lines.append(f'|{row["指令序号"]}|{request["path"]}|{p.get("channel", "—")}|{xy}|{row["用途"]}|'
                     f'{response["HTTP"]}|{outcome}|{body.get("svd_deg", "—")}|{body["virtual_time_s"]:.6f}|'
                     f'{row["移动距离_米"]:.6f}|{row["本次费用_秒"]:.6f}|')
    return '\n'.join(lines)+'\n'


def collect(inventory, verify):
    runs = []
    for link in inventory['运行关联']:
        label = link['编号']
        number = int(label[1])
        with zipfile.ZipFile(ARCHIVE / link['ZIP']) as z:
            package = z.read(link['正式包内路径'])
            h = formal_header(package)
            receipt = load(z.read(link['正式包内路径'].removesuffix('.jlog')+'.result.json'))
            attempts = [n for n in z.namelist() if n.startswith(f'q{number}form/formal-attempt-')
                        and n.endswith(f'-p{number}-{h["formal_index"]}-{h["case_code"]}.json')]
            assert len(attempts) == 1
            attempt = load(z.read(attempts[0]))
            for key in ['team_no', 'problem_no', 'formal_index', 'case_code']:
                assert h[key] == receipt[key] == attempt[key]
            assert receipt['package_sha256'] == digest(package)
            assert abs(millis(receipt['ended_at_utc'])-millis(h['created_at_utc'])) < 10
            assert 'jammer_count' not in receipt and 'jammer_count' not in h
            r = load(z.read(link['HTTP内目录']+'/运行结果.json'))
            items = [load(line) for line in z.read(link['HTTP内目录']+'/请求响应日志.jsonl').decode('utf-8').splitlines()]
        assert r['正常退出'] and r['异常'] is None
        protocol_audit = audit_actions(items, r)
        commands = pair_protocol(items, r)
        assert all(x['请求']['payload']['robot_id'] == h['team_no'] for x in commands)
        exit_time = commands[-1]['响应']['response']['real_timestamp_ms']
        assert abs(millis(h['created_at_utc'])-exit_time-link['生成减退出响应_毫秒']) < 1e-6
        assert millis(attempt['started_at_utc']) <= commands[0]['响应']['response']['real_timestamp_ms'] < exit_time
        evidence = audit_geometry(r, number)
        diagnosis = diagnose(r, number)
        provenance = r.get('运行来源', {})
        if number == 4:
            import subprocess
            commit = provenance['Git提交']
            assert re.fullmatch('[a-f0-9]{40}', commit) and provenance['Git工作区干净']
            for name, expected in provenance['源码_SHA256'].items():
                assert name.startswith(('src/', 'scripts/', 'configs/'))
                assert digest(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)) == expected
        summary = {k:h[k] for k in ['package_type', 'problem_no', 'formal_index', 'case_code',
                                   'client_version', 'created_at_utc', 'content_encryption', 'key_wrap_algorithm']}
        record = {'编号': label, '题号': number, '正式公开头摘要': summary,
                  '开始时间UTC': attempt['started_at_utc'], '结束时间UTC': receipt['ended_at_utc'],
                  '结束时间北京时间': beijing(receipt['ended_at_utc']), '正式包_SHA256': digest(package),
                  '关联': link, '结果': r, '协议审计': protocol_audit, '完成证据': evidence, '诊断': diagnosis,
                  '记录完整性': {'原始记录总行数': len(items), '请求数': len(commands), '响应数': len(commands),
                                 '策略事件数': len(r['动作记录']), '缺失请求响应数': 0,
                                 '策略事件逐条吻合': True, '序号连续且串行响应顺序核验': True},
                  '版本判定': f'运行记录中的Git提交{provenance["Git提交"]}，源码哈希与该提交逐一吻合' if provenance else
                              '方案名、配置、七站坐标和动作显示为第三问方案一路线版；本次原日志未记录源码哈希，不能追认精确提交'}
        path = MODELS / (label+'.json.gz')
        jsonl_path = TABLES / (label+'_指令响应.jsonl')
        md_path = TABLES / (label+'_指令序列.md')
        raw = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in commands).encode('utf-8')
        md = command_markdown(label, commands).encode('utf-8')
        if verify:
            # JSON将整型频道键转换为字符串；比较原始序列化字节，避免类型转换误报。
            assert gzip.decompress(path.read_bytes()) == encoded(record)
            assert jsonl_path.read_bytes() == raw and md_path.read_bytes() == md
        else:
            write(path, record)
            TABLES.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_bytes(raw)
            md_path.write_bytes(md)
            print(f'{label}：清除{r["清除数"]}，{r["虚拟总时间_秒"]:.3f}秒，'
                  f'{r["总路程_米"]/1000:.3f}公里，{evidence["证据性质"]}，{len(commands)}对请求响应', flush=True)
        runs.append(record)
    return runs


def summarize(runs):
    rows, groups = [], []
    for run in runs:
        r, d, e = run['结果'], run['诊断'], run['完成证据']
        rows.append({'编号':run['编号'], '题号':run['题号'], '正式次数':run['正式公开头摘要']['formal_index'],
                     '案例码':run['正式公开头摘要']['case_code'], '结束时间北京时间':run['结束时间北京时间'],
                     '清除数':r['清除数'], '官方目标总数':None, '模型全清证据':e['模型全清证据'],
                     '证据性质':e['证据性质'], '可能遗漏数范围':e['可能遗漏数范围'],
                     '虚拟时间_秒':r['虚拟总时间_秒'], '路程_米':r['总路程_米'],
                     '检测次数':r['检测次数'], '切换次数':r['频道切换次数'], '失败清除次数':r['清除失败次数'],
                     '时间分项_秒':r['时间分项_秒'], '固定站完成数':d['固定站完成数'],
                     '主动补测次数':d['主动补测次数'], '补测无信号次数':d['补测无信号次数'],
                     '扫描中清除数':d['扫描中清除数'], '尾程清除数':d['尾程清除数'],
                     '尾程时间_秒':d['尾程时间_秒'], '尾程路程_米':d['尾程路程_米'],
                     '记录完整性':run['记录完整性'], '已发现源均已清除':d['已发现源均已清除']})
    for number in (3, 4):
        group = [r for r in rows if r['题号'] == number]
        groups.append({'题号':number, '场次':len(group), '合计清除':sum(r['清除数'] for r in group),
                       '具有模型全清证据场次':sum(r['模型全清证据'] for r in group),
                       '全清待核实场次':sum(not r['模型全清证据'] for r in group),
                       '平均虚拟时间_秒':sum(r['虚拟时间_秒'] for r in group)/len(group),
                       '平均路程_米':sum(r['路程_米'] for r in group)/len(group),
                       '合计尾程时间_秒':sum(r['尾程时间_秒'] for r in group),
                       '合计尾程清除数':sum(r['尾程清除数'] for r in group),
                       '合计扫描中清除数':sum(r['扫描中清除数'] for r in group),
                       '合计失败清除次数':sum(r['失败清除次数'] for r in group),
                       '按总时间加权移动占比':sum(r['时间分项_秒']['移动'] for r in group)/sum(r['虚拟时间_秒'] for r in group),
                       '合计请求响应对数':sum(r['记录完整性']['请求数'] for r in group)})
    return {'口径':'已完成的六场正式测试；行为来自唯一时间匹配的原始HTTP记录。正式包未解密，目标数/真值/官方分数未公开。',
            '逐场':rows, '分题汇总':groups,
            '跨题比较限制':'第三问全向与第四问含定向源的地图和任务不同，时间差不能直接归因于算法优劣。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--source-check', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        create_archives()
    inventory = verify_archives(args.source_check)
    runs = collect(inventory, args.verify)
    summary = summarize(runs)
    if args.verify:
        assert read(SUMMARY) == summary
        print('两组30份原始文件、6场正式记录、1178对指令响应及几何/费用/完成证据审计通过。')
    else:
        write(SUMMARY, summary)
        print(json.dumps(summary['分题汇总'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
