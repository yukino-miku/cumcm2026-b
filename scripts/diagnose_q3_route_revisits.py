"""只读提取S11-2/S11-5延后决策证据；依赖本机已关联的私有日志，不运行策略。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'results/tables/第三问/新版演练评估_s11_s22.json'
OUTPUT = ROOT / 'results/tables/第三问/方案一折返诊断_S11-2_S11-5.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gather():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    result = {
        '性质': '原运行事件回溯；不是新策略试验，不估计已实现节省',
        '口径': '末次固定检测之后的路程含必要补测和清除，不等于可避免路程；圆半径为当时保守外包覆盖圆半径',
        '来源SHA256': {REPORT.relative_to(ROOT).as_posix(): digest(REPORT)},
        '逐局诊断': [],
    }
    for row in report['逐局记录']:
        if row['显示编号'] not in {'S11-2', 'S11-5'}:
            continue
        source = ROOT / row['关联本地目录'] / '运行结果.json'
        name = source.relative_to(ROOT).as_posix()
        source_hash = digest(source)
        assert source_hash == report['来源SHA256'][name], name
        result['来源SHA256'][name] = source_hash
        run = json.loads(source.read_text(encoding='utf-8'))
        events = run['动作记录']
        last_fixed = max(e['步骤'] for e in events if e.get('用途', '').startswith('搜索站'))
        first_positive, latest_positive, delays, clears = {}, {}, [], []
        fixed, current, length, tail_length = None, [0., 0.], 0., 0.
        for event in events:
            channel = event.get('频道')
            if event.get('用途', '').startswith('搜索站'):
                fixed = int(event['用途'].removeprefix('搜索站'))
            if event['动作'] in {'检测', '清除'}:
                distance = math.dist(current, event['位置'])
                length += distance
                tail_length += distance if event['步骤'] > last_fixed else 0.
                current = event['位置']
            if event['动作'] == '检测' and event.get('结果') in {'direction', 'near'}:
                snapshot = {k: event[k] for k in ['步骤', '用途', '结果', '外包覆盖圆半径_米'] if k in event}
                first_positive.setdefault(channel, snapshot)
                latest_positive[channel] = snapshot
            if event['动作'] == '延后目标':
                delays.append({
                    **{k: event[k] for k in ['步骤', '频道', '原因', '本次测向绕行_米', '阈值_米'] if k in event},
                    '最近固定站序号': fixed,
                    '当时最近正观测': latest_positive[channel],
                })
            if event['动作'] == '清除' and event['结果'] == 'success':
                clears.append({'频道': channel, '清除步骤': event['步骤'],
                               '首次发现': first_positive[channel],
                               '清除前最近正观测': latest_positive[channel]})
        assert math.isclose(length, row['总路程_米'], abs_tol=1e-6)
        assert math.isclose(tail_length, row['按阶段路程_米']['末次固定检测之后'], abs_tol=1e-6)
        assert len(clears) == row['成功清除数']
        tail = [e for e in clears if e['清除步骤'] > last_fixed]
        result['逐局诊断'].append({
            '编号': row['显示编号'], '总路程_米': length,
            '总虚拟时间_秒': row['总虚拟时间_秒'],
            '末次固定检测步骤': last_fixed,
            '末次固定检测之后路程_米': tail_length,
            '末次固定检测之后路程占比': tail_length / length,
            '末段清除目标': tail,
            '延后记录': delays,
        })
    assert len(result['逐局诊断']) == 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='重新读取私有来源并核对已有诊断，不写文件')
    args = parser.parse_args()
    result = gather()
    if args.check:
        assert result == json.loads(OUTPUT.read_text(encoding='utf-8'))
    else:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in result['逐局诊断']:
        print(f"{row['编号']}：末段 {row['末次固定检测之后路程_米']/1000:.3f} 公里，"
              f"占 {row['末次固定检测之后路程占比']:.2%}；"
              f"末段清除 {[e['频道'] for e in row['末段清除目标']]}；"
              f"延后 {len(row['延后记录'])} 次")
    print('折返回溯核验通过；未运行策略或连接模拟器。')


if __name__ == '__main__':
    main()
