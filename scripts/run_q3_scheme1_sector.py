#!/usr/bin/env python
"""方案一扇形版入口：默认本地构造，显式--mode http才连接模拟器。"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cumcm2026_b.q3_local_env import make_case
from cumcm2026_b.q3_protocol import RobotClient, HttpTransport
from cumcm2026_b.q3_sector_strategy import SchemeOneSector, SectorConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['local', 'http'], default='local')
    parser.add_argument('--robot-id', default=os.environ.get('CUMCM_ROBOT_ID'))
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--count', type=int, choices=range(10, 17), default=13)
    parser.add_argument('--layout', choices=['uniform', 'boundary', 'cluster', 'center'], default='uniform')
    parser.add_argument('--radius', choices=['min', 'max', 'mixed'], default='min')
    parser.add_argument('--error', choices=['hash', 'zero', 'plus', 'minus', 'alternating'], default='hash')
    parser.add_argument('--no-reuse', action='store_true', help='关闭机会复用扫描；扇形覆盖检查及按需补漏仍执行')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/q3_scheme1_sector.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'http' and not args.robot_id:
        parser.error('HTTP模式须提供--robot-id或CUMCM_ROBOT_ID环境变量')
    parameters = json.loads(args.config.read_text(encoding='utf-8'))
    if args.no_reuse: parameters['opportunistic'] = False
    config = SectorConfig(**parameters)
    output = args.output or ROOT/'local-only/第三问'/('sector-'+args.mode+'-'+uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    env = make_case(args.seed, args.count, args.layout, args.radius, args.error) if args.mode == 'local' else None
    client = RobotClient(env if env is not None else HttpTransport(args.base_url),
                         'local-robot' if env is not None else args.robot_id, output/'请求响应日志.jsonl')
    try:
        result = SchemeOneSector(client, config).run()
        result['运行类型'] = '本地构造，非官方成绩' if env is not None else 'HTTP接口运行；演练或正式属性以模拟器显示为准'
        files = sorted((ROOT/'src/cumcm2026_b').glob('*.py'))+[Path(__file__)]
        result['运行源码SHA256'] = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
        try:
            result['Git提交'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            result['Git工作区有改动'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            result['Git提交'], result['Git工作区有改动'] = None, None
        if env is not None:
            result['离线真值核验'] = env.truth_for_evaluation()
            result['本地场景参数'] = {k: getattr(args, k) for k in ['seed', 'count', 'layout', 'radius', 'error']}
        (output/'运行结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({k: result[k] for k in ['运行类型', '运行成功', '清除数', '虚拟总时间_秒', '程序运行时间_秒',
                                               '已验收扇形', '扇形调度统计', '异常']}, ensure_ascii=False, indent=2))
        print(f'日志与结果：{output}')
        return 0 if result['运行成功'] else 1
    finally:
        client.close()


if __name__ == '__main__': raise SystemExit(main())
