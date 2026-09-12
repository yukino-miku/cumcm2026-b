#!/usr/bin/env python
"""第四问方案二：19固定站、圆心出发17.1公里最短路线；其他机制沿用方案一。"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cumcm2026_b.q3_protocol import RobotClient, HttpTransport
from cumcm2026_b.q4_local_env import make_case
from cumcm2026_b.q4_scheme2 import SchemeTwo, SchemeTwoConfig


def provenance(config_path):
    files = (list((ROOT / 'src/cumcm2026_b').glob('*.py'))
             + list((ROOT / 'src/cumcm2026_b/q4_scheme2').glob('*.py'))
             + [Path(__file__).resolve(), ROOT / 'configs/q4_scheme2.json'])
    result = {'源码_SHA256': {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                            for p in sorted(files)},
              '实际配置文件_SHA256': sha256(config_path.read_bytes()).hexdigest(),
              '策略入口': 'q4_scheme2'}
    try:
        result['Git提交'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        result['Git工作区干净'] = not bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result['Git提交'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['local', 'http'], default='local')
    parser.add_argument('--robot-id', default=os.environ.get('CUMCM_ROBOT_ID'))
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--count', type=int, choices=range(10, 17), default=13)
    parser.add_argument('--layout', choices=['uniform', 'boundary', 'cluster', 'center'], default='uniform')
    parser.add_argument('--radius', choices=['min', 'max', 'mixed'], default='min')
    parser.add_argument('--error', choices=['hash', 'zero', 'plus', 'minus', 'alternating'], default='hash')
    parser.add_argument('--directional-fraction', type=float, default=.5)
    parser.add_argument('--orientation', choices=['random', 'outward', 'inward', 'tangent'], default='random')
    parser.add_argument('--probe-spacing', type=float, help='优先拉开相邻补测站的间距，默认100米')
    parser.add_argument('--detour-limit', type=float, help='主动补测或完整小范围覆盖的绕行门槛，默认800米')
    parser.add_argument('--no-opportunistic', action='store_true')
    parser.add_argument('--no-route-planning', action='store_true')
    parser.add_argument('--no-direction-feedback', action='store_true')
    parser.add_argument('--no-small-sweep', action='store_true')
    parser.add_argument('--no-task-routing', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/q4_scheme2.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'http' and not args.robot_id:
        parser.error('连接接口需要--robot-id；请先在模拟器选择第四问演练')
    if not 0 <= args.directional_fraction <= 1:
        parser.error('--directional-fraction必须在0和1之间')
    parameters = json.loads(args.config.read_text(encoding='utf-8'))
    for argument, parameter in [('probe_spacing', 'probe_spacing_m'),
                                ('detour_limit', 'localization_detour_limit_m')]:
        if getattr(args, argument) is not None:
            parameters[parameter] = getattr(args, argument)
    for argument, parameter, value in [
            ('no_opportunistic', 'opportunistic', False), ('no_route_planning', 'route_planning', False),
            ('no_direction_feedback', 'direction_feedback', False), ('no_small_sweep', 'small_sweep_limit', 0),
            ('no_task_routing', 'task_routing', False)]:
        if getattr(args, argument):
            parameters[parameter] = value
    if args.no_route_planning:
        parameters['task_routing'] = False
    try:
        config = SchemeTwoConfig(**parameters)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    metadata = provenance(args.config)
    output = args.output or ROOT / 'local-only/第四问' / ('方案二-' + args.mode + '-' + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    environment = make_case(args.seed, args.count, args.layout, args.radius, args.error,
                            args.directional_fraction, args.orientation) if args.mode == 'local' else None
    client = RobotClient(environment if environment is not None else HttpTransport(args.base_url),
                         'local-robot' if environment is not None else args.robot_id, output / '请求响应日志.jsonl')
    client.record({'类型': '运行来源', **metadata, '实际配置': parameters})
    try:
        result = SchemeTwo(client, config).run()
        result['运行来源'] = metadata
        result['运行类型'] = '本地构造，非官方成绩' if environment is not None else 'HTTP接口运行；测试属性以模拟器及官方日志为准'
        # 真值仅在退出后提供给离线评估，策略只接收协议客户端。
        if environment is not None and environment.ended:
            result['离线真值核验'] = environment.truth_for_evaluation()
            result['本地场景参数'] = {k: getattr(args, k) for k in
                ['seed', 'count', 'layout', 'radius', 'error', 'directional_fraction', 'orientation']}
        (output / '运行结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + '\n', encoding='utf-8')
        keys = ['运行类型', '流程正常完成', '全部完成证据', '结束状态', '清除数',
                '虚拟总时间_秒', '总路程_米', '未发现频道', '异常']
        print(json.dumps({k: result[k] for k in keys}, ensure_ascii=False, indent=2))
        if '离线真值核验' in result:
            truth = result['离线真值核验']
            print(json.dumps({'离线实际全清': truth['全部清除'], '离线遗漏频道': truth['遗漏频道']}, ensure_ascii=False))
        print(f'固定站最短路线：17100米；日志与结果：{output}')
        return 0 if result['流程正常完成'] else 1
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
