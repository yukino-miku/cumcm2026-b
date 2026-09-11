#!/usr/bin/env python
"""方案一统一入口：默认本地构造；--mode http 才连接已就绪的官方接口。"""
import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cumcm2026_b.q3_protocol import RobotClient, HttpTransport
from cumcm2026_b.q3_local_env import make_case
from cumcm2026_b.q3_routed_strategies import SchemeOneRouted as SchemeOne, RoutedOneConfig as StrategyConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["local", "http"], default="local")
    p.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID"))
    p.add_argument("--base-url", default="http://127.0.0.1:2026")
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--count", type=int, choices=range(10, 17), default=13)
    p.add_argument("--layout", choices=["uniform", "boundary", "cluster", "center"], default="uniform")
    p.add_argument("--radius", choices=["min", "max", "mixed"], default="min")
    p.add_argument("--error", choices=["hash", "zero", "plus", "minus", "alternating"], default="hash")
    p.add_argument("--no-opportunistic", action="store_true")
    p.add_argument("--no-route-planning", action="store_true", help="恢复原版每站处理全部已知目标的调度")
    p.add_argument("--config", type=Path, default=ROOT/"configs/q3_scheme1_routed.json")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.mode == "http" and not args.robot_id: p.error("连接官方接口须提供--robot-id或CUMCM_ROBOT_ID环境变量")
    parameters = json.loads(args.config.read_text(encoding="utf-8"))
    if args.no_opportunistic: parameters["opportunistic"] = False
    if args.no_route_planning: parameters["route_planning"] = False
    config = StrategyConfig(**parameters)
    output = args.output or ROOT/"local-only/第三问"/(args.mode+"-"+uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    local_env = make_case(args.seed, args.count, args.layout, args.radius, args.error) if args.mode == "local" else None
    transport = local_env if local_env is not None else HttpTransport(args.base_url)
    client = RobotClient(transport, "local-robot" if local_env else args.robot_id, output/"请求响应日志.jsonl")
    try:
        result = SchemeOne(client, config).run()
        result["运行类型"] = "本地构造，非官方成绩" if local_env else "HTTP接口运行；演练或正式属性以模拟器显示及官方日志为准"
        if local_env:
            result["离线真值核验"] = local_env.truth_for_evaluation()
            result["本地场景参数"] = {k: getattr(args, k) for k in ["seed", "count", "layout", "radius", "error"]}
        (output/"运行结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ["运行类型", "运行成功", "清除数", "虚拟总时间_秒", "程序运行时间_秒", "异常"]}, ensure_ascii=False, indent=2))
        print(f"日志与结果：{output}")
        return 0 if result["运行成功"] else 1
    finally: client.close()


if __name__ == "__main__": raise SystemExit(main())
