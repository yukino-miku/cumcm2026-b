#!/usr/bin/env python
"""已撤回扇形策略；兼容旧命令，转入此前方案一路线版。"""
import sys

from run_q3_scheme1 import main


if __name__ == '__main__':
    print('扇形策略已按用户要求撤回；本命令改为运行此前路线版 run_q3_scheme1.py。', flush=True)
    # 兼容旧入口的关闭复用参数，实际采用路线版的机会检测开关。
    sys.argv[1:] = ['--no-opportunistic' if arg == '--no-reuse' else arg for arg in sys.argv[1:]]
    raise SystemExit(main())
