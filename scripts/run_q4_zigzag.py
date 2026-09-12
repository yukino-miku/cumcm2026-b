#!/usr/bin/env python
"""折线方案已撤回：兼容旧入口，执行原第四问几何补测方案。"""
from run_q4 import main

if __name__ == '__main__':
    print('折线方案已按用户要求撤回；转入当前 run_q4.py，具体策略与门槛由所选配置决定。', flush=True)
    raise SystemExit(main())
