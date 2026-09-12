#!/usr/bin/env python
"""折线方案已撤回：兼容旧入口，执行原第四问几何补测方案。"""
from run_q4 import main

if __name__ == '__main__':
    print('折线方案已按用户要求撤回；转入原几何方案 run_q4.py，默认补测间距100米、绕行门槛400米。', flush=True)
    raise SystemExit(main())
