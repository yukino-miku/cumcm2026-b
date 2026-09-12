"""明确映射旧运行/绘图源码；只接受当前字节或指定归档的精确SHA256。"""
from hashlib import sha256

ARCHIVES = {
    'scripts/run_q4.py': 'docs/第四问/历史代码/run_q4_ccb0ad5.py.txt',
    'scripts/build_q4_inner13_assets.py': 'docs/第四问/历史代码/build_q4_inner13_assets_ccb0ad5.py.txt',
    'scripts/analyze_q4_practice_4.py': 'docs/第四问/历史代码/analyze_q4_practice_4_ccb0ad5.py.txt',
}


def matches_source(root, relative, expected):
    candidates = [root/relative]
    if relative in ARCHIVES:
        candidates.append(root/ARCHIVES[relative])
    return any(p.is_file() and sha256(p.read_bytes()).hexdigest() == expected for p in candidates)
