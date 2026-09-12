"""明确映射旧运行/绘图源码；只接受当前字节或指定归档的精确SHA256。"""
from hashlib import sha256

ARCHIVES = {
    'scripts/run_q4.py': ['docs/第四问/历史代码/run_q4_ccb0ad5.py.txt', 'docs/第四问/历史代码/run_q4_aae5f71.py.txt'],
    'scripts/build_q4_inner13_assets.py': 'docs/第四问/历史代码/build_q4_inner13_assets_ccb0ad5.py.txt',
    'scripts/analyze_q4_practice_4.py': 'docs/第四问/历史代码/analyze_q4_practice_4_ccb0ad5.py.txt',
    **{name: 'docs/第四问/历史代码/' + name.rsplit('/', 1)[-1].replace('.py', '_aae5f71.py.txt')
       for name in ['scripts/run_q4_zigzag.py', 'scripts/q4_archive_sources.py',
                    'scripts/build_q4_spacing_assets.py', 'scripts/build_q4_zigzag_assets.py',
                    'scripts/analyze_q4_spacing_practice.py', 'tests/test_q4_zigzag.py']},
}


def matches_source(root, relative, expected):
    candidates = [root/relative]
    if relative in ARCHIVES:
        values = ARCHIVES[relative]
        candidates.extend(root/path for path in ([values] if isinstance(values, str) else values))
    return any(p.is_file() and sha256(p.read_bytes()).hexdigest() == expected for p in candidates)


def verify_manifest(root, expected):
    for relative, digest in expected.items():
        assert matches_source(root, relative, digest), relative
