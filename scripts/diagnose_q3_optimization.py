#!/usr/bin/env python
"""只读诊断两组已完成演练的优化机会，不执行策略或连接模拟器。"""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/tables/第三问/两组演练评估_s1_s2.json"
OUTPUT = ROOT / "results/tables/第三问/两组演练_优化机会诊断.json"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    provenance = {SOURCE.relative_to(ROOT).as_posix(): sha256(SOURCE.read_bytes()).hexdigest()}
    groups = {}
    for group in ("s1", "s2"):
        selected = [r for r in data["逐局记录"] if r["组"] == group]
        distances, measurements = defaultdict(float), defaultdict(int)
        details = []
        for row in selected:
            path = ROOT / row["关联本地目录"] / "运行结果.json"
            name = path.relative_to(ROOT).as_posix()
            digest = sha256(path.read_bytes()).hexdigest()
            assert digest == data["来源SHA256"][name]
            provenance[name] = digest
            result = json.loads(path.read_text(encoding="utf-8"))
            position = (0, 0)
            radii, sites = {}, defaultdict(set)
            count, length = 0, 0.0
            for event in result["动作记录"]:
                if event["动作"] not in ("检测", "清除"):
                    continue
                reason = event["用途"]
                category = ("搜索站" if reason.startswith("搜索站") else
                            "固定扫描" if reason.startswith("固定扫描站") else reason)
                step = math.dist(position, event["位置"])
                distances[category] += step
                length += step
                position = event["位置"]
                channel = event["频道"]
                if event["动作"] == "检测":
                    measurements[category] += 1
                    # 仅用动作前已经取得的证书；不能用本次或未来观测提前判断。
                    if (reason.startswith("固定扫描站") and
                            radii.get(channel, math.inf) <= 19.5 and len(sites[channel]) >= 2):
                        count += 1
                    if event["结果"] == "direction":
                        radii[channel] = event["外包覆盖圆半径_米"]
                        sites[channel].add(tuple(event["位置"]))
                    elif event["结果"] == "near":
                        radii[channel] = 5.0
                elif event["结果"] == "success":
                    radii.pop(channel, None)
            assert abs(length - result["总路程_米"]) < 1e-5
            details.append({"编号": row["显示编号"], "已具双站与清除证书后的固定检测数": count})
        n = len(selected)
        total = sum(r["已具双站与清除证书后的固定检测数"] for r in details)
        groups[group] = {
            "样本数": n,
            "平均到各类动作位置的移动距离_米": {k: v / n for k, v in distances.items()},
            "平均检测次数_按用途": {k: v / n for k, v in measurements.items()},
            "已具双站与清除证书后的固定检测总数": total,
            "上述检测平均每局次数": total / n,
            "上述检测平均每局对应费用_秒": total * 5 / n,
            "逐局计数": details,
        }
    paths = [Path(__file__), ROOT / "src/cumcm2026_b/q3_strategy.py",
             ROOT / "src/cumcm2026_b/q3_scheme2.py", ROOT / "src/cumcm2026_b/q3_geometry.py"]
    for path in paths:
        provenance[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    output = {
        "口径": "按原日志动作发生前证书识别候选可省检测；移动距离按到达动作的用途归类，不是因果费用拆分",
        "限制": "没有删动作重放或运行优化方案；后续区域、清除位置、频道顺序和路程可能改变，所列检测费用不是净节省预测",
        "分组": groups,
        "来源SHA256": provenance,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
