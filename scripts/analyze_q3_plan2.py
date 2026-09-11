#!/usr/bin/env python
"""第三问方案二的离线几何与费用计算；不连接模拟器，不执行在线策略。"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cumcm2026_b.q1_geometry import solve_bearings

TABLE = ROOT / "results/tables/第三问/方案二计算结果.json"
FIGURES = ROOT / "results/figures/第三问"
R, Q, RHO = 1800.0, 1000.0, 1150.0
NOTICE = "离线方案计算，非模拟器成绩；覆盖保证来自解析证明，热力图为10米格网示意"


def stations_for(n: int, rho: float, inner: float | None) -> np.ndarray:
    ring = rho * np.column_stack((np.cos(np.arange(n)*2*np.pi/n), np.sin(np.arange(n)*2*np.pi/n)))
    start = [[0.0, 0.0]] + ([[inner, 0.0]] if inner is not None else [])
    return np.vstack((start, ring))


def route_cost(n: int, rho: float, count: int) -> dict:
    length = rho + (n-1)*2*rho*math.sin(math.pi/n)
    return {"站点数": count, "扫描路程_米": length, "移动_秒": length/5,
            "检测次数": 20*count, "切换次数": 19*count,
            "检测及切换_秒": 119*count, "扫描总虚拟时间_秒": length/5+119*count}


def endpoint_bound(n: int, rho: float, split: float, inner: float) -> float:
    delta = 2*math.pi/n
    f = lambda r: r*r+rho*rho-2*r*rho*math.cos(delta)
    return max(split+inner, math.sqrt(max(f(split), f(R))))


def enclosing_circle(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """对少量凸多边形顶点枚举支持点；结果覆盖所有输入顶点。"""
    candidates = [v for v in vertices]
    candidates += [(a+b)/2 for a, b in itertools.combinations(vertices, 2)]
    for a, b, c in itertools.combinations(vertices, 3):
        matrix = 2*np.vstack((b-a, c-a))
        if abs(np.linalg.det(matrix)) > 1e-8:
            center = a + np.linalg.solve(matrix, [np.dot(b-a, b-a), np.dot(c-a, c-a)])
            candidates.append(center)
    center = min(candidates, key=lambda x: np.linalg.norm(vertices-x, axis=1).max())
    return center, float(np.linalg.norm(vertices-center, axis=1).max())


def setup_style() -> None:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10,
                         "svg.fonttype": "path", "svg.hashsalt": "q3-plan2",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    (FIGURES/f"{name}.png").write_bytes(buffer.getvalue())
    svg = FIGURES/f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def decorate(ax, stations, labels=False):
    ax.add_patch(Circle((0, 0), R, fill=False, color="#663B8A", lw=2, label="1800米目标边界"))
    ax.scatter(stations[2:, 0], stations[2:, 1], c="#163550", s=25, zorder=5, label="外围站")
    ax.scatter([0, 500], [0, 0], c=["#CB573A", "#008576"], s=60, marker="s", zorder=6, label="原点与内部站")
    if labels:
        for i, (x, y) in enumerate(stations):
            label = "O" if i == 0 else "I" if i == 1 else f"P{i-1}"
            offset = (9, 9) if i < 2 else (10*x/RHO, 10*y/RHO)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha="center", va="center", fontsize=9, color="#142A3B", zorder=7)
    ax.set(xlim=(-1920, 1920), ylim=(-1920, 1920), xlabel="东向坐标 x（米）", ylabel="北向坐标 y（米）")
    ax.set_aspect("equal")
    ax.grid(alpha=.13)


def main() -> None:
    stations = stations_for(12, RHO, 500.0)
    names = ["O", "I"]+[f"P{i}" for i in range(1, 13)]
    cost = route_cost(12, RHO, len(stations))
    assert abs(np.linalg.norm(np.diff(stations, axis=0), axis=1).sum()-cost["扫描路程_米"]) < 1e-8
    bound = endpoint_bound(12, RHO, 400.0, 500.0)
    assert bound < Q
    beta = 2*math.asin(Q/R)
    beta_inner = 2*math.acos(R/(2*Q))
    candidates = []
    for n in range(11, 25):
        delta = 2*math.pi/n
        q_safe = 990.0
        if R*math.sin(delta) > q_safe:
            continue
        rho = R*math.cos(delta)-math.sqrt(q_safe*q_safe-R*R*math.sin(delta)**2)
        has_inner = rho > q_safe
        candidate_bound = endpoint_bound(n, rho, 500, 400) if has_inner else max(rho, math.sqrt(R*R+rho*rho-2*R*rho*math.cos(delta)))
        assert candidate_bound <= q_safe+1e-8
        candidates.append({"外围站数": n, "外围半径_米": rho, "内部补充站": [400, 0] if has_inner else None,
                           "连续双覆盖距离上界_米": candidate_bound,
                           **route_cost(n, rho, n+1+int(has_inner))})
    small_margin_rho = R/(2*math.cos(2*math.pi/14))
    small_margin = {"外围站数": 14, "外围半径_米": small_margin_rho,
                    "连续双覆盖距离上界_米": small_margin_rho,
                    **route_cost(14, small_margin_rho, 15)}

    grid = np.arange(-1800, 1800+10, 10, dtype=float)
    xx, yy = np.meshgrid(grid, grid)
    inside = xx*xx+yy*yy <= R*R+1e-7
    points = np.column_stack((xx[inside], yy[inside]))
    vec = stations[None, :, :]-points[:, None, :]
    distances = np.linalg.norm(vec, axis=2)
    counts = np.sum(distances <= Q+1e-8, axis=1)
    second = np.partition(distances, 1, axis=1)[:, 1]
    assert counts.min() >= 2
    assert second.max() <= bound+1e-8
    near = distances.min(axis=1) <= 5
    best_sine = np.zeros(len(points))
    for i, j in itertools.combinations(range(len(stations)), 2):
        valid = (distances[:, i] <= Q+1e-8) & (distances[:, j] <= Q+1e-8) & (distances[:, i] > 5) & (distances[:, j] > 5)
        denominator = distances[:, i]*distances[:, j]
        cross = abs(vec[:, i, 0]*vec[:, j, 1]-vec[:, i, 1]*vec[:, j, 0])
        sine = np.divide(cross, denominator, out=np.zeros(len(points)), where=valid)
        best_sine = np.maximum(best_sine, np.minimum(sine, 1))
    best_angle = np.degrees(np.arcsin(best_sine))
    best_angle[near] = np.nan
    min_index = int(np.nanargmin(best_angle))

    samples = []
    targets = {"外圆正东": [1800, 0], "外圆15度": [R*math.cos(math.pi/12), R*math.sin(math.pi/12)],
               "内部非共线": [400, 300], "中心共线": [-100, 0], "外围近站": [1150, 10]}
    for title, target in targets.items():
        target = np.asarray(target, dtype=float)
        ds = np.linalg.norm(stations-target, axis=1)
        positive = (ds <= Q+1e-8) & (ds > 5)
        bearings = np.degrees(np.arctan2(target[1]-stations[positive, 1], target[0]-stations[positive, 0])) % 360
        solved = solve_bearings(stations[positive], bearings, 1.0)
        entry = {"名称": title, "真实目标_米": target.tolist(), "接收半径_米": Q,
                 "误差设定": "各站真实测向误差为0度，求交仍保留正负1度误差界；未模拟HTTP角度舍入",
                 "有方向站": [name for name, yes in zip(names, positive) if yes], "示向度_度": bearings.tolist(),
                 "角域类型": solved.kind, "角域直径_米": solved.diameter, "角域顶点_米": solved.vertices.tolist()}
        if solved.kind not in ("unbounded", "empty"):
            center, radius = enclosing_circle(solved.vertices)
            assert np.linalg.norm(solved.vertices-center, axis=1).max() <= radius+1e-9
            entry.update({"角域最小覆盖圆心_米": center.tolist(), "角域最小覆盖圆半径_米": radius,
                          "仅角域是否保证20米清除": radius <= 20})
        samples.append(entry)

    indistinguishable = []
    for target in [[-149., 0.], [-6., 0.]]:
        ds = np.linalg.norm(stations-np.asarray(target), axis=1)
        codes = ["near" if d <= 5 else "direction" if d <= Q+1e-8 else "no_signal" for d in ds]
        assert codes == ["direction", "direction"]+["no_signal"]*12
        indistinguishable.append({"目标": target, "响应状态": codes, "两站示向度": [180, 180], "距离_米": ds.tolist()})

    setup_style()
    def field(values):
        result = np.full(xx.shape, np.nan)
        result[inside] = values
        return result
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.9))
    for i, station in enumerate(stations):
        axes[0].add_patch(Circle(station, Q, fill=False, color="#6BA9AF", alpha=.32, lw=.8,
                                label="1000米保证接收边界" if i == 0 else None))
    axes[0].plot(stations[:, 0], stations[:, 1], color="#C65A2E", lw=1.9, label="扫描路线", zorder=4)
    for a, b in zip(stations[:-1], stations[1:]):
        axes[0].annotate("", a+.65*(b-a), a+.35*(b-a), arrowprops={"arrowstyle": "->", "color": "#C65A2E", "lw": 1.7}, zorder=5)
    axes[0].add_patch(Circle((0, 0), 400, fill=False, color="#008576", ls="--", lw=1.2, label="证明分区：半径400米"))
    decorate(axes[0], stations, True)
    axes[0].set_title("14站布局与固定扫描路线")
    axes[0].legend(loc="upper right", fontsize=8)
    cmap = ListedColormap(["#B9DCE4", "#75B9CD", "#3E92B4", "#24668C", "#153F69"])
    im = axes[1].pcolormesh(xx, yy, field(np.minimum(counts, 6)), cmap=cmap, norm=BoundaryNorm(np.arange(1.5, 7.5), cmap.N), shading="auto", rasterized=True)
    decorate(axes[1], stations)
    axes[1].set_title("1000米接收范围覆盖次数（最少2次）")
    cb = fig.colorbar(im, ax=axes[1], shrink=.75, ticks=range(2, 7))
    cb.ax.set_yticklabels(["2", "3", "4", "5", "≥6"])
    cb.set_label("覆盖次数")
    fig.suptitle("方案二：原点＋内部站(500,0)＋半径1150米的正十二边形", fontsize=15)
    fig.text(.5, .02, NOTICE, ha="center", fontsize=9, color="#566674")
    fig.tight_layout(rect=(0, .06, 1, .93))
    save(fig, "方案二_双重覆盖与路线")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.9))
    im = axes[0].pcolormesh(xx, yy, field(Q-second), cmap="YlGnBu", shading="auto", rasterized=True)
    decorate(axes[0], stations)
    axes[0].set_title(f"第二近站的接收余量（全域最少{Q-bound:.2f}米）")
    fig.colorbar(im, ax=axes[0], shrink=.75).set_label("1000米－第二近站距离（米）")
    im = axes[1].pcolormesh(xx, yy, field(best_angle), cmap="magma", vmin=0, vmax=90, shading="auto", rasterized=True)
    decorate(axes[1], stations)
    axes[1].set_title("可接收站对的最大锐交会角（0°表示退化）")
    axes[1].scatter([-100], [0], marker="x", s=80, color="#00C6B1", zorder=8)
    axes[1].annotate("共线反例：(-100,0)", (-100, 0), xytext=(-1550, -1250), arrowprops={"arrowstyle": "->", "color": "#008576"}, color="#008576", fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=.75).set_label("真实几何锐交会角（度）")
    fig.suptitle("双重覆盖保证接收次数；定位仍需检查交会角与实际可行域", fontsize=15)
    fig.text(.5, .035, "右图按接收半径1000米计算；near点留白；真实位置仅用于离线分析，在线不可使用", ha="center", fontsize=9, color="#566674")
    fig.text(.5, .008, NOTICE, ha="center", fontsize=9, color="#566674")
    fig.tight_layout(rect=(0, .07, 1, .93))
    save(fig, "方案二_覆盖裕度与交会角")

    report = {"性质": "离线解析方案及构造几何验证，未运行模拟器，未实施在线算法",
              "参数": {"目标圆半径_米": R, "最小接收半径_米": Q, "外围半径_米": RHO, "内部站_米": [500, 0], "格网步长_米": 10},
              "站点_按扫描顺序": [{"名称": name, "坐标_米": s.tolist()} for name, s in zip(names, stations)],
              "主方案": {**cost, "相邻外围站距离_米": 2*RHO*math.sin(math.pi/12), "连续第二近距离最大值_米": bound,
                         "保证接收余量_米": Q-bound, "相对七站扫描增加_秒": cost["扫描总虚拟时间_秒"]-2273},
              "站数下界": {"任意站覆盖目标边界的最大弧长_度": math.degrees(beta),
                          "能覆盖原点的站的最大边界弧长_度": math.degrees(beta_inner),
                          "11站且原点双覆盖时的边界弧长总上界_度": math.degrees(2*beta_inner+9*beta),
                          "双重边界覆盖必需弧长_度": 720, "全域双覆盖站数下界": 12,
                          "说明": "仅下界，未证明12站可构造，也未证明本方案14站全局最优"},
              "10米余量的对称候选族": candidates, "小余量对照": small_margin,
              "格网验证": {"域内点数": len(points), "最少覆盖次数": int(counts.min()), "最多覆盖次数": int(counts.max()),
                           "最大第二近距离_米": float(second.max()), "最小最佳锐交会角_度": float(np.nanmin(best_angle)),
                           "一个退化格点_米": points[min_index].tolist(), "说明": "格网为独立数值抽查，不替代连续覆盖证明"},
              "零真实误差定位构造": samples, "全14站反馈相同的不同目标": indistinguishable,
              "来源": {"脚本": str(Path(__file__).relative_to(ROOT)), "脚本SHA256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "第一问源码SHA256": sha256((ROOT/'src/cumcm2026_b/q1_geometry.py').read_bytes()).hexdigest(),
                       "Python": platform.python_version(), "NumPy": np.__version__, "SciPy": scipy.__version__, "Matplotlib": matplotlib.__version__}}
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"主方案": report["主方案"], "格网验证": report["格网验证"],
                      "构造": [{k: v for k, v in s.items() if k in ["名称", "有方向站", "角域类型", "角域直径_米", "角域最小覆盖圆半径_米"]} for s in samples],
                      "输出": str(TABLE.relative_to(ROOT))}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
