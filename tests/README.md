# 验证

第二问新增 [物理几何测试](test_q2_physical_geometry.py)与[鲁棒选点测试](test_q2_active_localization.py)，全库现为 84 项。统一入口 `scripts/run-q2.ps1` 还执行结果来源、连续可靠条件、第一问交叉核验和图表链接验收；实际记录见[第二问验证与复现记录](../docs/第二问/验证与复现记录.md)。

第一问共有 41 项测试：

- [test_q1_geometry.py](test_q1_geometry.py)：输入与方向约定、区域分类、已知几何解、覆盖反例、数值变换和近乎平行的边界情形。
- [test_q1_reference_polygons.py](test_q1_reference_polygons.py)：12 组可由角域交会实现的正多边形，与独立解析顶点、直径和覆盖结论比较。

在项目根目录运行 `pwsh -NoProfile -File .\scripts\run-q1.ps1`，先测试，再生成结果与图表。详细验证记录见 [第一问验证与复现记录](../docs/第一问/验证与复现记录.md)。原始材料完整性由 `scripts/verify-materials.ps1` 检查，与算法测试分开。
