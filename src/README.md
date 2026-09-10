# 源代码

第一问实现位于 [q1_geometry.py](cumcm2026_b/q1_geometry.py)，提供示向度转半平面、交集分类、顶点与直径计算、同直径圆覆盖判定，以及 JSON 命令行入口。

接口和复现方法见 [第一问阅读导航](../docs/第一问/阅读导航.md)。绘图及构造实验独立位于 [build_q1_assets.py](../scripts/build_q1_assets.py)，统一运行入口为 [run-q1.ps1](../scripts/run-q1.ps1)。

第二问新增 [q2_physical_geometry.py](cumcm2026_b/q2_physical_geometry.py)，处理线段与圆弧物理域、连续距离及解析候选点；[q2_active_localization.py](cumcm2026_b/q2_active_localization.py) 提供单场景第一问求解、鲁棒目标、二维搜索和近优候选点。核心代码不依赖绘图。字段及第三问复用方法见 [第二问阅读导航](../docs/第二问/阅读导航.md)。
