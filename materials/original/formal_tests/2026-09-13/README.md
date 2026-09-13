# 六场正式测试原始归档

对应用户提供的`q3form`和`q4form`，每题正式次数1、2、3。两份ZIP共30份原文件，每份保留原始字节；压缩归档可避免Git文本换行转换影响日志SHA256。

| 原始压缩包 | 正式材料 | 对应本地运行材料 |
|---|---|---|
| [q3form.zip](q3form.zip) | 3个启动JSON、3个正式JLOG、3个结果JSON | F3-1至F3-3，各含原始请求响应JSONL与运行结果JSON |
| [q4form.zip](q4form.zip) | 3个启动JSON、3个正式JLOG、3个结果JSON | F4-1至F4-3，各含原始请求响应JSONL与运行结果JSON |

ZIP内`q3form/`或`q4form/`保存用户文件的原名；`http/F3-1/`等保存唯一关联的本地记录。可在单独目录解压查看，勿覆盖模拟器源目录。[归档清单](归档清单.json)记录全部文件原始路径、包内路径、字节数、SHA256，以及队号/退出时间的唯一关联关系。

正式包正文保持加密，没有解密或验证官方签名。完整原包包含队号、封装票据与加密头；本轮没有使用其发起额外操作。现有GitHub仓库查询为公开，本轮未修改可见性，按用户上传正式材料和完整指令过程的要求保存。

完整分析、六张轨迹、逐指令中文表与派生请求响应见[正式测试报告](../../../../docs/正式测试/2026-09-13_第三四问正式测试分析.md)。派生JSONL保留每对原始请求和响应，不用报告中的简写替代原始字段。

在仓库根目录复核（无需本机模拟器目录）：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_formal_tests.py --verify
.\.venv\Scripts\python.exe -X utf8 .\scripts\build_formal_test_assets.py --verify
```

若原始源目录仍在本机，第一个命令追加`--source-check`可逐字节与原文件核对。本轮没有重新进入模拟器，没有消耗新的正式测试次数。
