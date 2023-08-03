# 影响力指数
开源项目影响力的大小取决于多个因素，对开源项目的影响力进行评估，不仅可以帮助开发者和组织确定哪些项目值得投入精力，还可以为开源项目的优化和改进提供针对性意见。OpenPerf 选择了一种加权 PageRank 算法 [OpenRank](https://blog.frankzhao.cn/how_to_measure_open_source_2/) 来计算项目的影响力，并作为量化一个项目影响力的基准单位。

该测试对2023年6月GitHub 中全域活跃项目进行统计与排名，使用了经典的度中心性算法和PageRank算法与OpenRank进行对比，表6列出OpenRank排名前10的对比结果。

由下表可得，MicrosoftDocs/azure-docs项目的度中心性和PageRank值最高，但OpenRank相对其他项目较低，home-assistant/core项目的OpenRank值排第1。由于OpenRank通过加权 PageRank 算法来计算项目的中心度，其计算出的值相对其他指标都偏高。该算法考虑了不同协作行为对项目产生的影响，故与其他指标排序结果不同。OpenPerf提供了2023年6月GitHub全域OpenRank排名前10的项目统计结果，以便其他开发者在提出新的项目影响力指数类指标时，与当前3类影响力指标进行对比。


### 项目影响力排名对比结果
|仓库|度中心性|PageRank|OpenRank|
|  ----  | ----  | ----  | ----  |
|home-assistant/core|0.015660|0.0035|2393.86
|NixOS/nixpkgs|0.008743|0.0008|2207.5
|microsoft/vscode|0.015247|0.003|1960.39
|flutter/flutter|0.012138|0.002|1460.34
|pytorch/pytorch|0.009624|0.0012|1421.18
|MicrosoftDocs/azure-docs|0.239616|0.08|1216.01
|dotnet/runtime|0.004141|0.0006|1181.12
|microsoft/winget-pkgs|0.061954|0.0075|1106.3
|godotengine/godot|0.203330|0.045|1105.51
|odoo/odoo|0.175534|0.043|907.97
