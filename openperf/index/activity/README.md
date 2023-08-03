# 活跃度指数
判断一个项目是否长期处于活跃状态对于开发者的技术选型，组件选取，是否参与该项目成为贡献者有着重要的意义。OpenPerf 选择了一种通过统计开发者协作行为数据并加权求和的方式[OpenActivity](https://blog.frankzhao.cn/how_to_measure_open_source_1/)来计算项目活跃度的计算方法。OpenActivity可以作为量化一个项目活跃度的基准单位。

该测试对2023年6月GitHub 中全域活跃项目进行统计与排名，将参与人数与日志增量作为项目活跃度衡量的指标。参与人数表示当月参与项目贡献的开发者数量；日志增量为当月该项目的日志总量。

从下表可以看出，pytorch/pytorch日志增量最高，但OpenActivity仅排第6。NixOS/nixpkgs当月参与的开发者数量较少，但OpenActivity排第1。不同指标下可以看出整体排序结果差异较大。这是由于OpenActivity 与其他两种指标不同，它是根据开发者的协作行为数据加权求和而得出的。OpenPerf提供了2023年6月GitHub全域OpenActivity 排名前10的项目统计结果，以便其他开发者在提出新的活跃度指数类指标时，与当前产出的基准结果进行对比，从而验证指标的合理性。
### 项目活跃度排名对比结果
|仓库|参与人数|日志增量|OpenActivity|
|  ----  | ----  | ----  | ----  |
| NixOS/nixpkgs|1908|33956|5163.91|
|home-assistant/core|3384|18402|4380.23|
|microsoft/vscode|3557|16907|3643.1|
|flutter/flutter|2649|15300|2938.33|
|MicrosoftDocs/azure-docs|1774|9944|2884.33|
|pytorch/pytorch|2102|52636|2839.17|
|odoo/odoo|1285|34123|2470.15|
|dotnet/runtime|862|16641|2298.13|
|godotengine/godot|2247|11204|2114.51|
|microsoft/winget-pkgs|539|26600|1703.35|
