# Activity Index
Determining whether a project remains active over the long term is crucial for developers in terms of technology selection, component choices, and deciding whether to contribute to the project. OpenPerf has adopted a method called [OpenActivity](https://blog.frankzhao.cn/how_to_measure_open_source_1/) for calculating project activity by statistically analyzing and weighting the sum of developer collaboration behavior data. OpenActivity serves as a benchmark unit for quantifying a project's activity level.

This test involved statistics and ranking of globally active projects on GitHub in June 2023, using the number of participants and the log increment as indicators of project activity. The number of participants indicates the count of developers who contributed to the project during the month; the log increment represents the total number of logs for the project in the month.

From the table below, it can be observed that pytorch/pytorch had the highest log increment but ranked only sixth in OpenActivity. NixOS/nixpkgs had fewer developers participating in the month but ranked first in OpenActivity. There are significant differences in the overall ranking results under different indicators. This is because OpenActivity, unlike the other two indicators, is derived from a weighted sum of developer collaboration behavior data. OpenPerf has provided the top 10 project statistics for GitHub global OpenActivity in June 2023, allowing other developers to compare with the current baseline results when proposing new indices of activity, thereby verifying the validity of the indices.

### Project Activity Ranking Comparison
| Repository | Number of Participants | Log Increment | OpenActivity |
|------------|------------------------|---------------|--------------|
| NixOS/nixpkgs | 1908 | 33956 | 5163.91 |
| home-assistant/core | 3384 | 18402 | 4380.23 |
| microsoft/vscode | 3557 | 16907 | 3643.1 |
| flutter/flutter | 2649 | 15300 | 2938.33 |
| MicrosoftDocs/azure-docs | 1774 | 9944 | 2884.33 |
| pytorch/pytorch | 2102 | 52636 | 2839.17 |
| odoo/odoo | 1285 | 34123 | 2470.15 |
| dotnet/runtime | 862 | 16641 | 2298.13 |
| godotengine/godot | 2247 | 11204 | 2114.51 |
| microsoft/winget-pkgs | 539 | 26600 | 1703.35 |

#### References
1. [How to Measure Open Source](https://blog.frankzhao.cn/how_to_measure_open_source_1/)
2. [Open Digger GitHub Metrics](https://github.com/X-lab2017/open-digger/blob/master/src/metrics/indices.ts#L109)
