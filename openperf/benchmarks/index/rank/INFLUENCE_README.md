# Influence Index
The influence of an open source project depends on multiple factors. Assessing the influence of open source projects not only helps developers and organizations determine which projects are worth investing in but also provides targeted suggestions for optimization and improvement of open source projects. OpenPerf has chosen a weighted PageRank algorithm, [OpenRank](https://blog.frankzhao.cn/how_to_measure_open_source_2/), to calculate the influence of projects and serve as a benchmark unit for quantifying a project's influence.

This test involved statistics and ranking of globally active projects on GitHub in June 2023, using classic degree centrality and PageRank algorithms for comparison with OpenRank. Table 6 lists the comparison results of the top 10 rankings by OpenRank.

From the table below, the MicrosoftDocs/azure-docs project has the highest degree centrality and PageRank values but a relatively low OpenRank compared to other projects, while the home-assistant/core project ranks first in OpenRank. Since OpenRank calculates the centrality of projects using a weighted PageRank algorithm, its values tend to be higher compared to other metrics. This algorithm takes into account the impact of different collaborative behaviors on projects, hence the difference in ranking results with other metrics. OpenPerf provides the top 10 project statistics for GitHub global OpenRank in June 2023, allowing other developers to compare when proposing new indices of project influence.

### Project Influence Ranking Comparison
| Repository | Degree Centrality | PageRank | OpenRank |
|------------|-------------------|----------|----------|
| home-assistant/core | 0.015660 | 0.0035 | 2393.86 |
| NixOS/nixpkgs | 0.008743 | 0.0008 | 2207.5 |
| microsoft/vscode | 0.015247 | 0.003 | 1960.39 |
| flutter/flutter | 0.012138 | 0.002 | 1460.34 |
| pytorch/pytorch | 0.009624 | 0.0012 | 1421.18 |
| MicrosoftDocs/azure-docs | 0.239616 | 0.08 | 1216.01 |
| dotnet/runtime | 0.004141 | 0.0006 | 1181.12 |
| microsoft/winget-pkgs | 0.061954 | 0.0075 | 1106.3 |
| godotengine/godot | 0.203330 | 0.045 | 1105.51 |
| odoo/odoo | 0.175534 | 0.043 | 907.97 |

#### References
1. [How to Measure Open Source](https://blog.frankzhao.cn/how_to_measure_open_source_2/)
2. [Open Digger GitHub Metrics](https://github.com/X-lab2017/open-digger/blob/master/src/metrics/indices.ts#L21)
