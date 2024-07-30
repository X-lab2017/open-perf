# Open-Source-Collaboration-Network
Open source repository collaboration network and npm artifact library dependency network mapping dataset

# 各文件说明：
### 注：为了使用和读取方便，所有程序文件都成对地以py和ipynb形式各保存了一份！
#### co-net_request.ipynb<br/>
    - 对Github APIs爬取数据的测试文件。因为爬数据的工作时间较长，为保证工作无失误，先在少量的仓库上尝试爬取一点作为测试。<br/>
#### get_repo_co.ipynb <br/>
    - 针对npm数据所涉及的所有github仓库，爬取其贡献者信息（形成list），为仓库协作网络的构建做准备<br/>
#### gen_npm_graph.ipynb <br/>
    - 基于npm信息数据以及npm的依赖关系数据建立网络<br/>
#### npm_net.ipynb（已弃用）<br/>
    - 同上。实际上这份文件缺了对图数据地持久化操作和并行化运行，上一个文件对此进行了补充！<br/>
#### get_repo_co.ipynb<br/>
    - 基于仓库协作的贡献者信息数据来建立仓库协作网络<br/>
#### reflection_of_npm&repo.ipynb<br/>
    - 利用join操作，制作数据集reflection_of_npm_and_repo，表示了从repo到npm的映射关系。<br/>
#### repo_graph.png 和 npm_graph.png
    - 分别是协作网络和npm网络的可视化图

# Task_Intro（任务介绍）
### 目标
此数据集旨在映射npm包注册表与相应的开源仓库之间的关系。它旨在解决由于个人贡献和仓库名称更改导致的npm注册表中元数据不完整或过时的挑战，从而促进这些网络的准确预测和映射。

### 内容
两个网络无法完全映射，但两个网络的子集可以有相应的关系，并且可以根据npm包信息中的repo_url字段进行映射。

### 开源仓库协作网络：
    节点：各个仓库repo                                   //////代表个别开发者或团队。
    边：代表协作关系，包括提交、审阅和讨论等贡献。
    属性：包括贡献数量、贡献的性质（代码、文档等）以及协作持续时间等指标。

### npm工件库依赖网络：
    节点：代表单独的npm包。
    边：代表依赖链接，即一个包依赖于另一个包。
    属性：包括版本号、更新频率和受欢迎程度指标（下载量、描述）。

### 数据收集方法：
1. 协作网络的数据通过公共API从流行的源代码托管平台如GitHub、GitLab和Bitbucket收集。您也可以直接下载opendigger提供的样本数据集，并建议比较一年的行为数据。https://github.com/X-lab2017/open-digger/blob/master/sample_data/README.md

2. npm工件库依赖网络的数据从npm注册表的公共API提取，重点是package.json文件以映射依赖关系。您可以在npm.org上进行爬取。以下是提供的全局npm库及其依赖关系：
    npm依赖项：npm_dependencies.zip 7.15M
    npm包：npm_packages.zip 69.28M

### 潜在用例：
获取两个网络的指标：度、聚类系数、平均路径长度、直径、中心性、密度、模块性、连通分量等。

### 可视化两个网络映射
    通过检查依赖链及其对软件可靠性的影响来研究软件生态系统的弹性。
    评估软件开发实践随时间的趋势。

### 格式
数据集以适合机器学习和网络分析的格式提供，如表格数据的CSV和结构化元数据的JSON。

### 输出结果
    包含开源仓库协作网络和npm工件库依赖网络的完整数据集。
    数据集的使用说明，详细说明数据项、来源、收集和处理方法。
    数据分析报告，总结关键发现和见解。