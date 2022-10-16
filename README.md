# OpenPerf
Benchmark suit for large scale socio-technical datasets in open collaboration

该项目受如下几个工作的启发：

1. **ImageNet**：https://image-net.org/

The ImageNet project is a large visual database designed for use in visual object recognition software research. More than 14 million images have been hand-annotated by the project to indicate what objects are pictured and in at least one million of the images, bounding boxes are also provided. Since 2010, the ImageNet project runs an annual software contest, the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), where software programs compete to correctly classify and detect objects and scenes.

2. **DataPerf**：https://dataperf.org/

A new benchmark suite for machine learning datasets and data-centric algorithms proposed by researchers from Coactive.AI, ETH Zurich, Google, Harvard University, Landing.AI, Meta, Stanford University, and TU Eindhoven. DataPerf is a suite of benchmarks that evaluate the quality of training and test data, and the algorithms for constructing or optimizing such datasets, such as core set selection or labeling error debugging, across a range of common ML tasks such as image classification. We plan to leverage the DataPerf benchmarks through challenges and leaderboards.

3. **Open Graph Benchmark**：https://ogb.stanford.edu/

The Open Graph Benchmark (OGB) is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on graphs. In addition, the research team also proposed OGB Large-Scale Challenge (OGB-LSC), a collection of three real-world datasets for facilitating the advancements in large-scale graph ML.

4. **Papers With Code: The latest in Machine Learning**：https://paperswithcode.com/

The mission of Papers with Code is to create a free and open resource with Machine Learning papers, code, datasets, methods and evaluation tables.

5. **Codabench**：https://www.codabench.org/

Codabench is a platform allowing you to flexibly specify a benchmark. First you define tasks, e.g. datasets and metrics of success, then you specify the API for submissions of code (algorithms), add some documentation pages, and [CLICK] your benchmark is created, ready to accept submissions of new algorithms. Participant results get appended to an ever-growing leaderboard.

6. **AIOps Challenge（国际智能运维挑战赛）**：https://aiops-challenge.com/

是将人工智能的能力与运维相结合，基于已有的运维数据（日志、监控信息、应用信息等）并通过机器学习的方式来进一步解决自动化运维没办法解决的问题。即以大数据平台和机器学习（算法平台）为核心从各个监控系统中抽取数据、面向用户提供服务，以此来提升运维效率。

7. **The SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection)**：https://sparse.tamu.edu/

The SuiteSparse Matrix Collection (formerly known as the University of Florida Sparse Matrix Collection), is a large and actively growing set of sparse matrices that arise in real applications. The Collection is widely used by the numerical linear algebra community for the development and performance evaluation of sparse matrix algorithms. It allows for robust and repeatable experiments: robust because performance results with artificially-generated matrices can be misleading, and repeatable because matrices are curated and made publicly available in many formats.

8. **Workshop on Graph Learning Benchmarks**：https://graph-learning-benchmarks.github.io/glb2022

GLB is the Workshop of the Graph Learning Benchmarks. Inspired by the conference tracks in the computer vision and natural language processing communities that are dedicated to establishing new benchmark datasets and tasks, we call for contributions that establish novel ML tasks on novel graph-structured data which have the potential to (i) identifying systematic failure modes of existing GNNs and providing new technical challenges for the development of new models which highlight diverse future directions, (ii) raising the attention of the synergy of graph learning, and (iii) crowdsourcing benchmark datasets for various tasks of graph ML.

OpenPerf 的主要目标：
- 构建一系列的数据集，特别是基于图的数据集，用于各类数据科学分析方法的评测
- 构建数据集上的不同挑战与任务，大部分来自真实的开源治理与社区运营的场景
- 构建不同任务上的评价指标，用于科学的性能评测，并能够指导现实问题
- 构建自动化的任务评测平台
- 举办国际化的挑战赛


