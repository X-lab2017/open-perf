# 开源社区情绪分类

### 研究背景
当前基于开源领域评论文本的情感分析任务相对于餐厅和电商等热门领域较少，开发者的情感会影响任务质量，生产力，创造力，团队和谐以及工作满意度。通过对评论的情感分析可以获取开发者针对项目具体某个方面的行为和见解，有利于项目健康的发展，达到提高开发人员工作效率的目的。

### 任务描述
Github大多数的评论均为中性，如何在海量评论文本中挖掘包含开发者观点的评论？不同类型的评论文本表达出的观点也会不同，同时开发者也会针对开源社区的不同方面发表观点，如何获取细粒度的开发者情感也是该任务的难点。

### 任务难点
情感的多样性：常见的情感分类任务以二分类三分类为主，在GitHub中部分开发者的情感是相对复杂的，例如，乐观，厌恶，愤怒，认同，质疑等，通过细粒度的分析可以挖掘更多有价值的观点信息。

评价对象的多样性：开发者往往会针对开源项目某一具体方面进行讨论，例如代码重构，代码风格，安全性，文档可读性等等，如何获取不同方面的评论文本进行情感分析也是难点之一。

#### 参考资料
1. Rishi D. Affective sentiment and emotional analysis of pull request comments on github[D]. University of Waterloo, 2017.
2. Kaur R, Chahal K K, Saini M. Analysis of Factors Influencing Developers' Sentiments in Commit Logs: Insights from Applying Sentiment Analysis[J]. e-Informatica Software Engineering Journal, 2022, 16(1): 220102.