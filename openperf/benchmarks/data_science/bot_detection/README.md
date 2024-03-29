# 开源自动化机器人识别与分类

### 研究背景
合作作为一种社会现象，在软件开发生命周期中的作用越来越重要。当前流行的社交编码平台，如GitHub，Bit-Bucket和GitLab，提供了为开发者共享工作空间的环境，然而，大规模的合作也带来了仓库维护者的巨大工作压力，需要他们完成与贡献者的沟通，审核源代码，处理贡献者许可协议问题，阐述项目准则，执行测试和构建代码，合并拉取请求等多项工作。为了减轻这些重复任务的负担，开源软件项目最近开始利用各种软件机器人来简化他们的运营。机器人的应用也带来了一系列的问题，包括冒充、垃圾邮件、偏见和安全风险等。因此，许多开源研究人员需要识别开源软件机器人账户和行为。

### 任务描述
如何设计和实现一个模型，以识别并分类开源软件项目中的机器人行为？该模型需要能够准确地识别开源软件项目中的机器人行为，并能够根据他们的行为模式和目标对他们进行有效的分类。除了高精度的预测性能，该模型也需要具有较强的可解释性和合理性。

### 任务难点
行为模式的多样性：不同的机器人有不同的功能和目标，这就导致他们的行为模式也多样化。机器人可以完成自动回复用户问题、代码审查、代码合并等任务。这种多样性使得机器人行为的识别和分类变得困难。  
行为模式的复杂性：例如，一个代码审查机器人可能需要分析代码的复杂度，代码的风格，代码的正确性等多个方面。这种复杂性使得机器人行为的识别和分类需要高度的专业知识和精确的模型。  
混合行为的处理：有些机器人可能会和人类开发者一起参与到项目的开发中，这就会产生混合行为。需要模型能够对其进行区分和处理。

### 数据集
该任务选择了GHTorrent数据集中2021年3月至2022年3月最活跃的仓库，以确保泛化能力和准确性。同时将活动数据超过100条日志的账户识别为“活跃账户”数据集，从全局账户中随机选择了一部分账户作为“随机账户”数据集。为了扩大数据集并确保与BIMAN和BoDeGha算法的比较实验的可信度，对BIMAN和BoDegHa的数据进行了处理，获取了他们的账户的GitHub ID，并选择了过去一年内活动的账户（活动数据大于10条日志）。将"活跃账户"、"随机账户"、"BIMAN账户"和"BoDeGha账户"的数据合并成一个数据集，称为"混合账户"。然后，对"混合账户"的数据进行清理，选择了17个相关特征以确保数据的全面性。最后经过科学的标签标注流程和一个可视化标注系统，尽可能的提高标签标注的准确性，构成了OSS机器人分类数据集。

### 评价指标
为了评估机器人识别模型的性能，本节采用了一系列标准的机器学习评估指标，包括精确度（Accuracy）、查准率（Precision）、查全率（Recall）、F1分数（F1-score）以及AUC值（Area Under the ROC Curve）。准确度、精确度和召回率主要评估了模型的分类能力，即模型正确识别正负类的能力；F1分数是精确度和召回率的综合指标，它能在一定程度上平衡精确度和召回率的权重；AUC值则反映了模型在面对不同分类阈值时的性能表现。

### 模型实验

OSS机器人识别任务中，OpenPerf采用了多种机器学习模型进行实验，包括逻辑回归（LogisticRegression）、决策树（DecisionTreeClassifier）、支持向量机（SVM）、高斯朴素贝叶斯（GaussianNB）、K近邻（KNeighborsClassifier）、随机森林（RandomForestClassifier），以及专门针对OSS机器人识别任务设计的模型BotHunter、BoDeGHa和BotHawk。最终结果如下表。下图可以看出不同算法在不同评价指标下的对比情况。

|Model|Accuracy|Precision|Recall|F1-score|AUC|
|  ----  | ----  | ----  | ----  | ----  | ----  |
|LogisticRegression|0.9234|0.4427|0.5376|0.4856|0.5760|
|DecisionTree|0.7995|0.2188|0.7707|0.3408|0.5024|
|SVM|0.8936	|0.3495|0.6767|0.4609|0.5414|
|GaussianNB|0.9548|0.7514|0.4887|0.5923|0.5319|
|KNeighborsClassifier|0.8309|0.2472|0.7406|0.3706|0.5119|
|RandomForest|0.8817|0.3441|0.8383|0.4880|0.6486|
|BotHunter|0.8649|0.2200|0.7528|0.3405|0.5512|
|BoDeGHa|0.8286|0.2354|0.7910|0.3628|0.5049|
|BotHawk|0.8799|0.8930|0.8715|0.8821|0.9472|

![Alt text](result.png)

#### 参考资料
1. Bi F, Zhu Z, Wang W, Xia X, Khan H A, Pu P. BotHawk: An Approach for Bots Detection in Open Source Software Projects[J]. 2023. arXiv preprint arXiv:2307.13386.
