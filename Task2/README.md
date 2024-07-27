# 任务2
任务要求：The task aims to collect and organize question-answering data from the open-source community to construct an open, multi-domain question-answering dataset, and to implement a question-answering model based on this dataset. By analyzing the behavior and language characteristics of members in the open-source community during the process of asking and answering questions, as well as the diversity and complexity of questions, you will design and train an efficient question-answering model capable of accurately understanding and answering various questions, thereby promoting knowledge sharing and communication within the open-source community.

The relevant code and dataset for this task need to be provided in the repository.

# 数据集介绍








# 模型代码介绍
### 0.环境配置
运行以下代码配置所需的conda enviroment
```bash
conda create -n qagnn python=3.7
source activate qagnn
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.4.0
pip install nltk spacy==2.1.6
python -m spacy download en

# for torch-geometric
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
```

### 1.下载数据




### 2.训练模型
For CommonsenseQA, run
```
./run_qagnn__csqa.sh
```
For OpenBookQA, run
```
./run_qagnn__obqa.sh
```
For MedQA-USMLE, run
```
./run_qagnn__medqa_usmle.sh
```
模型需要两种类型的输入
* `--{train,dev,test}_statements`: 处理jason格式的输入. 主要由 `load_input_tensors` 里的函数 `utils/data_utils.py`.
* `--{train,dev,test}_adj`: 处理为每个问题提取的KG子图 主要由 `load_sparse_adj_data_with_contextnode` 里的函数 `utils/data_utils.py`.

### 3. 评估训练模型
For CommonsenseQA, run
```
./eval_qagnn__csqa.sh
```
Similarly, for other datasets (OpenBookQA, MedQA-USMLE), run `./eval_qagnn__obqa.sh` and `./eval_qagnn__medqa_usmle.sh`.

