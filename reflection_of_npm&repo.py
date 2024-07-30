import pandas as pd
import numpy as np
from clickhouse_driver import Client
import networkx as nx
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取npm包相关的数据
file_path = "C:/Users/lenovo/Desktop/2024-05-13-16-54-33_EXPORT_CSV_13529018_459_0.csv"
npm_packages_df = pd.read_csv(file_path)
npm_packages_df = npm_packages_df.dropna()

## 提取出上述url的owner和repo名对（用以后续爬取贡献信息）
def get_owner_repo(url):
    # 定义正则表达式模式
    pattern = r'\/([^\/]+)\/([^\/]+)\.git(?:\/|$)'
    match = re.search(pattern, url)
    if match:
        # o = match.group(1)
        r = match.group(2)
        # return o, r       ## 可以返回出来（调用函数时，申明这对变量）
        return r
    else:
        return None, None

## 取得每一行的repo，变成数据框的一整列
npm_packages_df["repo_name"] = npm_packages_df["repository_url"].apply(get_owner_repo)
print(npm_packages_df)

# 读取repo_co相关的数据
repo_co_df = pd.read_csv("contributors_of_repo.csv")

## 现在我们有了repo_co_df和npm_packages_df两个数据框，他们具有共同的字段repo_name
## 因此，我们实际上可以使用类似于数据库的JOIN算子，对他们进行连接，从而形成更大的数据集
    ## 我们想从repo_co映射到npm，所以repo_co作为左表，npm作为右表
# result = repo_co_df.join(npm_packages_df)
reflection_of_repo_to_npm = pd.merge(repo_co_df, npm_packages_df, on="repo_name", how='inner')

## 数据集reflection_of_npm_and_repo则表示了从repo到npm的映射
reflection_of_repo_to_npm
reflection_of_repo_to_npm.to_csv("reflection_of_repo_to_npm.csv")