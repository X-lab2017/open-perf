import pandas as pd
from copy import deepcopy
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_contr_df = pd.read_csv('./contributors_of_repo.csv')

def str2lst(s):
    str_list = s.strip('[]').split(',')
    return [item.strip("' ") for item in str_list]

def filter_bot(lst):
    return [x for x in lst if not x.endswith('[bot]')]

# 筛除bot账户
repo_contr_df_nobot = deepcopy(repo_contr_df)
repo_contr_df_nobot['developers'] = repo_contr_df.apply(lambda row: filter_bot(str2lst(row['developers'])),
                                                        axis=1)

G = nx.Graph()

for index, row in repo_contr_df_nobot.iterrows():
    repo1 = row['repo_name']
    developers = set(row['developers'])
    for other_index, other_row in repo_contr_df_nobot.iterrows():
        if other_index != index:
            repo2 = other_row['repo_name']
            # 交集不为空表示存在共同贡献者
            if len(developers.intersection(set(other_row['developers']))) > 0:
                G.add_edge(repo1, repo2)

nx.draw(G, node_color='lightblue', edge_color='gray')
plt.savefig('./repo_co.png')

print(1)
