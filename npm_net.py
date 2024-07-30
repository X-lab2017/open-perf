import pandas as pd
import numpy as np
from clickhouse_driver import Client
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# # 读取数据
# npm_packages_df = pd.read_csv('/OpenSource/npm_data/2024-05-13-16-54-33_EXPORT_CSV_13529018_459_0.csv')
# npm_dependencies_df = pd.read_csv('/OpenSource/npm_data/2024-05-13-17-23-22_EXPORT_CSV_13529969_552_0.csv')
# print(npm_packages_df.shape)
# print(npm_dependencies_df.shape)

# 从clickhouse读取数据
client = Client(host='cc-uf6764sn662413tc9.public.clickhouse.ads.aliyuncs.com',
                user='liangchen',
                password='Liangchen123',
<<<<<<< HEAD
                database='supply_chain')
=======
                database='supply_chain'
                )
>>>>>>> 31a322e7e828c6b5b38d9ce840af227d00a95972
tables = client.execute('SHOW TABLES')

npm_record_query = 'SELECT * FROM npm_records'
npm_record_result = client.execute(npm_record_query)
npm_dependencies_query = 'SELECT * FROM npm_dependencies'
npm_dependencies_result = client.execute(npm_dependencies_query)

<<<<<<< HEAD
## 包信息 和 包依赖 的df
=======
>>>>>>> 31a322e7e828c6b5b38d9ce840af227d00a95972
npm_packages_df = pd.DataFrame(npm_record_result, columns=['package_id', 'name', 'version', 'description', 'repository_type', 'repository_url', 'license', 'homepage', 'time'])
npm_dependencies_df = pd.DataFrame(npm_dependencies_result, columns=['package_id', 'dependency_name', 'dependency_verison', 'type'])

# 构建一个包id到名称的映射字典
package_id_to_name = pd.Series(npm_packages_df.name.values, index=npm_packages_df.package_id).to_dict()

# 创建一个有向图
G = nx.DiGraph()

for package_id, package_name in package_id_to_name.items():
    G.add_node(package_name)

# 添加边（依赖关系）
for index, row in npm_dependencies_df.iterrows():
    package_id = row['package_id'] ## 取出id
    dependency_name = row['dependency_name']

    if package_id in package_id_to_name.keys(): # 存在性判断
        package_name = package_id_to_name[package_id] ### 根据id取出name
        if dependency_name in package_id_to_name.values(): # 存在性判断
            G.add_edge(package_name, dependency_name) #### id和name连起来。

print("Add edges over.")

num_edges = G.number_of_edges()
<<<<<<< HEAD
print("Number of edges:", num_edges) ## 边数

isolated_nodes = list(nx.isolates(G))# 注: 去除没有依赖的边
G.remove_nodes_from(isolated_nodes)
num_nodes = G.number_of_nodes()
print("Number of nodes:", num_nodes) ## 节点数
=======
print("Number of edges:", num_edges)

isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)
num_nodes = G.number_of_nodes()
print("Number of nodes:", num_nodes)
>>>>>>> 31a322e7e828c6b5b38d9ce840af227d00a95972


# # 可视化依赖网络
# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(G, k=0.1)
# nx.draw(G, pos, with_labels=True, node_size=1, font_size=8, node_color='lightblue', edge_color='gray', arrows=True, alpha=0.8)
# plt.title('npm Dependency Network')
# plt.savefig('./network.png')
# plt.close()

# # 分析网络
# degree_centrality = nx.degree_centrality(G)
# clustering_coefficient = nx.clustering(G.to_undirected())
# average_path_length = nx.average_shortest_path_length(G) if nx.is_weakly_connected(G) else None
# diameter = nx.diameter(G) if nx.is_weakly_connected(G) else None
#
# print("Degree Centrality: ", degree_centrality)
# print("Clustering Coefficient: ", clustering_coefficient)
# print("Average Path Length: ", average_path_length)
# print("Diameter: ", diameter)
