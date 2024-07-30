import pandas as pd
import numpy as np
from clickhouse_driver import Client
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from networkx.readwrite import json_graph
import json
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
                database='supply_chain'
                )
tables = client.execute('SHOW TABLES')

npm_record_query = 'SELECT * FROM npm_records'
npm_record_result = client.execute(npm_record_query)
npm_dependencies_query = 'SELECT * FROM npm_dependencies LIMIT'
npm_dependencies_result = client.execute(npm_dependencies_query)

print("Get data successfully!")

npm_packages_df = pd.DataFrame(npm_record_result, columns=['package_id', 'name', 'version', 'description', 'repository_type', 'repository_url', 'license', 'homepage', 'time'])
npm_dependencies_df = pd.DataFrame(npm_dependencies_result, columns=['package_id', 'dependency_name', 'dependency_verison', 'type'])

# 保留共同package_id行
common_package_ids = pd.merge(npm_packages_df[['package_id']], npm_dependencies_df[['package_id']], on='package_id', how='inner')['package_id']
npm_packages_df = npm_packages_df[npm_packages_df['package_id'].isin(common_package_ids)]
npm_dependencies_df = npm_dependencies_df[npm_dependencies_df['package_id'].isin(common_package_ids)]

print("save into dataframe")
print(len(common_package_ids))
print(len(npm_packages_df))
print(len(npm_dependencies_df))

# 构建一个包id到名称的映射字典
package_id_to_name = pd.Series(npm_packages_df.name.values, index=npm_packages_df.package_id).to_dict()

def build_graph_chunk(df_chunk):
    G_chunk = nx.DiGraph()
    for index, row in df_chunk.iterrows():
        package_id = row['package_id']
        dependency_name = row['dependency_name']
        if package_id in package_id_to_name.keys():
            package_name = package_id_to_name[package_id]
            if dependency_name in package_id_to_name.values():
                G_chunk.add_edge(package_name, dependency_name)
    return G_chunk

def merge_graphs(graphs):
    merged_graph = nx.DiGraph()
    for G in graphs:
        merged_graph = nx.compose(merged_graph, G)
    return merged_graph

def parallel_build_graph(df, package_id_to_name, num_workers=10):
    len_df = len(df)
    chunk_size = len_df // num_workers
    graphs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len_df, chunk_size):
            df_chunk = df.iloc[i:i + chunk_size]
            futures.append(executor.submit(build_graph_chunk, df_chunk))
        for future in futures:
            G_chunk = future.result()
            graphs.append(G_chunk)
    G = merge_graphs(graphs)
    return G

G = parallel_build_graph(npm_dependencies_df, package_id_to_name)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

json_data = json_graph.node_link_data(G)
with open('./npm_graph.json', 'w') as file:
    json.dump(json_data, file)

pos = nx.spring_layout(G)

# 使用matplotlib绘制图
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='k')
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, arrows=False)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

# 显示图
plt.title('Npm Dependency Network')
plt.axis('off')  # 关闭坐标轴
plt.savefig('./npm_graph.png')

# # 创建一个有向图
# G = nx.DiGraph()
#
# for package_id, package_name in package_id_to_name.items():
#     G.add_node(package_name)
#
# num_nodes = G.number_of_nodes()
# print("Number of nodes:", num_nodes)
#
# # 添加边（依赖关系）
# for index, row in npm_dependencies_df.iterrows():
#     package_id = row['package_id']
#     dependency_name = row['dependency_name']
#
#     if package_id in package_id_to_name.keys():
#         package_name = package_id_to_name[package_id]
#         if dependency_name in package_id_to_name.values():
#             G.add_edge(package_name, dependency_name)
