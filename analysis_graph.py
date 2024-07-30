import networkx as nx
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('./repo_graph.json', 'r') as file:
    data = json.load(file)

G = nx.node_link_graph(data)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# 计算度中心性
degree_centrality = nx.degree_centrality(G)
# 找出度中心性最高的前N个节点
top_n = 10
top_nodes = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:top_n]
print("Top {} nodes by degree centrality:".format(top_n))
for node, centrality in top_nodes:
    print(f"Node: {node}, Degree Centrality: {centrality}")

# 聚类系数
clustering_coefficient = nx.clustering(G)
average_clustering_coefficient = nx.average_clustering(G)
print("Average Clustering Coefficient:", average_clustering_coefficient)

# # 计算平均路径长度
# if nx.is_connected(G):
#     average_path_length = nx.average_shortest_path_length(G)
#     print("Average Path Length:", average_path_length)
# else:
#     print("The graph is not connected. Cannot compute average path length.")
#
# # 计算直径
# if average_path_length is not None:
#     diameter = nx.diameter(G)
#     print("Diameter:", diameter)
# else:
#     print("Cannot compute diameter because the graph is not connected.")

# 布局图
pos = nx.spring_layout(G)

# 使用matplotlib绘制图
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='k')
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, arrows=False)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

# 突出显示度中心性最高的节点
for node, centrality in top_nodes:
    nx.draw_networkx_nodes(G, pos, node, node_color='red', node_size=500)

# 显示图
plt.title('Repository Collaboration Network')
plt.axis('off')  # 关闭坐标轴
plt.savefig('./repo_graph.png')