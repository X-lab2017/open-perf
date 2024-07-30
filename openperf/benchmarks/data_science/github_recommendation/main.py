import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import scipy as sp
# import nxviz as nv
from itertools import combinations
from collections import defaultdict
import os

def add_metadata_to_nodes(graph, nodes_data, metadata_columns=['followers', 'popularity', 'genres', 'chart_hits']):
    """
    Add metadata attributes to nodes in the graph based on provided nodes_data DataFrame.
    """
    for node_id in graph.nodes():
        if node_id in nodes_data['name'].values:
            metadata = nodes_data.loc[nodes_data['name'] == node_id, metadata_columns].to_dict('records')[0]
            # Replace missing values in chart_hits by 0
            if pd.isnull(metadata['chart_hits']):
                metadata['chart_hits'] = 0 
            graph.nodes[node_id].update(metadata)
        else:
            print(f"No match found for developer: {node_id}")
    return graph


def plot_network_graph_with_degree_centrality(graph, with_labels=False, alpha=0.7, width=0.3, font_size=6):
    """
    Plot a network graph with node sizes based on 'popularity' attribute and colors based on degree centrality.

    Parameters:
        graph (networkx.Graph): The graph to be plotted.

    Returns:
        None
    """
    degree_centrality = nx.degree_centrality(graph)

    colormap = plt.cm.cool
    sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(degree_centrality.values()), vmax=max(degree_centrality.values())))

    colors = {}
    for node, centrality in degree_centrality.items():
        color = colormap(centrality)
        colors[node] = color

    sizes = nx.get_node_attributes(graph, 'popularity')

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    nx.draw_networkx(graph, node_size=list(sizes.values()), with_labels=with_labels, node_color=list(colors.values()), ax=ax, alpha=alpha, width=width, font_size=font_size)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Degree Centrality')
    plt.savefig('img/graph.jpg')


def add_metadata_centrality(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)

    for node, data in G.nodes(data=True):
        data['degree_centrality']=degree_centrality[node]
        data['betweenness_centrality']=betweenness_centrality[node]
        data['eigenvector_centrality']=eigenvector_centrality[node]
    return G


def get_open_triangles(G, node, popularity_lvl):
    """
    Identifies all open triangles around a specified node in a given graph, 
    considering only adjacent nodes exceeding a specified popularity level. 
    An open triangle is defined as a set of three nodes where two are connected to the third, 
    but not to each other.

    Parameters:
    - G (networkx.Graph): The network graph.
    - node (hashable): The node around which open triangles are to be found.
    - popularity_lvl (int or float): The threshold for considering a node's popularity.

    Returns:
    - list of tuples: Each tuple represents an open triangle, consisting of the node and two adjacent nodes.
    """
    open_triangle_nodes = []
    neighbors = [n for n in G.neighbors(node) if G.nodes[n].get('popularity', 0) > popularity_lvl]
    
    for n1, n2 in combinations(neighbors, 2):
        if not G.has_edge(n1, n2):
            open_triangle_nodes.append((n1, node, n2))
    
    return open_triangle_nodes


def find_top_recommended_pairs(graph):
    """
    Analyzes a given graph to identify the top 10 recommended pairs of nodes (developers) for collaboration. 
    These pairs are not directly connected but share common neighbors in the graph, suggesting potential synergy.
    
    Parameters:
    - graph (nx.Graph): The network graph of developers and collaborations.
    
    Returns:
    - list of tuples: The top 10 recommended pairs of nodes, along with their recommendation strength.
    """
    recommended = defaultdict(int)

    for n in graph.nodes:
        neighbors = set(graph.neighbors(n))
        for n1, n2 in combinations(neighbors, 2):
            if not graph.has_edge(n1, n2):
                recommended[(n1, n2)] += 1

    top10_pairs = sorted(recommended.items(), key=lambda x: x[1], reverse=True)[:10]
    return top10_pairs



if __name__ == '__main__':

    # edges.csv contains collaborative relationships
    edges = pd.read_csv('dataset/edges.csv')
    print(edges.shape)

    # nodes.csv contains various nides informations
    nodes = pd.read_csv('dataset/nodes.csv')
    print(nodes.shape)

    # Data preprocessing
    nodes = nodes.loc[nodes['popularity'] >= nodes['popularity'].quantile(0.99)]
    nodes = nodes.dropna(subset=['name', 'followers'])

    id_to_name = dict(zip(nodes['spotify_id'], nodes['name']))

    edges['id_0'] = edges['id_0'].map(id_to_name)
    edges['id_1'] = edges['id_1'].map(id_to_name)

    # Graph calculation 
    edges.rename(columns={'id_0': 'source', 'id_1': 'target'}, inplace=True)
    
    G = nx.from_pandas_edgelist(edges, 'source', 'target')
    G = add_metadata_to_nodes(G, nodes)
    print(G)

    # Choose subgraph to ease analysis
    top_01 = list(nodes['name'].loc[nodes['popularity']>=nodes['popularity'].quantile(0.75)])
    H = G.subgraph(top_01).copy()
    print(H)

    #Connect subgraph
    print('if the graph is connected:'+str(nx.is_connected(H)))
    nx.draw_networkx(H, node_color='lime', node_size=100, font_size=6)
    plt.savefig('img/graph_with_degree_centrality.jpg')

    Hc = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    print(Hc)

    plot_network_graph_with_degree_centrality(Hc, alpha=0.4, width=0.2)

    # Correlation calculation
    measures = [nx.eigenvector_centrality(Hc), 
                nx.betweenness_centrality(Hc), 
                nx.degree_centrality(Hc)]

    cor = pd.DataFrame.from_records(measures)

    # Highest impact
    e_cent, b_cent, d_cent = cor.T.idxmax()

    print(e_cent, b_cent, d_cent)

    print('J Balvin has collaborated with {} developers in our subgraph'
        .format(len(list(Hc.neighbors('J Balvin')))))

    top_cen = sorted(nx.degree_centrality(Hc).items(), key = lambda x: x[1], reverse=True)[0:3]
    top_bet = sorted(nx.betweenness_centrality(Hc).items(), key = lambda x: x[1], reverse=True)[0:3]
    top_eig = sorted(nx.eigenvector_centrality(Hc).items(), key = lambda x: x[1], reverse=True)[0:3]
    print(top_cen)
    print(top_bet)
    print(top_eig)

    print(nodes.sort_values(['popularity', 'followers'], ascending=False).head(10))

    Hc = add_metadata_centrality(Hc)

    print('There is {} cliques in the subgraph'.format(len(list(nx.find_cliques(Hc)))))
    print('There is {} triangles in the subgraph'.format(len(list(nx.triangles(Hc)))))

    open_triangles = get_open_triangles(Hc, 'Drake', 90)
    print(open_triangles)

    top_recommended_pairs = find_top_recommended_pairs(Hc)
    for pair, strength in top_recommended_pairs:
        print(f"Pair: {pair}, Strength: {strength}")