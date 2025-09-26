
#import library
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict



#read the edgelist from medium and large datasets
#G_medium = nx.read_edgelist('snacs2025-student4251938-medium.tsv', create_using=nx.DiGraph(), delimiter='\t')
#G_large = nx.read_edgelist('snacs2025-student4251938-large.tsv', create_using=nx.DiGraph(), delimiter='\t')
#print number of edges in both datasets
#print(len(G_medium.edges()), "edges in medium dataset")
#print(len(G_large.edges()), "edges in large dataset")
# print("End of Question 2.1")

# #num_nodes_medium = G_medium.number_of_nodes()
# #num_nodes_large = G_large.number_of_nodes()
# #print(f"Number of nodes in medium dataset: {num_nodes_medium}")
# #print(f"Number of nodes in large dataset: {num_nodes_large}")
# print("End of Question 2.2")

# #indegree and outdegree

# def plot_degree_distributions(Graph, dataset):
#     indgree = [degree for node, degree in Graph.in_degree()]
#     outdegree = [degree for node, degree in Graph.out_degree()] 
#     plt.figure(figsize=(12, 5))

#     #indegree distribution
#     plt.subplot(1, 2, 1)
#     plt.hist(indgree, bins=50, log=True, color='skyblue', edgecolor='black')
#     plt.title(f'In-Degree Distribution ({dataset} dataset)')
#     plt.xlabel('In-Degree') 
#     plt.ylabel('Frequency (log scale)')

#     #outdegree distribution
#     plt.subplot(1, 2, 2)
#     plt.hist(outdegree, bins=50, log=True, color='salmon', edgecolor='black')
#     plt.title(f'Out-Degree Distribution ({dataset} dataset)')
#     plt.xlabel('Out-Degree')
#     plt.ylabel('Frequency (log scale)')

#     plt.tight_layout()
#     plt.savefig(f'degree_distribution_{dataset}.png')
#     plt.show()

#print("Plotting degree distributions...")
#plot_degree_distributions(G_medium, 'snacs2025-student4251938-medium')
#plot_degree_distributions(G_large, 'snacs2025-student4251938-large')
#print("End of Question 2.3")

#strongly and weekly connected components

# def get_component_values(graph):
#     # Weakly connected components
#     weak_components = list(nx.weakly_connected_components(graph))
#     num_weak = len(weak_components)
#     largest_weak = graph.subgraph(max(weak_components, key=len))
#     weak_nodes = largest_weak.number_of_nodes()
#     weak_edges = largest_weak.number_of_edges()

#     # Strongly connected components
#     strong_components = list(nx.strongly_connected_components(graph))
#     num_strong = len(strong_components)
#     largest_strong = graph.subgraph(max(strong_components, key=len))
#     strong_nodes = largest_strong.number_of_nodes()
#     strong_edges = largest_strong.number_of_edges()

#     #returning the values for both weakly and strongly connected components
#     return {
#         "num_weak": num_weak,
#         "num_strong": num_strong,
#         "weak_nodes": weak_nodes,
#         "weak_edges": weak_edges,
#         "strong_nodes": strong_nodes,
#         "strong_edges": strong_edges
#     }

# Analyze
#medium_stats = get_component_values(G_medium)
#large_stats = get_component_values(G_large)

# print("Medium Dataset:")
# #print(medium_stats)
# print("\nLarge Dataset:")
# #print(large_stats)
# print("End of Question 2.4")

#average clustering coefficient
# Convert to undirected for clustering approximation
#G_medium_undirected = G_medium.to_undirected()
#G_large_undirected = G_large.to_undirected()

# Compute average clustering coefficient
#clustering_medium = nx.average_clustering(G_medium_undirected)
#clustering_large = nx.average_clustering(G_large_undirected)

#print("Medium Dataset - Avg Clustering Coefficient:", clustering_medium)
#print("Large Dataset - Avg Clustering Coefficient:", clustering_large)
# print("End of Question 2.5")

#distance distribution for largest weekly conneceted componenets

# def plot_distance_distribution(graph, dataset):
#     #wcc- weakly connected components nodes
#     largest_wcc = max(nx.weakly_connected_components(graph), key=len)
#     subgraph = graph.subgraph(largest_wcc).to_undirected()

#     distance_counts = Counter()
#     for node in subgraph.nodes():
#         lengths = nx.single_source_shortest_path_length(subgraph, node)
#         for target, dist in lengths.items():
#             if node != target:
#                 distance_counts[dist] += 1
    
#     distances = sorted(distance_counts.items())
#     x, y = zip(*distances)

#     plt.figure(figsize=(8, 6))
#     plt.bar(x, y, color='mediumseagreen', edgecolor='black')
#     plt.ylabel('Frequency (log scale)')
#     plt.xlabel('Distance')
#     plt.title(f'Distance Distribution in Largest WCC ({dataset} dataset)')
#     plt.tight_layout()
#     plt.savefig(f'distance_distribution_{dataset}.png')
#     plt.show()


#plot_distance_distribution(G_medium, 'snacs2025-student4251938-medium.tsv')
#plot_distance_distribution(G_large, 'snacs2025-student4251938-large.tsv')
#rint("End of Question 2.6")

# Largest WCC
#largest_wcc_medium = G_medium_undirected.subgraph(max(nx.weakly_connected_components(G_medium), key=len))
#largest_wcc_large = G_large_undirected.subgraph(max(nx.weakly_connected_components(G_large), key=len))

# Average shortest path length
#avg_distance_medium = nx.average_shortest_path_length(largest_wcc_medium)
#avg_distance_large = nx.average_shortest_path_length(largest_wcc_large)

#print("Medium dataset - Avg Distance:", avg_distance_medium)
#print("Large dataset - Avg Distance:", avg_distance_large)
#rint("End of Question 2.7")


#BONUS QUESTION
import igraph as ig
import random


print("Starting analysis of the huge dataset...")
# # Load the huge dataset
path = "/vol/share/groups/liacs/scratch/SNACS/huge.tsv"


in_degree = defaultdict(int)
out_degree = defaultdict(int)
unique_nodes = set()
num_edges = 0

with open(path, "r") as f:
    num_edges = 0
    for line in f:
        num_edges += 1
        # Extract unique nodes from each line
        src, dst = line.strip().split("\t")[:2]
        unique_nodes.add(src)
        unique_nodes.add(dst)
        out_degree[src] += 1
        in_degree[dst] += 1
		

print(f"Number of edges: {num_edges}")
print(f"Number of unique nodes: {len(unique_nodes)}")
print("End of Question 2.1 and 2.2")

# Prepare degree lists
in_degrees = [in_degree[node] for node in unique_nodes]
out_degrees = [out_degree[node] for node in unique_nodes]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(in_degrees, bins=50, log=True, color='skyblue', edgecolor='black')
plt.title('In-Degree Distribution (huge dataset)')
plt.xlabel('In-Degree')
plt.ylabel('Frequency (log scale)')

plt.subplot(1, 2, 2)
plt.hist(out_degrees, bins=50, log=True, color='salmon', edgecolor='black')
plt.title('Out-Degree Distribution (huge dataset)')
plt.xlabel('Out-Degree')
plt.ylabel('Frequency (log scale)')

plt.tight_layout()
plt.savefig('degree_distribution_huge.png')
plt.show()
print("End of Question 2.3")


# Parameters for sampling
sample_size = 10000  # Number of edges to sample
sampled_edges = []
total_edges = 0

# First, count total edges
with open(path, "r") as f:
    for _ in f:
        total_edges += 1

# Randomly select line numbers to sample
sample_indices = set(random.sample(range(total_edges), min(sample_size, total_edges)))

# Read and collect sampled edges
with open(path, "r") as f:
    for idx, line in enumerate(f):
        if idx in sample_indices:
            src, dst = line.strip().split("\t")[:2]
            sampled_edges.append((src, dst))

# Build igraph from sampled edges
g_sample = ig.Graph(directed=True)
nodes = set([src for src, dst in sampled_edges] + [dst for src, dst in sampled_edges])
g_sample.add_vertices(list(nodes))
g_sample.add_edges(sampled_edges)

# Analyze components
weak_components = g_sample.clusters(mode="weak")
num_weak = len(weak_components)
largest_weak = weak_components.giant()
print(f"Sampled: Number of weakly connected components: {num_weak}")
print(f"Sampled: Largest weakly connected component: {largest_weak.vcount()} nodes, {largest_weak.ecount()} edges")

strong_components = g_sample.clusters(mode="strong")
num_strong = len(strong_components)
largest_strong = strong_components.giant()
print(f"Sampled: Number of strongly connected components: {num_strong}")
print(f"Sampled: Largest strongly connected component: {largest_strong.vcount()} nodes, {largest_strong.ecount()} edges")
print("End of Question 2.4 (sampled)")

