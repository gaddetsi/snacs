Number of edges: 912118714
Number of unique nodes: 8112742

/vol/home/s4251938/Desktop/snacs/ass1/snacs/SNACA1.py:239: DeprecationWarning: Graph.clusters() is deprecated; use Graph.connected_components() instead
  weak_components = g_sample.clusters(mode="weak")
Sampled: Number of weakly connected components: 9832
Sampled: Largest weakly connected component: 8 nodes, 7 edges
/vol/home/s4251938/Desktop/snacs/ass1/snacs/SNACA1.py:245: DeprecationWarning: Graph.clusters() is deprecated; use Graph.connected_components() instead
  strong_components = g_sample.clusters(mode="strong")
Sampled: Number of strongly connected components: 19832
Sampled: Largest strongly connected component: 1 nodes, 0 edges
End of Question 2.4 (sampled)

# g = ig.Graph.Read_Edgelist(path, directed=True)

# print(f"Number of edges: {g.ecount()}")
# print(f"Number of nodes: {g.vcount()}")

# # Degree distributions
# in_degrees = g.indegree()
# out_degrees = g.outdegree()

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.hist(in_degrees, bins=50, log=True, color='skyblue', edgecolor='black')
# plt.title('In-Degree Distribution (huge)')
# plt.xlabel('In-Degree')
# plt.ylabel('Frequency (log scale)')

# plt.subplot(1, 2, 2)
# plt.hist(out_degrees, bins=50, log=True, color='salmon', edgecolor='black')
# plt.title('Out-Degree Distribution (huge)')
# plt.xlabel('Out-Degree')
# plt.ylabel('Frequency (log scale)')
# plt.tight_layout()
# plt.savefig('degree_distribution_huge.png')
# plt.show()

# # Connected components
# weak_components = g.clusters(mode="weak")
# strong_components = g.clusters(mode="strong")

# print(f"Weakly connected components: {len(weak_components)}")
# print(f"Strongly connected components: {len(strong_components)}")

# largest_weak = weak_components.giant()
# largest_strong = strong_components.giant()

# print(f"Largest weakly connected component: {largest_weak.vcount()} nodes, {largest_weak.ecount()} edges")
# print(f"Largest strongly connected component: {largest_strong.vcount()} nodes, {largest_strong.ecount()} edges")
# # Memory-efficient sampling for distance distribution and average shortest path length
# sample_size = 100  # Much smaller sample size
# subgraph = largest_weak  # Keep as directed for efficiency
# node_indices = list(range(subgraph.vcount()))
# sample_nodes = random.sample(node_indices, min(sample_size, len(node_indices)))

# distance_counts = Counter()
# total_path_length = 0
# total_pairs = 0
# for v in sample_nodes:
#     lengths = subgraph.shortest_paths_dijkstra(source=v)[0]
#     for dist in lengths:
#         if dist > 0 and dist < float('inf'):
#             distance_counts[dist] += 1
#             total_path_length += dist
#             total_pairs += 1

# distances = sorted(distance_counts.items())
# if distances:
#     x, y = zip(*distances)
#     plt.figure(figsize=(8, 6))
#     plt.bar(x, y, color='mediumseagreen', edgecolor='black')
#     plt.ylabel('Frequency (log scale)')
#     plt.xlabel('Distance')
#     plt.title('Distance Distribution in Largest WCC (huge, sampled, dijkstra)')
#     plt.tight_layout()
#     plt.savefig('distance_distribution_huge_sampled_dijkstra.png')
#     plt.show()

# if total_pairs > 0:
#     avg_distance = total_path_length / total_pairs
#     print(f"Sampled average shortest path length in largest WCC: {avg_distance}")
# else:
#     print("No paths found in sample.")

# # Average clustering coefficient
# clustering_coeff = g.transitivity_avglocal_undirected()
# print(f"Average clustering coefficient: {clustering_coeff}")

# # Distance distribution in largest WCC
# subgraph = largest_weak.as_undirected()
# distance_counts = Counter()
# for v in range(subgraph.vcount()):
#     lengths = subgraph.shortest_paths(v)[0]
#     for dist in lengths:
#         if dist > 0 and dist < float('inf'):
#             distance_counts[dist] += 1

# distances = sorted(distance_counts.items())
# x, y = zip(*distances)
# plt.figure(figsize=(8, 6))
# plt.bar(x, y, color='mediumseagreen', edgecolor='black')
# plt.ylabel('Frequency (log scale)')
# plt.xlabel('Distance')
# plt.title('Distance Distribution in Largest WCC (huge)')
# plt.tight_layout()
# plt.savefig('distance_distribution_huge.png')
# plt.show()

# # Average shortest path length in largest WCC
# avg_distance = subgraph.average_path_length()
# print(f"Average shortest path length in largest WCC: {avg_distance}")