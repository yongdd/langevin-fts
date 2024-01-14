import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from networkx.drawing.nx_agraph import graphviz_layout

def compute_energy(angles, nodes, weights, bfs_layers, successors, center_0):
    
    pos = {center_0: (1.0, 1.0)}

    idx = 0
    for layer in bfs_layers:
        for node in bfs_layers[layer]:
            if node not in successors:
                continue
            
            for count, child_node in enumerate(successors[node]):
                if node > child_node:
                    weight = weights[(child_node, node)]
                else:
                    weight = weights[(node, child_node)]
                
                x = pos[node][0] + weight*np.cos(angles[idx])
                y = pos[node][1] + weight*np.sin(angles[idx])
                pos[child_node] = (x, y)
                idx += 1

    energy = 0.0
    for i in nodes:
        for j in nodes:
            if i<j:
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                energy += 1/(dx**2 + dy**2)

    return energy, pos
                
def polymer_layout(G, k=1, num=1):
    
    # Find a trace from the root to each node
    nodes = G.nodes
    weights = nx.get_edge_attributes(G, 'weight')
    
    # print(weights)
    center_0 = list(nx.center(G))[0]
    bfs_layers = (dict(enumerate(nx.bfs_layers(G, [center_0]))))
    
    pos = {center_0: (1.0, 1.0)}
    angles_low = {center_0: 0.0}
    angles_high = {center_0: 2*np.pi}
    successors = dict(nx.bfs_successors(G, source=center_0))

    angles = {}
    for layer in bfs_layers:
        for node in bfs_layers[layer]:
            if node not in successors:
                continue
            # print(node, successors[node])
            for count, child_node in enumerate(successors[node]):
                if node > child_node:
                    weight = weights[(child_node, node)]
                else:
                    weight = weights[(node, child_node)]
                
                d_angle = (angles_high[node]-angles_low[node])/len(successors[node])
                angles_low[child_node] = angles_low[node] + count*d_angle
                angles_high[child_node] = angles_low[node] + (count+1.0)*d_angle
                angle = (angles_low[child_node] + angles_high[child_node])/2
                                                         
                x = pos[node][0] + weight*np.cos(angle)
                y = pos[node][1] + weight*np.sin(angle)
                pos[child_node] = (x, y)
                angles[child_node] = angle
    
    angle_list = list(angles.values())
    # print(angle_list)
    energy_func = lambda angle_list : compute_energy(angle_list, nodes, weights, bfs_layers, successors, center_0)[0]
    # print(energy_func(angle_list))
    
    res = minimize(energy_func, angle_list, method='BFGS', tol=1e-6)
    pos = compute_energy(res.x, nodes, weights, bfs_layers, successors, center_0)[1]
    print(res)
    
    return pos

# print(nx.__version__)

# G generation
# G = nx.random_tree(n=20, seed=1)
# G = nx.balanced_tree(2, 4)

random.seed(2)
G = nx.Graph()
weight_0 = round(random.randint(6,10)*0.1, 3)
G.add_edge(0, 1, weight=weight_0, monomer_type=random.choice(['A','B','C','D','E','F']))
for i in range(2,50):
   from_node = random.randint(0,i-1)
   weight = round(random.randint(6,10)*0.1, 3)
   monomer_type = random.choice(['A','B','C','D','E','F'])
   G.add_edge(from_node, i, weight=weight, monomer_type=monomer_type)

# visualize
color_map = []
for node in G:
    if len(G.edges(node)) == 1:
        color_map.append('yellow')
    else: 
        color_map.append('gray')  

pos = graphviz_layout(G, prog='twopi')
# pos = polymer_layout(G, k=1, num=1)
labels = nx.get_edge_attributes(G, 'weight')
edges = G.edges()
dict_color= {"A":"r", "B":"b", "C":"g", "D":"c", "E":"m", "F":"y"}
# print(labels)
colors = [dict_color[G[u][v]['monomer_type']] for u,v in edges]

plt.figure(figsize=(15,15))
nx.draw(G, pos, node_color=color_map, edge_color=colors, width=4, with_labels=True, node_size=80, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, rotate=False, font_size=8)
# plt.show()
plt.savefig("plot.png")