# First networkx library is imported 
# along with matplotlib
import re
import random
import pprint
import pydot
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

print(nx.__version__)

# G generation
# G = nx.random_tree(n=20, seed=1)
# G = nx.balanced_tree(2, 4)

random.seed(4)
G = nx.Graph()
G.add_edge(0,1, weight=4, species='A')
for i in range(2,20):
   from_node = random.randint(0,i-1)
   weight = random.randint(3,4)*3
   species = random.choice(['A','B'])
   G.add_edge(from_node, i, weight=weight, species=species)

# G = nx.Graph()
# G.add_edge(0, 1, weight=3,  species="B")
# G.add_edge(1, 2, weight=14, species="A")
# G.add_edge(2, 3, weight=3,  species="B")

#sub_graphs = set()
dict_sub_graphs = {}
def get_ordered_tree_branches_text(G, in_node, out_node):
    # find children
    edge_text = []
    edge_dict = []
    for edge in G.edges(in_node):
        #print(edge)
        if (edge[1] != out_node):
            text, weight = get_ordered_tree_branches_text(G, edge[1], edge[0])
            edge_text.append(text + str(weight))
            edge_dict.append([text, weight])

    #print(in_node, out_node, edge_text)
    if(len(edge_text) == 0):
        text = ""
    else:
        text = "(" + "".join(sorted(edge_text))  + ")"
    text += G[in_node][out_node]['species'] 
    if text in dict_sub_graphs:
        if dict_sub_graphs[text]['max_weight'] < G[in_node][out_node]['weight']:
            dict_sub_graphs[text]['max_weight'] = G[in_node][out_node]['weight']
    else:
        dict_sub_graphs[text] = {'max_weight': G[in_node][out_node]['weight'], "dependencies": edge_dict}
    return text, G[in_node][out_node]['weight']

for u,v,a in G.edges(data=True):
    get_ordered_tree_branches_text(G, u, v)
    get_ordered_tree_branches_text(G, v, u)

sorted_dict_sub_graphs = sorted(dict_sub_graphs.items(), key = lambda kv: kv[0], reverse=True)
pprint.pprint(sorted_dict_sub_graphs)

# for u,v,a in G.edges(data=True):
#     print(u, end=",")

# for u,v,a in G.edges(data=True):
#     print(v)

# for u,v,a in G.edges(data=True):
#     print(a["species"])

# for u,v,a in G.edges(data=True):
#     print(a["weight"]/10)

# visualize
color_map = []
for node in G:
    if len(G.edges(node)) == 1:
        color_map.append('yellow')
    else: 
        color_map.append('gray')  

plt.figure(figsize=(8,8))
pos = graphviz_layout(G, prog="twopi")
labels = nx.get_edge_attributes(G,'weight')
edges = G.edges()
dict_color= {"A":"red", "B":"blue"}
colors = [dict_color[G[u][v]['species']] for u,v in edges]
nx.draw(G, pos, node_color=color_map, edge_color=colors, width=4, with_labels=True)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()