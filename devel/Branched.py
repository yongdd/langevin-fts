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

random.seed(2)
G = nx.Graph()
G.add_edge(0,1, weight=random.randint(3,4)*3, monomer_type=random.choice(['A','B','C']))
for i in range(2,30):
   from_node = random.randint(0,i-1)
   weight = random.randint(3,4)*3
   monomer_type = random.choice(['A','B','C'])
   G.add_edge(from_node, i, weight=weight, monomer_type=monomer_type)

# G = nx.Graph()
# for i in range(0,8):
#     G.add_edge(i,i+1, weight=12, monomer_type="C")

# k=9
# for i in range(0,4):
#     for j in range(0,5):
#         G.add_edge(i,k, weight=9, monomer_type="A")
#         k += 1
# for i in range(4,8):
#     for j in range(0,5):
#         G.add_edge(i,k, weight=9, monomer_type="B")
#         k += 1

# G = nx.Graph()
# G.add_edge(0, 1, weight=3,  monomer_type="B")
# G.add_edge(1, 2, weight=14, monomer_type="A")
# G.add_edge(2, 3, weight=3,  monomer_type="B")

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
    text += G[in_node][out_node]['monomer_type'] 
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


for u,v,a in G.edges(data=True):
    if (a["monomer_type"] == "A"):
        monomer_id = 0
    elif(a["monomer_type"] == "B"):
        monomer_id = 1
    elif(a["monomer_type"] == "C"):
        monomer_id = 2
    print("\t\t%d\t%s\t%d\t%d" % (monomer_id, a["weight"]/10, u, v))


# for u,v,a in G.edges(data=True):
#     print(u, end=",")
# print("")

# for u,v,a in G.edges(data=True):
#     print(v, end=",")
# print("")

# for u,v,a in G.edges(data=True):
#     print("\"" + a["monomer_type"] + "\"", end=",")
# print("")

# for u,v,a in G.edges(data=True):
#     print(a["weight"]/10, end=",")
# print("")

w_a = []
w_b = []
w_c = []

for i in range(5*4*3):
    temp = random.uniform(-1,1)
    w_a.append(temp)
    # print( "%17.10e" % (temp), end=",")
    # if (i % 3 == 2):
    #     print("")
# print("")
for i in range(5*4*3):
    temp = random.uniform(-1,1)
    w_b.append(temp)
    # print( "%17.10e" % (temp), end=",")
    # if (i % 3 == 2):
    #     print("")
# print("")
for i in range(5*4*3):
    temp = random.uniform(-1,1)
    w_c.append(temp)
    # print( "%17.10e" % (temp), end=",")
    # if (i % 3 == 2):
    #     print("")
# print("")

for k in range(3):
    for j in range(4):
        for i in range(5):
            print( "%17.10e %17.10e %17.10e" % (w_a[i*4*3 + j*3 + k], w_b[i*4*3 + j*3 + k], w_c[i*4*3 + j*3 + k]))

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
dict_color= {"A":"red", "B":"blue", "C":"green"}
colors = [dict_color[G[u][v]['monomer_type']] for u,v in edges]
nx.draw(G, pos, node_color=color_map, edge_color=colors, width=4, with_labels=True)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()