import nltk
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import floyd_warshall, connected_components
import operator
from collections import defaultdict
import numpy as np
import networkx as nx
import json
from collections import defaultdict


# Some definitions

def acosh(x):
    return np.log(x + np.sqrt(x**2-1))

# Hyperbolic distance
def dist(u,v):
    z  = 2 * np.linalg.norm(u-v)**2
    uu = 1. + z/((1-np.linalg.norm(u)**2)*(1-np.linalg.norm(v)**2))
    return acosh(uu)

# Hyperbolic distance from 0
def hyp_dist_origin(x):
    return np.log((1+np.linalg.norm(x))/(1-np.linalg.norm(x)))

def make_edge_set(): return ([],([],[]))
def add_edge(e, i,j):
    (v,(row,col)) = e
    row.append(i)
    col.append(j)
    v.append(1)


files = {
            'wikidata_data/student_of.tsv':'student_of',
            'wikidata_data/member_of.tsv':'member_of',
            'wikidata_data/part_of.tsv':'part_of',
        }

Ref_QtoLabel_f = defaultdict(dict)
Ref_QtoIDs_f = defaultdict(dict)
Ref_IDtoQs_f = defaultdict(dict)


for file,rel in files.items():
    with open(file, "r") as data:
        data_lines = data.readlines()

    data_lines = data_lines[1:]
    QtoLabel = dict()
    QtoIDs = defaultdict()
    IDtoQs = dict()
    e = make_edge_set()

    counter = 0
    triple_count = 0
    for line in data_lines:
        curr_line = line.split("\t")
        item = (curr_line[0].split("/"))[-1]
#         itemLabel = (curr_line[1].split("/"))[-1]
        influenced_by = (curr_line[1].split("/"))[-1]
        influenced_byLabel = (curr_line[2].split("/"))[-1].split("\n")[0]
        if influenced_by not in QtoLabel.keys():
            QtoLabel[influenced_by] = influenced_byLabel
        if item not in QtoIDs.keys():
            QtoIDs[item] = counter
            IDtoQs[counter] = item
            counter+=1
        if influenced_by not in QtoIDs.keys():
            QtoIDs[influenced_by] = counter
            IDtoQs[counter] = influenced_by
        add_edge(e,QtoIDs[item], QtoIDs[influenced_by])
        add_edge(e,QtoIDs[influenced_by], QtoIDs[item])
        triple_count+=1

    print("There are a total of "+str(triple_count)+" triples for the relationship "+str(rel)+".")

    # Take the largest lcc for the relationship.

    n = len(QtoIDs)
    X = csr_matrix(e, shape=(n, n))
    G = nx.from_scipy_sparse_matrix(X)
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    print("Total number of unique entities: "+str(G.number_of_nodes()))
    print("Total number of nodes in lcc: "+str(Gc.number_of_nodes()))
    Gc_final = nx.convert_node_labels_to_integers(Gc, ordering="decreasing degree", label_attribute="old_label")


    #Create the dict for old-id <-> new-id matching for syns
    RefDict = Gc_final.node
    IDtoQs_f = dict()
    QtoIDs_f = dict()
    for new_idx in RefDict.keys():
        old_idx = RefDict[new_idx]['old_label']
        curr_Q = IDtoQs[old_idx]
        IDtoQs_f[new_idx] = curr_Q
        QtoIDs_f[curr_Q] = new_idx

    
    #Write the final edgelist.
    nx.write_edgelist(Gc_final, "wikidata_edges/"+str(rel)+"/lcc.edges",data=False)
        
    #Take the labels only in the lcc.    
    keys_a = set(QtoLabel.keys())
    keys_b = set(QtoIDs_f.keys())
    intersection = keys_a & keys_b
    QtoLabel_f = dict()
    for item in intersection:
        QtoLabel_f[item] = QtoLabel[item]
    
    Ref_QtoLabel_f[rel] = QtoLabel_f
    Ref_QtoIDs_f[rel] = QtoIDs_f
    Ref_IDtoQs_f[rel] = IDtoQs_f

