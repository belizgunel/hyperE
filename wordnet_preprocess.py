import nltk
from nltk.corpus import wordnet as wn
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


# Creating the edge lists for hypernym (noun and verb separate), member_holonyms, topic_domains 
# relationships in Wordnet for their largest connected components. 

def load_wordnet():
    SynstoIDs  = dict()
    IDstoSyns = dict()
    all_syns = list(wn.all_synsets())
    
    for idx, x in enumerate(all_syns):
        SynstoIDs[x] = idx
        IDstoSyns[idx] = x
#         ID_dict[idx] = x.name().split('.')[0]

    n = len(all_syns)
    e = make_edge_set()

    for idx, x in enumerate(all_syns):
        for y in x.topic_domains():
            y_idx = SynstoIDs[y]
            add_edge(e, idx  , y_idx)
            add_edge(e, y_idx,   idx)
            
    #Sparse matrix with all syns
    X = csr_matrix(e, shape=(n, n))

    return SynstoIDs, IDstoSyns, X, all_syns

    
SynstoIDs, IDstoSyns, X, all_syns = load_wordnet()
G = nx.from_scipy_sparse_matrix(X)
Gc = max(nx.connected_component_subgraphs(G), key=len)

# Get some stats 
connected_comps = sorted(nx.connected_components(G), key = len, reverse=True)
for comp in connected_comps:
    if len(comp)>100:
        print(len(comp))

print("There are "+str(len(connected_comps))+ " connected components.")
print("There are a total of "+str(G.number_of_nodes())+" nodes in the graph")
print("Largest component has "+str(Gc.number_of_nodes())+ " nodes")

# reorder with nx
Gc_final = nx.convert_node_labels_to_integers(Gc, ordering="decreasing degree", label_attribute="old_label")

#Create the dict for old-id <-> new-id matching for syns
RefDict = Gc_final.node
IDsToSyns_f = dict()
SynsToIDs_f = dict()
for new_idx in RefDict.keys():
    old_idx = RefDict[new_idx]['old_label']
    curr_syn = IDstoSyns[old_idx]
    IDsToSyns_f[new_idx] = curr_syn
    SynsToIDs_f[curr_syn] = new_idx
    

#Write the final edgelist.
nx.write_edgelist(Gc_final, "release_edges_cc/topic_domain/lcc.edges",data=False)

# Read all the emb files, save their tau and emb_dict.

emb_files = {
            'release_emb_cc/hypernym/noun_lcc.emb':'hypernyms_noun',
             'release_emb_cc/member_holonym/lcc.emb':'member_holonyms', 
             'release_emb_cc/topic_domain/lcc.emb':'topic_domain',
            }

RelEmbDict = defaultdict(dict)
RelTauDict = defaultdict(dict)
for file in emb_files.keys():
    with open(file, 'r') as emb:
        emb_lines = emb.readlines()    
    emb_lines = emb_lines[1:]
    emb_dict = dict()
    rel = emb_files[file]
    for idx, line in enumerate(emb_lines):
        curr_line = line.split(',')
        curr_tau = curr_line[-1].split("\n")[0]
        curr_tau = np.float64(curr_tau)
        curr_line = curr_line[:-1]
        curr_idx = int(curr_line[0])                
        emb_dict[curr_idx] = np.asarray(list(map(np.float64, curr_line[1:])))
    RelEmbDict[rel] = emb_dict
    RelTauDict[rel] = curr_tau


#Create the reference doc for evaluation that only includes the connected components for each relationship.

edge_files = {
             'release_edges_cc/topic_domain/lcc.edges':'topic_domain',
             'release_edges_cc/hypernym/noun_lcc.edges':'hypernyms_noun', 
             'release_edges_cc/member_holonym/lcc.edges':'member_holonyms',
            }

data = defaultdict()
TotalRelCount = 0
ReltoCount = {}

for file in edge_files.keys():
    rel = edge_files[file]
    IDstoSyns_curr = Ref_IDsToSyns_f[rel]
    with open(file, 'r') as edg:
        edg_lines = edg.readlines()  
    curr_count = 0
    for line in edg_lines:
        curr_line = line.split(" ")
        syn1 = IDstoSyns_curr[int(curr_line[0])]
        syn2 = IDstoSyns_curr[int(curr_line[1].split("\n")[0])]
        entity_tup = (syn1,syn2)
        data[entity_tup] = rel
        curr_count+=1
        TotalRelCount+=1
    print(str(rel)+" :"+str(curr_count)+" relationships")
    ReltoCount[rel]=curr_count
    
print("There are a total of "+str(TotalRelCount)+" relationship triplets.")


#Do hyperbolic KBC for 10-dimensional embeddings for each relationship.

import numpy as np
import hyp_functions as hyp

vector_dim = 10
ReltoW = defaultdict()
for rel in RelEmbDict.keys():
    emb_dict_curr = RelEmbDict[rel]
    vocab_size = len(emb_dict_curr)
    W_curr = np.zeros((vocab_size, vector_dim))
    SynsettoIDs_curr = Ref_SynsToIDs_f[rel]
    for idx, vec in emb_dict_curr.items():
        W_curr[idx,:] = vec
    ReltoW[rel] = W_curr


TruePosAll = 0
ReltoCorrectCount = {}
ReltoPosCount = {}
#Just initialize.
for rel in RelEmbDict.keys():
    ReltoCorrectCount[rel]=0
    ReltoPosCount[rel]=0
    

MultRel=0
AccMultRel=0
for tup, rel in data.items():
    e1 = tup[0]
    e2 = tup[1]
    ReltoDist = {}
    for r, W in ReltoW.items():
        SynsettoIDs_curr = Ref_SynsToIDs_f[r]
        emb_dict_curr = RelEmbDict[r]
        relTau = RelTauDict[r]
        relTau = np.float64(relTau)
        if (e1 in SynsettoIDs_curr) and (e2 in SynsettoIDs_curr):
            vec_e1 = W[SynsettoIDs_curr[e1],:]
            vec_e2 = W[SynsettoIDs_curr[e2],:]
            ReltoDist[r] = (hyp.dist(vec_e1,vec_e2))/relTau
    pred = min(ReltoDist, key=ReltoDist.get)
    ReltoPosCount[pred]+=1
    if len(ReltoDist)>1:
        MultRel+=1
        if pred==rel:
            AccMultRel+=1
    curr_dist = ReltoDist[pred]
    if (curr_dist>0.99) and (curr_dist<1.01):
        TruePosAll+=1
        ReltoCorrectCount[rel]+=1 


for rel in ReltoCorrectCount.keys():
    correct_count = ReltoCorrectCount[rel]
    total_count = ReltoCount[rel]
    pos_count = ReltoPosCount[rel]
    print(str(rel)+":")
    print("Precision: "+str(correct_count/pos_count))
    print("Recall: "+str(correct_count/total_count))
    print("\n")

        
print("Number of tuples involved in more than one relationship: " + str(MultRel))
print("Overall accuracy for that: "+str(AccMultRel/MultRel))


# Find the top 10 nearest neighbor to a particular synset for each rel.
import hyp_functions as hyp

vector_dim = 10
rel = 'topic_domain'
emb_dict_curr = RelEmbDict[rel]
vocab_size = len(emb_dict_curr)
W = np.zeros((vocab_size, vector_dim))
relTau = RelTauDict[rel]


e1 = wn.synset('geometry.n.01')
e1_idx = SynsToIDs_f[e1]


for idx, vec in emb_dict_curr.items():
    W[idx,:] = vec
    
vec_e1 = emb_dict_curr[e1_idx] 
curr_dist = []    
for row_idx in range(W.shape[0]):
    curr_vec = W[row_idx,:]
    normalized_dist = (hyp.dist(curr_vec,vec_e1))/relTau
    curr_dist.append(normalized_dist)


curr_dist[e1_idx] = np.Inf
curr_closest_indices = np.argsort(curr_dist)[:2]
for r_idx in curr_closest_indices:
    relev_syn = IDsToSyns_f[r_idx]
    print(curr_dist[r_idx], relev_syn.name(), relev_syn.definition())


# Word analogy for a example synsets for each rel.

import hyp_functions as hyp

vector_dim = 10
rel = 'member_holonyms'
emb_dict_curr = RelEmbDict[rel]
vocab_size = len(emb_dict_curr)
W = np.zeros((vocab_size, vector_dim))
relTau = RelTauDict[rel]


# Choose the entities.
e1 = wn.synset('african_elephant.n.01')
e1_idx = SynsToIDs_f[e1]

e2 = wn.synset('elephantidae.n.01')
e2_idx = SynsToIDs_f[e2]

e3 = wn.synset('dog.n.01')
e3_idx = SynsToIDs_f[e3]


for idx, vec in emb_dict_curr.items():
    W[idx,:] = vec
    
vec_e1 = emb_dict_curr[e1_idx]
vec_e2 = emb_dict_curr[e2_idx]
vec_e3 = emb_dict_curr[e3_idx]


vec1_ = hyp.hyp_scale(-1, vec_e1)
left_sum = hyp.hyp_weighted_sum(1, 1, vec_e2, vec1_)
print("Print distance between e1 and e2")
vec_search = hyp.hyp_weighted_sum(1, 1, left_sum, vec_e3)

curr_dist = []    
for row_idx in range(W.shape[0]):
    curr_vec = W[row_idx,:]
    normalized_dist = (hyp.dist(curr_vec, vec_search))/relTau
    curr_dist.append(normalized_dist)


curr_dist[e1_idx] = np.Inf
curr_dist[e2_idx] = np.Inf
curr_dist[e3_idx] = np.Inf

curr_closest_indices = np.argsort(curr_dist)[:10]
for r_idx in curr_closest_indices:
    relev_syn = IDsToSyns_f[r_idx]
    print(curr_dist[r_idx], relev_syn.name(), relev_syn.definition())

# Write word embeddings in the GloVe format for the largest cc for each relationship.
# Get the most frequent meaning for each word (if it's involved in multiple synsets)

rel = "topic_domain"
emb_dict_curr = RelEmbDict[rel]


WordtoVec = dict()
WordtoLemma = dict()
for idx in IDsToSyns_f.keys():
    syn = IDsToSyns_f[idx]
    vec = emb_dict_curr[idx]
    for curr_lemma in syn.lemmas():
        word = curr_lemma.name()
        if word not in WordtoVec.keys():
            WordtoVec[word] = vec
            WordtoLemma[word] = curr_lemma
        if (word in WordtoVec.keys()) and (curr_lemma.count()>WordtoLemma[word].count()):
            WordtoVec[word] = vec
            WordtoLemma[word] = curr_lemma

print("There were "+str(len(IDsToSyns_f))+" synsets")
print("There are "+str(len(WordtoVec))+" words now")


lines = []
for word in WordtoVec.keys():
    curr_line = str(word) + " " + " ".join(list(map(str,WordtoVec[word])))
    lines.append(curr_line)

with open('wordnet_word_emb/domain_topic.txt', 'w') as f:
    f.write('\n'.join(lines))    
    
