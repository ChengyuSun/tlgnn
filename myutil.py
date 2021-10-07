import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from prepro import GRAPH_LABELS_SUFFIX, read_data_txt,resolve_data,read_subgraph_file
import sys
from numpy.random import permutation
from numpy import split

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        if node_features is None:
            self.node_features=None
        else:
            self.node_features = torch.LongTensor(node_features)
        self.edge_mat = 0
        self.max_neighbor = 0


def load_data(dataset,subgraph_size):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}


    data=read_data_txt(dataset)
    graph_infos=resolve_data(data)
    graph_num=len(graph_infos)
    for graph_idx in range(graph_num):
        g = nx.Graph()
        graph_label=data[GRAPH_LABELS_SUFFIX][graph_idx]
        if not graph_label in label_dict:
                mapped = len(label_dict)
                label_dict[graph_label] = mapped
        temp_adj=graph_infos[graph_idx][0]
        temp_node_labels=graph_infos[graph_idx][1]
        node_num=len(temp_node_labels)
        node_tags = []
        for node_idx in range(node_num):
            g.add_node(node_idx)
            node_label=temp_node_labels[node_idx]
            if not node_label in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[node_label] = mapped
            node_tags.append(feat_dict[node_label])
            targets=np.where(temp_adj[node_idx]==1)[0]
            [g.add_edge(node_idx,t) for t in targets]

        assert len(g) == node_num
        g_list.append(S2VGraph(g, graph_label, node_tags))

        
    sub2A=read_subgraph_file(dataset,subgraph_size,'sub2A')

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    

    print("# data: %d" % len(g_list))

    return g_list,len(label_dict),sub2A,data[GRAPH_LABELS_SUFFIX]


def load_hyper_data(dataset,subgraph_size,graph_labels):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading hyper data')
    g_list = []
    label_dict = {}

    hyper_A=read_subgraph_file(dataset,subgraph_size,'hyper_A')
    sub_att=read_subgraph_file(dataset,subgraph_size,'sub_att')
    graph_num=len(graph_labels)

    for graph_idx in range(graph_num):
        g = nx.Graph()
        graph_label=graph_labels[graph_idx]
        if not graph_label in label_dict:
                mapped = len(label_dict)
                label_dict[graph_label] = mapped
        temp_adj=hyper_A[graph_idx]
        node_num=len(temp_adj)
        
        # print('temp_adj:',temp_adj,node_num)
        node_features = sub_att[graph_idx]
        for node_idx in range(node_num):
            g.add_node(node_idx)
            targets=np.where(temp_adj[node_idx]==1)[0]
            # print('targets:',targets)
            [g.add_edge(node_idx,t) for t in targets]

        g_list.append(S2VGraph(g, graph_label, None,node_features))


    #add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        if len(g.g)==0 or len(g.g)==1:
            degree_list.append(0)
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        if len(g.g)==0 or len(g.g)==1:
            g.edge_mat=[]
        else:
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])
            # print('edge:',edges)
            # print(len(edges))
            # deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)
            # print(g.edge_mat.shape)
            # sys.exit()


    print('# classes: %d' % len(label_dict))

    print("# data: %d" % len(g_list))

    return g_list


def separate_data(graph_list,hyper_graph_list,motif2A_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    random.shuffle(train_idx)
    val_len=len(train_idx)//9
    val_idx=train_idx[:val_len]
    train_idx=train_idx[val_len:]
    
    train_graph_list = [graph_list[i] for i in train_idx]
    train_hyper_graph_list=[hyper_graph_list[i] for i in train_idx]
    train_motif2A_list=[motif2A_list[i] for i in train_idx]

    val_graph_list = [graph_list[i] for i in val_idx]
    val_hyper_graph_list=[hyper_graph_list[i] for i in val_idx]
    val_motif2A_list=[motif2A_list[i] for i in val_idx]
    
    test_graph_list = [graph_list[i] for i in test_idx]
    test_hyper_graph_list = [hyper_graph_list[i] for i in test_idx]
    test_motif2A_list=[motif2A_list[i] for i in test_idx]

    return train_graph_list, train_hyper_graph_list,train_motif2A_list,val_graph_list,val_hyper_graph_list,val_motif2A_list,test_graph_list,test_hyper_graph_list,test_motif2A_list

def separate_data_no_val(graph_list,hyper_graph_list,motif2A_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    # with open('./seperate.txt','a') as f:
    #     f.write('train: '+str(train_idx)+'\n'+'test: '+str(test_idx)+'\n')
    train_graph_list = [graph_list[i] for i in train_idx]
    train_hyper_graph_list=[hyper_graph_list[i] for i in train_idx]
    train_motif2A_list=[motif2A_list[i] for i in train_idx]

    test_graph_list = [graph_list[i] for i in test_idx]
    test_hyper_graph_list = [hyper_graph_list[i] for i in test_idx]
    test_motif2A_list=[motif2A_list[i] for i in test_idx]

    return train_graph_list, train_hyper_graph_list,train_motif2A_list,test_graph_list,test_hyper_graph_list,test_motif2A_list


def graphlet_diffuse_no_label(start_index,adj_original):

    node_rep=[0 for i in range(8)]

    neighbors_1=np.nonzero(adj_original[start_index])[0].tolist()
    for index_1_1,n1_1 in enumerate(neighbors_1):
        #2
        node_rep[0]+=1
        neighbors_2=np.nonzero(adj_original[n1_1])[0].tolist()
        if start_index in neighbors_2:
            neighbors_2.remove(start_index)
        for index_2_1,n2_3 in enumerate(neighbors_2):

            if adj_original[start_index][n2_3]==0:
                #3_1
                node_rep[1] += 1
                #4_2
                node_rep[5] +=len(neighbors_2[index_2_1+1:])

            neighbors_3=np.nonzero(adj_original[n2_3])[0].tolist()
            if start_index in neighbors_3:
                neighbors_3.remove(start_index)
            if n1_1 in neighbors_3:
                neighbors_3.remove(n1_1)
             # 4_1
            node_rep[4] += len(neighbors_3)



        for index_1_2,n1_2 in enumerate(neighbors_1[index_1_1+1:]):
            #3_3
            if adj_original[n1_1][n1_2]==1:
                node_rep[3]+=1
            else :
                #3_2
                node_rep[2] += 1
                #4_3
                for n1_3 in neighbors_1[index_1_2+index_1_1 + 2:]:
                    if adj_original[n1_1][n1_3]==0 and adj_original[n1_2][n1_3]==0:
                        node_rep[6] += 1
                #4_4
                for n2_1 in np.nonzero(adj_original[n1_1])[0]:
                    if n2_1!=start_index and n2_1!=n1_2 and \
                            adj_original[n1_2][n2_1]==0 and adj_original[start_index][n2_1]==0 :
                        node_rep[7] += 1
                for n2_2 in np.nonzero(adj_original[n1_2])[0]:
                    if n2_2 != start_index and n2_2 != n1_1 and\
                            adj_original[n1_1][n2_2]==0 and adj_original[start_index][n2_2]==0 :
                        node_rep[7] += 1
    return np.array(node_rep)