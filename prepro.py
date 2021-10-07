import ParallelSampler as cpp_para_sampler

import numpy as np
from numpy.core.fromnumeric import shape
from scipy.sparse import csr_matrix
import scipy as spy
import os
import sys

# config={'k': 1, 'threshold': 0.01, 'epsilon': '1e-5', 'add_self_edge': True, 'method': 'ppr', 'size_root': 1, 
#         'fix_target': True, 'sequential_traversal': True, 'type_': 0, 'name_data': 'arxiv', 
#         'dir_data': {'local': './data', 'is_adj_changed': False, 'is_feat_changed': False}, 'is_transductive': True}
# # config_subgraph_size=3


GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'
NODE_LABEL_NUMBER_SUFFIX='node_label_number'


def find_subgraph(adj_dense,subgraph_size):
    N=adj_dense.shape[0]
    assert N==adj_dense.shape[1]
    adj_csr=csr_matrix(adj_dense)
    node_targets=[i for i in range(N)]

    args_Cpp_sampler = [node_targets, N, 1, True, True, 1]
    args_Cpp_sampler = [adj_csr.indptr, adj_csr.indices, adj_csr.data] + args_Cpp_sampler + ["", "", ""]
    para_sampler = cpp_para_sampler.ParallelSampler(*args_Cpp_sampler)
    # subgraph_size=config_subgraph_size if config_subgraph_size<N else N
    para_sampler.preproc_ppr_approximate(subgraph_size, 0.85, 1e-5, '', '')
    cpp_config_list = [{'method': 'ppr', 'k': str(subgraph_size), 'num_roots': '1', 
                'threshold': '0.01', 'add_self_edge': 'true', 'return_target_only': 'false'}]
    cpp_aug_list = [{'hops'}]

    ret = para_sampler.parallel_sampler_ensemble(cpp_config_list, cpp_aug_list)
    res=np.array(ret[0].get_subgraph_node())
    return res


def read_data_txt(dataset):
    data = dict()
    dirpath = '/new_disk_B/scy/{}'.format(dataset)
    print('reading data...')
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = os.path.join(dirpath, f)
        suffix = f.replace(dataset, '')        
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath, dtype=int, delimiter=',')
    
    label_dict={}
    for node_idx in range(len(data[NODE_LABELS_SUFFIX])):
        ori_label=data[NODE_LABELS_SUFFIX][node_idx]
        if not ori_label in label_dict:
            mapped = len(label_dict)
            label_dict[ori_label] = mapped
        data[NODE_LABELS_SUFFIX][node_idx]=label_dict[ori_label]
    node_label_num=len(label_dict)
    print('node labels number: ', node_label_num)
    data[NODE_LABEL_NUMBER_SUFFIX]=node_label_num
    return data

def resolve_data(data):
    graph_ids = set(data['_graph_indicator.txt'])
    adj = data[ADJACENCY_SUFFIX]
    edge_index = 0
    node_index_begin = 0
    res=[]
    for g_id in set(graph_ids):
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()

        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int)
        while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
            temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
            edge_index += 1
        temp_node_labels = data[NODE_LABELS_SUFFIX][node_index_begin:node_index_begin + temp_nodN]
        node_index_begin += temp_nodN
        # temp_node_labels=np.eye(data[NODE_LABEL_NUMBER_SUFFIX])[temp_node_labels]
        res.append((temp_A,temp_node_labels))
        # print('temp_A:',temp_A.shape)
        # print('temp_node_labels:',temp_node_labels)
    return res 

def hyper_neighbor(subgraphs):
    N=subgraphs.shape[0]
    hyper_A=np.zeros((N,N),int)
    subgraphs_set_list=[set(i) for i in subgraphs] # remove -1
    # print(subgraphs_set_list)
    for j in range(N):
        for k in range(N):
            if subgraphs_set_list[j]&subgraphs_set_list[k]!=set([]):
                hyper_A[j][k]=hyper_A[k][j]=1
    hyper_A[np.diag_indices_from(hyper_A)] = 0

    return hyper_A

def write_subgraph(dataset,ori_data,node_label_number,subgraph_size):
    dir_name='./data/{}_{}'.format(dataset,subgraph_size)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    print('writting subgraph data into ',dir_name)
    N_graph=len(ori_data)

    sub_2_A_file=os.path.join(dir_name,'sub2A.txt')
    hyper_A_file=os.path.join(dir_name,'hyper_A.txt')
    sub_att_file=os.path.join(dir_name,'sub_att.txt')
    f_sub2A=open(sub_2_A_file,'w+')
    f_hyper=open(hyper_A_file,'w+')
    f_att=open(sub_att_file,'w+')
    f_sub2A.write(str(N_graph)+'\n')
    f_hyper.write(str(N_graph)+'\n')
    f_att.write(str(N_graph)+'\n')
    for i in range(N_graph):
        subgraphs_rep=[]
        subgraphs=find_subgraph(ori_data[i][0],subgraph_size)
        N_node=subgraphs.shape[0]
        sub_2_A=np.zeros((N_node,N_node),int)
        f_sub2A.write(str(N_node)+'\n')
        f_hyper.write(str(N_node)+'\n')
        f_att.write(str(N_node)+'\n')
        if subgraphs.shape==(N_node,):
            for j in range(N_node):
                subgraph_graph_i_node_j=np.array(subgraphs[j])
                sub_2_A[j][subgraph_graph_i_node_j]=1
                subgraph_i_j_rep=np.array(ori_data[i][1][subgraph_graph_i_node_j])
                # print(subgraph_graph_i_node_j.shape)
                # sys.exit()
                # subgraph_i_j_rep=np.eye(node_label_number)[subgraph_i_j_rep].reshape(1,-1)
                # app=np.zeros((1,(subgraph_size-len(subgraph_graph_i_node_j))*node_label_number))
                # subgraph_i_j_rep=np.hstack((subgraph_i_j_rep,app))
                subgraphs_rep.append(subgraph_i_j_rep)
            subgraphs_rep=np.array(subgraphs_rep).reshape(N_node,node_label_number)
        else:
            subgraph_node_num=subgraphs.shape[1]
            for k in range(N_node):               
                sub_2_A[k][subgraphs[k]]=1
            subgraphs_rep=np.array(ori_data[i][1][subgraphs])
            subgraphs_rep=np.eye(node_label_number)[subgraphs_rep]
            subgraphs_rep=np.sum(subgraphs_rep,1)
            # print('subgraphs_rep: ',subgraphs_rep.shape)
            # print(subgraphs_rep)
            # print('summation:',np.sum(subgraphs_rep,1))
            # sys.exit()
            # subgraphs_rep=np.eye(node_label_number)[subgraphs_rep].reshape(N_node,-1)
            # if subgraph_node_num<subgraph_size:
            #     app=np.zeros((N_node,(subgraph_size-subgraph_node_num)*node_label_number))
            #     subgraphs_rep=np.hstack((subgraphs_rep,app))
        
        for line in sub_2_A:
            for item in line:
                f_sub2A.write(str(item)+'\t')
            f_sub2A.write('\n')

        hyper_A=hyper_neighbor(subgraphs=subgraphs)
        for line in hyper_A:
            for item in line:
                f_hyper.write(str(item)+'\t')
            f_hyper.write('\n')
        for line in subgraphs_rep:
            for item in line:
                f_att.write(str(item)+'\t')
            f_att.write('\n')

    f_hyper.close()
    f_sub2A.close()
    f_att.close()
    
    print('writting success in ',dir_name)
    return 

def read_subgraph_file(dataset,config_size,suffix):
    res=[]
    file_name='./data/{}_{}/{}.txt'.format(dataset,config_size,suffix)
    if not os.path.exists(file_name):
        print('no file:',file_name) 
        data=read_data_txt(dataset)
        write_subgraph(dataset,resolve_data(data),data[NODE_LABEL_NUMBER_SUFFIX],config_size)
    f=open(file_name,'r')
    graph_num=f.readline().strip('\n')
    # print('graph_num:',graph_num)
    for graph_idx in range(int(graph_num)):
        # print(graph_idx)
        res_graph=[]
        node_num=f.readline().strip('\n')
        # print('node_num:',node_num)
        for node_idx in range(int(node_num)):
            line=f.readline().strip('\n\t').split('\t')
            if '.' in line[0]:
                line=np.array([float(i) for i in line])
            else:
                line=np.array([int(i) for i in line])
            res_graph.append(line)
        res.append(np.array(res_graph))
        #print('res_graph:',np.array(res_graph).shape)
    f.close()
    return res

# read_subgraph_file('MUTAG',3,'sub2A')
