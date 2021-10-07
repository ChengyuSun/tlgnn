import numpy as np

# General
import os
import numpy as np
from pathlib import Path
import typing
import time
import json
import copy
from typing import Dict, List
import multiprocessing
from multiprocessing import Pool
from itertools import accumulate 
from collections import OrderedDict
import pickle
import sys
from functools import partial


#Sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

# Pytorch lightning
import pytorch_lightning as pl

# Pytorch Geometric
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import MessagePassing, GINConv

# Similarity calculations
from fastdtw import fastdtw

# Networkx
import networkx as nx


# Our Methods
sys.path.insert(0, '..') # add config to path
import config
import subgraph_utils
from subgraph_mpn import SG_MPN
from datasets import SubgraphDataset
import anchor_patch_samplers
from anchor_patch_samplers import *
import gamma
import attention



sys.path.append("models/")
from mlp import MLP


class TLGNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, device, multi_head=0):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(TLGNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.hyper_mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        self.hyper_batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.hyper_mlps.append(MLP(num_mlp_layers, 2, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.hyper_mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.hyper_batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        # self.att_a=nn.Parameter(torch.rand(1))

        if multi_head:
            self.atts = nn.Parameter(torch.Tensor(np.random.rand(self.num_layers, 2)))
        else:
            self.atts=nn.Parameter(torch.Tensor(np.random.rand(2,)))

        self.multi_head=multi_head



    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / len(graph.g)] * len(graph.g))

            else:
                ###sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def hyper_next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.hyper_mlps[layer](pooled)
        h = self.hyper_batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def hyper_next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.hyper_mlps[layer](pooled)

        h = self.hyper_batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    # def attention_layer(self,h1,h2):
    #     a=nn.Parameter(torch.randn(1))
    #     b=nn.Parameter(torch.randn(1))
    #     sum=a+b
    #     return h1*a/sum+h2*a/sum

    def initialize_cc_ids(subgraph_list,networkx_graph):
        '''
        Initialize the 3D matrix of (n_subgraphs X max number of cc X max length of cc)

        Input:
            - subgraph_list: list of subgraphs where each subgraph is a list of node ids 

        Output:
            - reshaped_cc_ids_pad: padded tensor of shape (n_subgraphs, max_n_cc, max_len_cc)
        '''
        n_subgraphs = len(subgraph_list) # number of subgraphs

        # Create connected component ID list from subgraphs
        cc_id_list = []
        for curr_subgraph_ids in subgraph_list:
            subgraph = nx.subgraph(networkx_graph, curr_subgraph_ids) #networkx version of subgraph
            con_components = list(nx.connected_components(subgraph)) # get connected components in subgraph
            cc_id_list.append([torch.LongTensor(list(cc_ids)) for cc_ids in con_components])

        # pad number of connected components
        max_n_cc = max([len(cc) for cc in cc_id_list]) #max number of cc across all subgraphs
        for cc_list in cc_id_list:
            while True:
                if len(cc_list) == max_n_cc: break
                cc_list.append(torch.LongTensor([config.PAD_VALUE]))

        # pad number of nodes in connected components
        all_pad_cc_ids = [cc for cc_list in cc_id_list for cc in cc_list]
        assert len(all_pad_cc_ids) % max_n_cc == 0
        con_component_ids_pad = pad_sequence(all_pad_cc_ids, batch_first=True, padding_value=config.PAD_VALUE) # (batch_sz * max_n_cc, max_cc_len)
        reshaped_cc_ids_pad = con_component_ids_pad.view(n_subgraphs, max_n_cc, -1) # (batch_sz, max_n_cc, max_cc_len)

        return reshaped_cc_ids_pad # (n_subgraphs, max_n_cc, max_len_cc)


    def init_all_embeddings(self,cc_ids,node_embeddings):
        '''
        Initialize the CC and channel-specific CC embeddings for the subgraphs in the specified split
        ('all', 'train_val', 'train', 'val', or 'test')
        '''
        # initialize CC embeddings
        cc_embeddings = torch.sum(node_embeddings(cc_ids.to(self.device)), dim=2)
        
        self.N_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
        self.N_B_cc_embeds = Parameter(cc_embeddings.detach().clone())
        self.S_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
        self.S_B_cc_embeds = Parameter(cc_embeddings.detach().clone())
        self.P_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
        self.P_B_cc_embeds = Parameter(cc_embeddings.detach().clone())

        return


    def initialize_border_sets(self,fname, cc_ids, radius, networkx_graph,ego_graph_dict=None ):
        '''
        Creates and saves to file a matrix containing the node ids in the k-hop border set of each CC for each subgraph
        The shape of the resulting matrix, which is padded to the max border set size, is (n_subgraphs, max_n_cc, max_border_set_sz)
        '''
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        all_border_sets = []

        # for each component in each subgraph, calculate the k-hop node border of the connected component
        for s, subgraph in enumerate(cc_ids):
            border_sets = []
            for c, component in enumerate(subgraph):
                # radius specifies the size of the border set - i.e. the k number of hops away the node can be from any node in the component to be in the border set 
                component_border = subgraph_utils.get_component_border_neighborhood_set(networkx_graph, component, radius, ego_graph_dict)
                border_sets.append(component_border)
            all_border_sets.append(border_sets)

        #fill in matrix with padding
        max_border_set_len = max([len(s) for l in all_border_sets for s in l])
        border_set_matrix = torch.zeros((n_subgraphs, max_n_cc, max_border_set_len), dtype=torch.long).fill_(config.PAD_VALUE)
        for s, subgraph in enumerate(all_border_sets):
            for c,component in enumerate(subgraph):
                fill_len = max_border_set_len - len(component)
                border_set_matrix[s,c,:] = torch.cat([torch.LongTensor(list(component)),torch.LongTensor((fill_len)).fill_(config.PAD_VALUE)])
        
        # save border set to file 
        np.save(fname, border_set_matrix.cpu().numpy())
        return border_set_matrix # n_subgraphs, max_n_cc, max_border_set_sz


    def get_border_sets(self,dataset_name,graph_id,cc_ids,networkx_graph):
        '''
            Returns the node ids in the k-hop border of each subgraph (where k = neigh_sample_border_size) for the train, val, and test subgraphs
        '''

        # location where similarities are stored
        sim_path = './dataset/{}/{}/similarities/'.format(dataset_name,graph_id)

        # We need the border sets if we're using the neighborhood channel or if we're using the edit distance similarity function in the structure channel
        
        # load ego graphs dictionary
        ego_graph_path = './dataset/{}/{}/edo_graphs.txt'.format(dataset_name,graph_id)

        if ego_graph_path.exists():
            with open(str(ego_graph_path), 'r') as f:
                ego_graph_dict = json.load(f)
            ego_graph_dict = {int(key): value for key, value in ego_graph_dict.items()}
        else: ego_graph_dict = None

        # either load in the border sets from file or recompute the border sets
        neigh_path = sim_path / (str(self.hparams["neigh_sample_border_size"]) + '_' + str(config.PAD_VALUE) + '_train_border_set.npy')

        self.train_N_border = self.initialize_border_sets(neigh_path, cc_ids,  self.hparams["neigh_sample_border_size"], networkx_graph,ego_graph_dict)
    
           


    def prepare_data(self,dataset_name,subgraph_list,networkx_graph,graph_id):
            '''
            Initialize connected components, precomputed similarity calculations, and anchor patches
            '''
            print('--- Started Preparing Data ---', flush=True)

            # Intialize connected component matrix (n_subgraphs, max_n_cc, max_len_cc)
            cc_ids = self.initialize_cc_ids(subgraph_list)

            # initialize trainable embeddings for each cc
            print('--- Initializing CC Embeddings ---', flush=True)
            self.init_all_embeddings(cc_ids)

            # Initialize border sets for each cc
            print('--- Initializing CC Border Sets ---', flush=True)
            self.get_border_sets(dataset_name,graph_id,cc_ids,networkx_graph)

            # calculate similarities 
            print('--- Getting Similarities ---', flush=True)
            self.get_similarities(split='train_val')

            # Initialize neighborhood, position, and structure anchor patches
            print('--- Initializing Anchor Patches ---', flush=True)
            if self.hparams['use_neighborhood']: 
                self.anchors_neigh_int, self.anchors_neigh_border = init_anchors_neighborhood('train_val', \
                    self.hparams, self.networkx_graph, self.device, self.train_cc_ids, self.val_cc_ids, \
                        None, self.train_N_border, self.val_N_border, None) # we pass in None for the test_N_border
            else: self.anchors_neigh_int, self.anchors_neigh_border = None, None
            if self.hparams['use_position']: 
                self.anchors_pos_int = init_anchors_pos_int('train_val', self.hparams, self.networkx_graph, self.device, self.train_sub_G, self.val_sub_G, self.test_sub_G) 
                self.anchors_pos_ext = init_anchors_pos_ext(self.hparams, self.networkx_graph, self.device)
            else: self.anchors_pos_int, self.anchors_pos_ext = None, None
            if self.hparams['use_structure']:
                # pass in precomputed sampled structure anchor patches and random walks from which to further subsample
                self.anchors_structure = init_anchors_structure(self.hparams,  self.structure_anchors, self.int_structure_anchor_random_walks, self.bor_structure_anchor_random_walks) 
            else: self.anchors_structure = None

            print('--- Finished Preparing Data ---', flush=True)




    def read_subgraph_data(dataset_name,graph_id,networkx_graph,node_embeds):
        '''
        Read in the subgraphs & their associated labels
        '''

        sub_f='./dataset/{}/{}/subgraphs.pth'.format(dataset_name,graph_id)
        subgraph_list=[]

        with open(sub_f) as fin:
            subgraph_idx = 0
            for line in fin:
                nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
                if len(nodes) != 0:
                    if len(nodes) == 1: print(nodes)
                    l = line.split("\t")[1].split("-")                   
                    subgraph_list.append(nodes)
                    subgraph_idx += 1

        subgraph_list=torch.tensor(subgraph_list).long().squeeze()
    
        # renumber nodes to start with index 1 instead of 0
        mapping = {n:int(n)+1 for n in networkx_graph.nodes()}
        networkx_graph = nx.relabel_nodes(networkx_graph, mapping)

        new_subg = []
        for subg in subgraph_list:
            new_subg.append([c + 1 for c in subg])
        subgraph_list=new_subg

        # Initialize pretrained node embeddings
        node_embed_size = node_embeds.shape[1]
        zeros = torch.zeros(1, node_embed_size)
        embeds = torch.cat((zeros, node_embeds), 0) #there's a zeros in the first index for padding
    

        print('--- Finished reading in data ---')
        return subgraph_list,embeds




    def forward(self, batch_graph, batch_hyper_graph, batch_motif2A):
        #
        #   subgraph embedding
        #
        # create cc_embeds matrix for each channel: (batch_sz, max_n_cc, hidden_dim)

        batch_subgraph_embeddings=[]

        for graph in batch_graph:
            
            N_in_cc_embeds = torch.index_select(N_I_cc_embed, 0, subgraph_idx.squeeze(-1))
            N_out_cc_embeds = torch.index_select(N_B_cc_embed, 0, subgraph_idx.squeeze(-1))
            P_in_cc_embeds = torch.index_select(P_I_cc_embed, 0, subgraph_idx.squeeze(-1))
            P_out_cc_embeds = torch.index_select(P_B_cc_embed, 0, subgraph_idx.squeeze(-1))
            S_in_cc_embeds = torch.index_select(S_I_cc_embed, 0, subgraph_idx.squeeze(-1))
            S_out_cc_embeds = torch.index_select(S_B_cc_embed, 0, subgraph_idx.squeeze(-1))

            
            
            
            # for each layer in SubGNN:
            outputs = []
            for l in range(self.hparams['n_layers']):

                N_in_cc_embeds, _ = self.run_mpn_layer(dataset_type, self.neighborhood_mpns[l]['internal'], subgraph_ids, subgraph_idx, cc_ids, N_in_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='neighborhood', inside=True)
                N_out_cc_embeds, _ = self.run_mpn_layer(dataset_type, self.neighborhood_mpns[l]['border'], subgraph_ids, subgraph_idx, cc_ids, N_out_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='neighborhood', inside=False)
                
                outputs.extend([N_in_cc_embeds, N_out_cc_embeds])

                P_in_cc_embeds, P_in_position_embed = self.run_mpn_layer(dataset_type, self.position_mpns[l]['internal'], subgraph_ids, subgraph_idx,  cc_ids, P_in_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='position', inside=True)
                P_out_cc_embeds, P_out_position_embed = self.run_mpn_layer(dataset_type, self.position_mpns[l]['border'], subgraph_ids, subgraph_idx, cc_ids, P_out_cc_embeds, cc_embed_mask, NP_sim, layer_num=l, channel='position', inside=False)

                outputs.extend([P_in_position_embed, P_out_position_embed])
        

                S_in_cc_embeds, S_in_struc_embed = self.run_mpn_layer(dataset_type, self.structure_mpns[l]['internal'], subgraph_ids, subgraph_idx, cc_ids, S_in_cc_embeds, cc_embed_mask, I_S_sim, layer_num=l, channel='structure', inside=True)
                S_out_cc_embeds, S_out_struc_embed = self.run_mpn_layer(dataset_type, self.structure_mpns[l]['border'], subgraph_ids, subgraph_idx, cc_ids, S_out_cc_embeds, cc_embed_mask, B_S_sim, layer_num=l, channel='structure', inside=False)
                    
                outputs.extend([S_in_struc_embed, S_out_struc_embed])

            
            # concatenate all layers
            all_cc_embeds = torch.cat([init_cc_embeds] + outputs, dim=-1)

            batch_subgraph_embeddings.append(subgraph_embedding)

        #
        #   node aggregation
        #

        new_hyper_graph = []
        new_motif2A = []
        for graph_idx in range(len(batch_motif2A)):
            if batch_hyper_graph[graph_idx].max_neighbor == 0:
                continue
            new_hyper_graph.append(batch_hyper_graph[graph_idx])
            new_motif2A.append(batch_motif2A[graph_idx])
        batch_hyper_graph = new_hyper_graph
        batch_motif2A = new_motif2A

        X1_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        hyper_graph_pool= self.__preprocess_graphpool(batch_hyper_graph)
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block1 = self.__preprocess_neighbors_sumavepool(batch_graph)
        # list of hidden representation at each layer (including input)
        hidden_rep1 = [X1_concat]
        h1 = X1_concat
        hidden_rep2=[]

        if len(batch_hyper_graph)!=0:

            X2_concat = torch.cat([graph.node_features for graph in batch_hyper_graph], 0).to(self.device)
            hidden_rep2.append(X2_concat)
            motif2A = np.zeros((X1_concat.shape[0], X2_concat.shape[0]))
            start_node_id = 0
            start_motif_id = 0
            elem = 0
            for graph_idx in range(len(batch_motif2A)):
                if batch_hyper_graph[graph_idx].max_neighbor == 0:
                    continue
                mat = np.array(np.where(np.array(batch_motif2A[graph_idx]) == 1)).T
                elem += mat.shape[0]
                mat[:, 0] += start_motif_id
                mat[:, 1] += start_node_id
                for m in mat:
                    motif2A[m[1]][m[0]] = 1
                start_node_id += np.array(batch_motif2A[graph_idx]).shape[1]
                start_motif_id += np.array(batch_motif2A[graph_idx]).shape[0]
            motif2A = torch.FloatTensor(motif2A).to(self.device)
            Adj_block2 = self.__preprocess_neighbors_sumavepool(batch_hyper_graph)
            h2 = X2_concat.to(torch.float32)

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h1 = self.next_layer_eps(h1, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h1 = self.next_layer_eps(h1, layer, Adj_block=Adj_block1)
                if len(batch_hyper_graph) != 0:
                    h2 = self.hyper_next_layer_eps(h2, layer, Adj_block=Adj_block2)
                # h2 = self.next_layer_eps(h2, layer, Adj_block=Adj_block2)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h1 = self.next_layer(h1, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h1 = self.next_layer(h1, layer, Adj_block=Adj_block1)
                if len(batch_hyper_graph) != 0:
                    h2 = self.hyper_next_layer(h2, layer, Adj_block=Adj_block2)
                # h2 = self.next_layer(h2, layer, Adj_block=Adj_block2)

            if len(batch_hyper_graph) != 0:
                softmax = torch.nn.Softmax(0)
                if self.multi_head:
                    ab = softmax(self.atts[layer])
                else:
                    ab=softmax(self.atts)
                # h22 = torch.mm(motif2A, h2)
                h22=subgraph_embedding
                h22 = ab[1] * h22
                h1 = ab[0] * h1
                h1 = h1 + h22

            # h2 = 0.8 * h2 + 0.2 * torch.mm(motif2A.T, h1)
            # h2=torch.mm(motif2A,h2)
            # atten=torch.softmax(dim=1)
            hidden_rep1.append(h1)
            #hidden_rep2.append(h2)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for layer, h1 in enumerate(hidden_rep1):
            pooled_h = torch.spmm(graph_pool, h1)
            #pooled_h2 = torch.spmm(hyper_graph_pool, hidden_rep2[layer])
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                                          training=self.training)
            #score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h2), self.final_dropout,
                                          #training=self.training)

        return score_over_layer