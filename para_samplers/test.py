import ParallelSampler as cpp_para_sampler

import numpy as np
from numpy.core.fromnumeric import shape
from scipy.sparse import csr_matrix
import scipy as spy

n=20
m=20
density=0.3

B=spy.sparse.rand(m,n,density=density,format='csr',dtype=int)
B.data=np.ones((len(B.data),),int)
print(B.todense())
adj=B

# node_target=[i for i in range(20)]
node_target=[5,6,7]


config={'k': 1, 'threshold': 0.01, 'epsilon': '1e-5', 'add_self_edge': True, 'method': 'ppr', 'size_root': 1, 
        'fix_target': True, 'sequential_traversal': True, 'type_': 0, 'name_data': 'arxiv', 
        'dir_data': {'local': './data', 'is_adj_changed': False, 'is_feat_changed': False}, 'is_transductive': True}


args_Cpp_sampler = [node_target, 3, 1, config["fix_target"], config["sequential_traversal"], 1]
args_Cpp_sampler = [adj.indptr, adj.indices, adj.data] + args_Cpp_sampler + ["", "", ""]

para_sampler = cpp_para_sampler.ParallelSampler(*args_Cpp_sampler)

#para_sampler.preproc_ppr_approximate(200, 0.85, 1e-5, 'data/test/neighs.bin', 'data/test/scores.bin')
para_sampler.preproc_ppr_approximate(200, 0.85, 1e-5, '', '')


cpp_config_list = [{'method': 'ppr', 'k': '6', 'num_roots': '1', 
                'threshold': '0.01', 'add_self_edge': 'true', 'return_target_only': 'false'}]
cpp_aug_list = [{'hops'}]

# out=para_sampler.read_PPR_from_binary_file('data/testname/neighs.bin', 'data/testname/scores.bin', 200, 0.85, 1e-5)

# print(out)
ret = para_sampler.parallel_sampler_ensemble(cpp_config_list, cpp_aug_list)
print(len(ret))
print(ret[0].get_num_valid_subg())
print(ret[0].get_subgraph_node())




