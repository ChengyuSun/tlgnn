// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <stdint.h>

namespace py = pybind11;

typedef uint32_t NodeType;    // work for as large as ogbn-papers100M
typedef float PPRType;

class GraphStruct{
 public:
  GraphStruct() {}
  GraphStruct(
      std::vector<NodeType> indptr_,
      std::vector<NodeType> indices_,
      std::vector<float> data_) :
    indptr(indptr_), indices(indices_), data(data_) {
      num_nodes = indptr_.size() - 1;
      num_edges = indices_.size();
      assert(indptr_[num_nodes] == num_edges);
      assert(indptr_[0] == 0);
      // std::cout << "NUM NODES: " << num_nodes << "\tNUM EDGES: " << num_edges << std::endl << std::flush;
    }
  std::vector<NodeType> get_degrees();
  std::vector<NodeType> indptr;
  std::vector<NodeType> indices;
  std::vector<float> data;
  NodeType num_nodes;
  NodeType num_edges;
};

class SubgraphStruct{
 public:
  SubgraphStruct() {}
  void compute_hops();    // compute hops to target, fill in hop vec
  std::vector<NodeType> indptr;
  std::vector<NodeType> indices;
  std::vector<float> data;
  std::vector<NodeType> origNodeID;
  std::vector<NodeType> origEdgeID;
  std::vector<NodeType> target;
  // additional info to augment node feature
  std::vector<NodeType> hop;     // length = num subg nodes. 
  std::vector<PPRType> ppr;   // ppr for the single target
};

class SubgraphStructVec{
 public:
  SubgraphStructVec() {}
  SubgraphStructVec(int num_subgraphs) {
    indptr_vec.resize(num_subgraphs);
    indices_vec.resize(num_subgraphs);
    data_vec.resize(num_subgraphs);
    origNodeID_vec.resize(num_subgraphs);
    origEdgeID_vec.resize(num_subgraphs);
    target_vec.resize(num_subgraphs);
    hop_vec.resize(num_subgraphs);
    ppr_vec.resize(num_subgraphs);
  }
  std::vector<std::vector<NodeType>> indptr_vec;
  std::vector<std::vector<NodeType>> indices_vec;
  std::vector<std::vector<float>> data_vec;
  std::vector<std::vector<NodeType>> origNodeID_vec;
  std::vector<std::vector<NodeType>> origEdgeID_vec;
  std::vector<std::vector<NodeType>> target_vec;
  // additional info
  std::vector<std::vector<NodeType>> hop_vec;
  std::vector<std::vector<PPRType>> ppr_vec;
  int num_valid_subg_cur_batch = 0;

  void add_one_subgraph_vec(SubgraphStruct& subgraph, int p);
  // getters
  int get_num_valid_subg();
  const std::vector<std::vector<NodeType>>& get_subgraph_indptr();
  const std::vector<std::vector<NodeType>>& get_subgraph_indices();
  const std::vector<std::vector<float>>& get_subgraph_data();
  const std::vector<std::vector<NodeType>>& get_subgraph_node();
  const std::vector<std::vector<NodeType>>& get_subgraph_edge_index();
  const std::vector<std::vector<NodeType>>& get_subgraph_target();
  const std::vector<std::vector<NodeType>>& get_subgraph_hop();
  const std::vector<std::vector<PPRType>>& get_subgraph_ppr();
};


class ParallelSampler{
 public:
  ParallelSampler(
      std::vector<NodeType> indptr_full_,
      std::vector<NodeType> indices_full_,
      std::vector<float> data_full_,
      std::vector<NodeType> nodes_target_,
      int num_sampler_per_batch_,
      int max_num_threads_,
      bool fix_target_,
      bool sequential_traversal_,
      int num_subgraphs_ensemble=1,
      std::string path_indptr="",
      std::string path_indices="",
      std::string path_data="") {
    if (indptr_full_.size() == 0) {
      read_array_from_bin(path_indptr, indptr_full_);
    }
    if (indices_full_.size() == 0) {
      read_array_from_bin(path_indices, indices_full_);
    }
    // if (data_full_.size() == 0) {
    //   assert(path_data.length() == 0);    // temporarily only deal with binary adj. Make the loader a template in the future. 
    //   data_full_.resize(indices_full_.size(), 1.);
    // }
    // NOTE TODO: right now we don't use any info from data array (we simply fill in 1. in subgraph)
    std::srand(std::time(0));
    num_sampler_per_batch = num_sampler_per_batch_;   // this must be > 0
    if (max_num_threads_ <= 0) {                      // this can be <= 0, in which case we just let openmp decide 
      max_num_threads = omp_get_max_threads();
    } else {
      max_num_threads = max_num_threads_;
      omp_set_num_threads(max_num_threads);
      assert(max_num_threads == omp_get_max_threads());
    }
    nodes_target = nodes_target_;
    fix_target = fix_target_;
    sequential_traversal = sequential_traversal_;
    graph_full = GraphStruct(indptr_full_, indices_full_, data_full_);
    subgraphs_ensemble = std::vector<SubgraphStructVec>(num_subgraphs_ensemble, SubgraphStructVec(num_sampler_per_batch));
  }
  // utilility functions
  NodeType num_nodes();
  NodeType num_edges();
  NodeType num_nodes_target();
  // sampler: right now for all samplers, we return node-induced subgraph
  std::vector<SubgraphStructVec> parallel_sampler_ensemble(
        std::vector<std::unordered_map<std::string, std::string>> configs_samplers, 
        std::vector<std::set<std::string>> configs_aug
      );
  void preproc_ppr_approximate(int k, float alpha, float epsilon, 
        std::string fname_neighs, std::string fname_scores);
  NodeType get_idx_root();             // for assertion of correctness
  bool is_seq_root_traversal();   // for assertion of correctness
  void shuffle_targets();
  // ----------------
  // [DANGER] For extreme memory size optimization
  void drop_full_graph_info();
  
 private:
  void read_array_from_bin(std::string name_file, std::vector<NodeType> &ret);
  std::vector<std::set<NodeType>> _get_roots_p(int num_roots);
  SubgraphStruct _node_induced_subgraph(std::unordered_map<NodeType, PPRType>& nodes_touched, std::set<NodeType>& targets, bool include_self_conn);
  SubgraphStruct khop(std::set<NodeType>& targets, std::unordered_map<std::string, std::string>& config, std::set<std::string>& config_aug);   // config should specify k and budget
  SubgraphStruct ppr(std::set<NodeType>& targets, std::unordered_map<std::string, std::string>& config, std::set<std::string>& config_aug);    // config should specify k
  SubgraphStruct nodeIID(std::set<NodeType>& targets, std::set<std::string>& config_aug);    // config doesn't need to specify anything
  SubgraphStruct dummy_sampler(std::set<NodeType>& targets);
  void cleanup_history_subgraphs_ensemble();
  void write_PPR_to_binary_file(std::string name_out_neighs, std::string name_out_scores, int k, float alpha, float epsilon);
  bool read_PPR_from_binary_file(std::string name_in_neighs, std::string name_in_scores, int k, float alpha, float epsilon);
  std::vector<NodeType> nodes_target;
  std::vector<std::vector<NodeType>> top_ppr_neighs;
  std::vector<std::vector<PPRType>> top_ppr_scores;      // may not essentially need it   // TODO may change it to be even more compact, since we only need to preserve relative values
  GraphStruct graph_full;
  std::vector<SubgraphStructVec> subgraphs_ensemble;
  int num_sampler_per_batch;
  int max_num_threads;
  double time_sampler_total = 0;
  double time_induction_total = 0;
  bool fix_target;
  bool sequential_traversal;
  NodeType idx_root = 0;  // counter for sequential traversing the roots
                          // shared among all sampler ensembles
};
