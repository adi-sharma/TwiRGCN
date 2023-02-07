import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Data
from torch_scatter import scatter_add


#########################

class My_Data(Data):
    def __init__(self, edge_index=None, x=None, uniq_times=None, max_idx_entities=None, max_idx_times=None):
        super().__init__()
        self.edge_index = edge_index
        self.x = x
        self.max_idx_entities = max_idx_entities
        self.max_idx_times = max_idx_times
        self.uniq_times = uniq_times
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index' or key == 'max_idx_entities':
            return self.x.size(0)
        elif key == 'max_idx_times':
            return self.uniq_times.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
#########################
        
def edge_normalization_data(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

#########################

def generate_graph_data(nbhood_facts, q_head, q_tail, q_time, num_rels):

    h,r,t,st,et = np.transpose(nbhood_facts)
    
    # Handling h, r, t preprocessing
    uniq_entity, edges = np.unique((h, t), return_inverse=True)
    h, t = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((h, r, t)).transpose()
    
    h = torch.tensor(h, dtype = torch.long)
    t = torch.tensor(t, dtype = torch.long)
    
    edge_index = torch.stack((h, t))
    edge_type = torch.from_numpy(r)
    
    # Handling start and end time preprocessing
    uniq_time = np.unique((st, et))
    
    st = torch.tensor(st, dtype = torch.long)
    et = torch.tensor(et, dtype = torch.long)
    
    q_times_as_array = np.full(len(r), q_time)
    num_edg = len(r)
    
    max_idx_ent = len(uniq_entity)
    max_idx_tim = len(uniq_time)
    
    # Loading graph data structure
    data = My_Data(x = torch.from_numpy(uniq_entity), edge_index = edge_index, uniq_times = torch.from_numpy(uniq_time),
                   max_idx_entities = torch.tensor(max_idx_ent, dtype = torch.long),
                   max_idx_times = torch.tensor(max_idx_tim, dtype = torch.long))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization_data(edge_type, edge_index, len(uniq_entity), num_rels)
    data.start_time = st
    data.end_time = et
    data.q_times_as_array = torch.tensor(q_times_as_array, dtype = torch.long)
    data.edges_in_batch = torch.tensor(num_edg, dtype = torch.long)
    
    data.q_head = torch.tensor(q_head, dtype = torch.long)
    data.q_tail = torch.tensor(q_tail, dtype = torch.long)
    data.q_time = torch.tensor(q_time, dtype = torch.long)
    
    
    return data

#########################

def padding_index_vector_with_neg_one(idx_vec):
    max_len = idx_vec[-1]
    m_pad = torch.nn.ConstantPad1d((0, max_len - idx_vec[0]), -1)
    # out_np = np.pad(np.arange(idx_vec[0]), (0, max_len - idx_vec[0]), 'constant', constant_values=(-1, -1))
    out_torch = m_pad(torch.arange(idx_vec[0])).unsqueeze(0)
    num_uniq = idx_vec[0].unsqueeze(0)

    for i, val in enumerate(idx_vec):
        if i == 0:
          continue
        else:
          i_pad = torch.nn.ConstantPad1d((idx_vec[i-1], max_len - idx_vec[i]), -1)
          # idx_seq_np = np.pad(np.arange(val), (0, max_len - val), 'constant', constant_values=(-1, -1))
          idx_seq_torch = i_pad(torch.arange(idx_vec[i-1], idx_vec[i])).unsqueeze(0)
          num_current = idx_vec[i] - idx_vec[i-1]
          out_torch = torch.cat((out_torch, idx_seq_torch), 0)
          num_uniq = torch.cat((num_uniq, num_current.unsqueeze(0)), 0)
          
    return out_torch.cuda(), num_uniq.cuda()

        

#########################

def element_wise_division(a, b):
    N = a.shape[0]
    shp = a.shape[1:]
    
    return torch.div(a.view(N, -1).transpose(0,1), b).transpose(0,1).view(N, *shp)  


#########################

def cosine_similarity_batched(x1, x2):
    x1_norm = x1 / torch.linalg.norm(x1, dim=1)[:, None]
    x2_norm = x2 / torch.linalg.norm(x2, dim=0)[None,:]
    return torch.matmul(x1_norm, x2_norm)

#########################

def vec_to_idx_vex(vec_for_idx):
    idx_vec_final = torch.zeros(vec_for_idx[0], dtype = torch.long).unsqueeze(0)        

    for i, val in enumerate(vec_for_idx):
        if i == 0:
            continue
        else:
            idx_temp = torch.full((1,vec_for_idx[i]), i, dtype = torch.long)
            idx_vec_final = torch.cat((idx_vec_final, idx_temp), dim=1)
    return idx_vec_final.squeeze()

#########################

def cos_sim_for_edge_attn(ab, bc):
    bc_norm = bc / torch.linalg.norm(bc, dim=1)[:, None]
    ab_norm = ab / torch.linalg.norm(ab, dim=1)[:, None]    
    return torch.sum((ab_norm * bc_norm),dim=1)

#########################

