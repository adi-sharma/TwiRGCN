from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from transformers import DistilBertModel
import math
import numpy as np

from torch_geometric.data import Data
from torch_scatter import scatter_add

from twirgcn_data_utils import cos_sim_for_edge_attn


###########################################################
##############      Basic Functions      ##################

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
        

###########################################################

def get_BERT_model(args):
    pretrained_weights = 'distilbert-base-uncased'
    roberta_model = DistilBertModel.from_pretrained(pretrained_weights)
    if args.lm_frozen == 1:
        print('Freezing LM params')
        for param in roberta_model.parameters():
            param.requires_grad = False
    else:
        print('Unfrozen LM params')       
    return roberta_model      


###########################################################

def element_wise_mult(a, b):
    N = b.shape[0]
    shp = b.shape[1:]
    return torch.mul(a, b.view(N, -1).transpose(0,1)).transpose(0,1).view(N, *shp) 

################          End           ###################        
########################################################### 



###########################################################
############      TwiRGCN Layer      ##############


class TwiRGCN_Layer(MessagePassing):
    """
    Will comment later
    
    """

    def __init__(self, args, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(TwiRGCN_Layer, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.scale_edge_thick = torch.tensor(1.0, dtype = torch.float, requires_grad=True)        
        self.attn_mode = args.attn_mode
        
    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, start_time, end_time, ques_time_emb, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, start_time=start_time, end_time=end_time, ques_time_emb=ques_time_emb,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, start_time, end_time, ques_time_emb, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        
        if self.attn_mode == 'interval':
            # avg(cos(t1, tq), cos(t2, tq))
            time_component = torch.cat((cos_sim_for_edge_attn(start_time, ques_time_emb).unsqueeze(0), 
                                        cos_sim_for_edge_attn(end_time, ques_time_emb).unsqueeze(0)), dim=1)
            edge_thickness = torch.mean(time_component, 1)
        elif self.attn_mode == 'average':
            # cos(avg(t1, t2), tq)
            time_component = (start_time +  end_time) / 2      
            edge_thickness = cos_sim_for_edge_attn(time_component, ques_time_emb) 
                
        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            edge_minus_time = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)                 
            out = element_wise_mult(edge_thickness.squeeze(), edge_minus_time)  
        
        return out if edge_norm is None else out * edge_norm.view(-1, 1)


    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


################          End           ###################        
###########################################################