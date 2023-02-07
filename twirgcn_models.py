import math
from this import d
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tkbc.models import TComplEx

from twirgcn_data_utils import padding_index_vector_with_neg_one, element_wise_division, cosine_similarity_batched, vec_to_idx_vex
from twirgcn_base import TwiRGCN_Layer, get_BERT_model


###########################################################


###########################################################
###############            twi-rgcn          ##############


class TwiRGCN(nn.Module):
    # def __init__(self, tkbc_model, args):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.roberta_model = get_BERT_model(args)
        
        # NEW RGCN STUFF
        self.num_relations = tkbc_model.embeddings[1].weight.shape[0] // 2 # divide by 2 since embeddings contain reciprocal 
        self.num_bases = 4       # hard coded need to change later !!!
        self.dropout_ratio = 0.3 # hard coded need to change later !!!
        
        self.dataset_name = args.dataset_name
        self.no_gating = args.no_gating


        # Getting TComplex Embeddings
        self.tkbc_model = tkbc_model
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data
        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        self.entity_time_embedding = nn.Embedding(num_entities + num_times, self.tkbc_embedding_dim)
        self.entity_time_embedding.weight.data.copy_(full_embed_matrix)
        self.num_entities = num_entities
        self.num_times = num_times
        
        zero_emb = torch.zeros(self.tkbc_embedding_dim, requires_grad=False)  ## For Batching
        self.zero_embedding = zero_emb.detach().cuda()

        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
            for param in self.tkbc_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')


        ## LAYERS
        self.conv1 = TwiRGCN_Layer(args,
            self.tkbc_embedding_dim, self.tkbc_embedding_dim, self.num_relations, num_bases=self.num_bases)
        self.conv2 = TwiRGCN_Layer(args,
            self.tkbc_embedding_dim, self.tkbc_embedding_dim, self.num_relations, num_bases=self.num_bases)

        self.linear = nn.Linear(768, self.tkbc_embedding_dim) # to project question embedding for score pooling
        self.lin_ques_for_attn = nn.Linear(768, self.tkbc_embedding_dim) # to project question embedding for edge attn
        self.prob_select_linear = nn.Linear(768, 1) # to get probability of ent or time answer     
        self.sig_m = nn.Sigmoid()        
        
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding
    
    def trgcn_attn_forward_pass(self, entity, edge_index, edge_type, start_time, end_time, ques_time_emb, edge_norm):
        st = self.entity_time_embedding(start_time + self.num_entities)
        et = self.entity_time_embedding(end_time + self.num_entities)
        
        x = self.entity_time_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, st, et, ques_time_emb, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type, st, et, ques_time_emb, edge_norm)
        
        return x
        

    def forward(self, a):
        a.cuda()
        question_tokenized = a.input_ids
        question_attention_mask = a.attention_mask
        head = a.q_head
        tail = a.q_tail
        time = a.q_time
        
        # Getting pretrained BERT embedding for question    
        ques_emb_bert = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        question_embedding = self.linear(ques_emb_bert)     #.squeeze()
        
        # To get ques to batched num edges
        num_edges_per_batch_idx = vec_to_idx_vex(a.edges_in_batch)
        ques_batched_to_num_edges = ques_emb_bert[num_edges_per_batch_idx]   
        time_for_edge_attn = self.lin_ques_for_attn(ques_batched_to_num_edges)
        
        # Get unique entities and times per subgraph
        unique_times = a.uniq_times + self.num_entities
        unique_entities = a.x
        
        # Updating embeddings of entities and times in given subgraph
        subgraph_entity_emb = self.trgcn_attn_forward_pass(unique_entities, a.edge_index, a.edge_type, a.start_time, a.end_time, time_for_edge_attn, a.edge_norm)
        subgraph_time_emb = self.entity_time_embedding(unique_times)
        
        # Padding with zero embedding for batch masking during mean pooling
        sg_entities_w_zeros = torch.cat((subgraph_entity_emb, self.zero_embedding.unsqueeze(0)), 0) ## For Batching
        sg_times_w_zeros = torch.cat((subgraph_time_emb, self.zero_embedding.unsqueeze(0)), 0) ## For Batching
        
        # Mean pooling over all unique entities and times in given subgraph        
        padded_uniq_entities, num_uniq_entities = padding_index_vector_with_neg_one(a.max_idx_entities)
        padded_uniq_times, num_uniq_times = padding_index_vector_with_neg_one(a.max_idx_times)
        pooled_entity_emb = element_wise_division(torch.sum(sg_entities_w_zeros[padded_uniq_entities], dim=1), num_uniq_entities)       
        pooled_time_emb = element_wise_division(torch.sum(sg_times_w_zeros[padded_uniq_times], dim=1), num_uniq_times)
        
        # Calculating answer
        if self.no_gating == 1:
            # TwiRGCN w/o answer gating
            predicted_embedding = (question_embedding + pooled_entity_emb + pooled_time_emb)/3 # c_p = 3
        else:
            # TwiRGCN gated based on answer being entity or time
            p_ent = self.sig_m(self.prob_select_linear(ques_emb_bert)) # probability of ent or time answer
            predicted_embedding = (question_embedding + (pooled_entity_emb * p_ent) + (pooled_time_emb * (1 -p_ent)))/3 # c_p = 3

        # Finding entity/time closest to predicted answer  
        score = cosine_similarity_batched(predicted_embedding, self.entity_time_embedding.weight.T)
        
        # Scalining score        
        final_score = score * 30 
                             
        return final_score



################          End           ###################
###########################################################