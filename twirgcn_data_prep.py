from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List
import json

import numpy as np
import torch
import utils
from tqdm import tqdm
import random
from torch_geometric.data import Dataset
from twirgcn_data_utils import generate_graph_data

from transformers import RobertaTokenizer
from transformers import DistilBertTokenizer
from transformers import BertTokenizer


###########################################################


class QA_Dataset(Dataset):
    def __init__(self, 
                split,
                dataset_name,
                tokenization_needed=True):
        filename = 'data/{dataset_name}/questions/{split}.pickle'.format(
            dataset_name=dataset_name,
            split=split
        )
        # questions = pickle.load(open(filename, 'rb'))
        with open(filename, "rb") as f:
            questions = pickle.load(f)
        self.tokenizer_class = DistilBertTokenizer 
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.all_dicts = utils.getAllDicts(dataset_name)
        print('Total questions = ', len(questions))
        self.data = questions
        self.tokenization_needed = tokenization_needed

    def getEntitiesLocations(self, question):
        question_text = question['question']
        entities = question['entities']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e]
            location = question_text.find(e)
            loc_ent.append((location, e_id))
        return loc_ent

    def getTimesLocations(self, question):
        question_text = question['question']
        times = question['times']
        ts2id = self.all_dicts['ts2id']
        loc_time = []
        for t in times:
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id']) # add num entities
            location = question_text.find(str(t))
            loc_time.append((location, t_id))
        return loc_time

    def isTimeString(self, s):
        # todo: cant do len == 4 since 3 digit times also there
        if 'Q' not in s:
            return True
        else:
            return False

    def textToEntTimeId(self, text):
        if self.isTimeString(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id'])
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id

    def rawFactToIdFact(self, fact):
        start_time, end_time = self.timesToIds([fact[3], fact[4]])
        sub, obj = self.entitiesToIds([fact[0], fact[2]])
        rel = self.all_dicts['rel2id'][fact[1]]
        out_fact = [sub, rel, obj, start_time, end_time]
        return out_fact

    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e])
        return output
    
    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'
    
    def getEntityToText(self, entity_wd_id):
        return self.all_dicts['wd_id_to_text'][entity_wd_id]
    
    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return self.getEntityToText(ent)
    
    def getEntityIdToWdId(self, id):
        return self.all_dicts['id2ent'][id]

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            output.append(ts2id[(t, 0, 0)])
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return answers
    
    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):
        s, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return s, answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored 
    # while the position with the value of False will be unchanged.
    # 
    # so we want to pad with True
    def padding_tensor(self, sequences, max_len = -1):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool) # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False # fills good area with False
        return out_tensor, mask
    
    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def prepare_data(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        entity_time_ids = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in data:
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            question_text.append(q_text)
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        return {'question_text': question_text, 
                'entity_time_ids': entity_time_ids, 
                'answers_arr': answers_arr}
    
    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1]) # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = [] # we want for each paraphrase
            template = question['template']
            for nl_question in question['paraphrases']:
                keyword_dict = self.get_keyword_dict(template, nl_question)
                keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output


class QA_Dataset_TimeQuestions(QA_Dataset):
    def __init__(self, split, dataset_name, tokenization_needed=True):
        super().__init__(split, dataset_name, tokenization_needed)
        print('Preparing data for split %s' % split)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.num_relations = len(self.all_dicts['rel2id'])
        self.max_question_seq_length = 30
        self.prepared_data = self.prepare_data_(self.data)
        
        if split == 'train': # TODO do this only for train
            self.prepared_data = self.removeNoAnswerQuestions(self.prepared_data)
        self.prepared_data = self.fixNoSubgraphQuestions(self.prepared_data)
        
        self.answer_vec_size = self.num_total_entities + self.num_total_times

    def removeNoAnswerQuestions(self, data):
        out_data = {}
        for key in data.keys():
            out_data[key] = []
        for id in range(len(data['question_text'])):
            if len(data['answers_arr'][id]) > 0:
                for key in data.keys():
                    out_data[key].append(data[key][id])
        return out_data

    def fixNoSubgraphQuestions(self, data):
        out_data = {}
        for key in data.keys():
            out_data[key] = []
        count = 0
        for id in range(len(data['question_text'])):
            if len(data['nbhood_facts'][id]) == 0:
                count += 1
                data['nbhood_facts'][id] = np.array([[0,0,0,0,0]])
            for key in data.keys():
                out_data[key].append(data[key][id])
                
        print('Fixed %d no subgraph questions' % count)
        return out_data

    def time2id(self, time):
        return self.all_dicts['ts2id'][(time, 0, 0)]

    def answer2id(self, answer):
        # print(answer)
        if isinstance(answer, int):
            if (answer, 0, 0) in self.all_dicts['ts2id']:
                return self.all_dicts['ts2id'][(answer, 0, 0)] + self.num_total_entities 
            else:
                return -100
        else:
            if answer in self.all_dicts['ent2id']:
                return self.all_dicts['ent2id'][answer]
            else:
                return -100

    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        nbhood_facts_list = []
        ent2id = self.all_dicts['ent2id']
        # self.data=[]
        for i,question in enumerate(data):
            # print(question)

            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            
            if 'nbhood_facts' in question.keys():
                nbhood_facts_transformed = [self.rawFactToIdFact(f) for f in question['nbhood_facts']]
                nbhood_facts_transformed = np.array(nbhood_facts_transformed)
                nbhood_facts_list.append(nbhood_facts_transformed)
            # annotation = question['annotation']
            # head = ent2id[annotation['head']]
            # tail = ent2id[annotation['tail']]
            # entities = list(question['entities'])

            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in entities_list_with_locations] # ordering necessary otherwise set->list conversion causes randomness
            
            # Padding at -1
            if len(entities) == 0:
                # entities = [0] # TODO: is this padding?
                entities = [-1]
            head = entities[0] # take an entity
            
            if len(entities) > 1:
                tail = entities[1]
            else:
                # tail = 0
                tail = -1
            times_in_question = question['times']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0] # take a time. if no time then 0
                # exit(0) 
            else:
                # print('No time in qn!')
                # time = 0
                time = -1
            
            # time += num_total_entities
            heads.append(head)
            times.append(time)
            tails.append(tail)
            question_text.append(q_text)
            
            # get answer ids this way
            # print(question)
            # if x is an entity not in kg, answer2id returns -100
            # we need to remove that
            # this should only affect train, not test
            answers = [self.answer2id(x) for x in question['answers']]
            if -100 in answers:
                answers = [x for x in answers if x >= 0]
            answers_arr.append(answers)
            
        out =  {'question_text': question_text, 
                'head': heads, 
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}

        # nbhood_facts is a list of facts for a question
        # each fact is 5 tuple, containing integer ids of the entity/relation/timestamp
        if len(nbhood_facts_list) > 0:
            out['nbhood_facts'] = nbhood_facts_list

        return out

    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        # return len(self.data)
        return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        if len(answers_arr) == 0:
            answers_single = 0
        else:
            answers_single = random.choice(answers_arr)
        nbhood_facts = data['nbhood_facts'][index]
        q_tokenized = self.tokenizer(question_text, padding='max_length', truncation=True, max_length=self.max_question_seq_length, return_tensors="pt")
        graph_data = generate_graph_data(nbhood_facts, head, tail, time, self.num_relations)
        graph_data.answers_single = torch.tensor(answers_single, dtype = torch.long)
        graph_data.input_ids = q_tokenized['input_ids']
        graph_data.attention_mask = q_tokenized['attention_mask']
        return graph_data
        
    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3] for item in items]))
        nbhood_facts_np = [item[4] for item in items]
        nbhood_facts = [torch.from_numpy(q_facts) for q_facts in nbhood_facts_np]
        answers_single = torch.from_numpy(np.array([item[5] for item in items])) # answers is the last one
        return b['input_ids'], b['attention_mask'], heads, tails, times, nbhood_facts, answers_single # last one has to be answer

    def get_dataset_ques_info(self):
        type2num={}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]]=0
            type2num[question["type"]]+=1
        return {"type2num":type2num, "total_num":len(self.data_ids_filtered)}.__str__()

