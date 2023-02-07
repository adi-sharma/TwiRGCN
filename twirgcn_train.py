import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np
import math
import json

from twirgcn_models import TwiRGCN
from twirgcn_data_prep import QA_Dataset_TimeQuestions
from torch_geometric.loader import DataLoader

#----------------------
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#----------------------
from pytorchtools import EarlyStopping
import utils
from tqdm import tqdm

from utils import loadTkbcModel, loadTkbcModel_complex #
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict

###########################################################

###########################################################
###############      Args Handling      ###################


parser = argparse.ArgumentParser(
    description="Temporal KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='model_tkbc_60kent.ckpt', type=str,
    help="Pretrained tkbc model checkpoint"
)
parser.add_argument(
    '--model', default='model1', type=str,
    help="Which model to use."
)
parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)
parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)
parser.add_argument(
    '--max_epochs', default=100, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--eval_k', default=1, type=int,
    help="Hits@k used for eval. Default 10."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--num_transformer_heads', default=8, type=int,
    help="Num heads for transformer"
)
parser.add_argument(
    '--transformer_dropout', default=0.1, type=float,
    help="Dropout transformer"
)
parser.add_argument(
    '--ls', default=0.0, type=float,
    help="Label smoothing"
)
parser.add_argument(
    '--num_transformer_layers', default=6, type=int,
    help="Num layers for transformer"
)
parser.add_argument(
    '--batch_size', default=256, type=int,
    help="Batch size."
)
parser.add_argument(
    '--valid_batch_size', default=50, type=int,
    help="Valid batch size."
)
parser.add_argument(
    '--frozen', default=1, type=int,
    help="Whether entity/time embeddings are frozen or not. Default frozen."
)
parser.add_argument(
    '--lm_frozen', default=1, type=int,
    help="Whether language model params are frozen or not. Default frozen."
)
parser.add_argument(
    '--lr', default=2e-4, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--mode', default='train', type=str,
    help="Whether train or eval."
)
parser.add_argument(
    '--eval_split', default='valid', type=str,
    help="Which split to validate on"
)
parser.add_argument(
    '--dataset_name', default='wikidata_big', type=str,
    help="Which dataset."
)
parser.add_argument(
    '--pct_train', default='100', type=str,
    help="Percentage of training data to use."
)
parser.add_argument(
    '--combine_all_ents',default="None",choices=["add","mult","None"],
    help="In score combination, whether to consider all entities or not"
)
parser.add_argument(
    '--simple_time',default=1.0,type=float,help="sampling rate of simple_time ques"
)
parser.add_argument(
    '--before_after',default=1.0,type=float,help="sampling rate of before_after ques"
)
parser.add_argument(
    '--first_last',default=1.0,type=float,help="sampling rate of first_last ques"
)
parser.add_argument(
    '--time_join',default=1.0,type=float,help="sampling rate of time_join ques"
)
parser.add_argument(
    '--simple_entity',default=1.0,type=float,help="sampling rate of simple_ent ques"
)

###########################################################

parser.add_argument(
    '--decay',default=1,type=bool,help="0 = no grad decay"
)
parser.add_argument(
    '--num_bases',default=4,type=int,help="num_bases for trgcn"
)
parser.add_argument(
    '--attn_mode', default='normal', type=str,
    help="Whether normal, mean or max."
)
parser.add_argument(
    '--decay_step',default=10,type=int,help="Value of decay_step for optimizer decay"
)
parser.add_argument(
    '--decay_gamma',default=0.4,type=float,help="Value of decay_gamma for optimizer decay"
)
parser.add_argument(
    '--patience',default=10,type=int,help="Value of decay_step for optimizer decay"
)
parser.add_argument(
    '--no_gating',default=0,type=bool,help="0 = TwiRGCN gated based on answer being entity or time"
)

args = parser.parse_args()


################          End           ###################
###########################################################


###########################################################
##############           Eval            ##################

def append_log_to_file(eval_log, epoch, filename):
    f = open(filename, 'a+')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()

def eval(args, qa_model, dataset, batch_size = 128, split='valid', k=10):
    num_workers = 2
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k # not change name in fn signature since named param used in places
    # k_list = [1, 3, 10]
    # k_list = [1, 5]
    k_list = [1, 10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,)
    topk_answers = []
    topk_scores = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        # todo: is there a more elegant way?
        if i_batch * batch_size == len(dataset.data):
            break
        answers_single = a.answers_single # last one assumed to be target
        
        if args.model == 'trgcn_simple_cat_select_sep':
            prob, scores = qa_model.forward(a)
        else:
            scores = qa_model.forward(a)
            
        for s in scores:
            # pred = dataset.getAnswersFromScores(s, k=max_k)
            pred_scores, pred = dataset.getAnswersFromScoresWithScores(s, k=max_k)
            topk_answers.append(pred)
            topk_scores.append(pred_scores.detach().cpu().numpy())
        loss = qa_model.loss(scores, answers_single.cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    # do eval for each k in k_list
    # want multiple hit@k
    eval_accuracy_for_reporting = 0
    model_outputs_for_saving = []
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)
        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['type']
            if 'simple' in question_type:
                simple_complex_type = 'simple'
            else:
                simple_complex_type = 'complex'
            entity_time_type = question['answer_type']
            # question_type = question['template']
            predicted = topk_answers[i][:k]
            
            # mark no answer ques as correct like in EXAQT
            if len(set(actual_answers).intersection(set(predicted))) > 0 or len(actual_answers)==0:
            # if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0
            if k == 1:
                item = {}
                item['id'] = question['uniq_id']
                item['is_correct'] = val_to_append
                item['question'] = question['paraphrases'][0]
                item['topk_pred'] = topk_answers[i]
                item['topk_scores'] = [round(x,3) for x in topk_scores[i].tolist()]
                item['true_answers'] = list(actual_answers)
                model_outputs_for_saving.append(item)
            for qtype in question_type: # can be multiple types
                question_types_count[qtype].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1
        # model_out_fname = 'model_outputs_trgcn_simple_cat_10_4_timequestions.json'
        
        # Saving test outputs to json for qualitative analysis
        if args.load_from != '':
            if args.eval_split == 'test':
                print("Saving test outputs to json for qualitative analysis")
                model_out_fname = 'outputs_from_model/{dataset_name}/outputs_{model_file}_{dataset_name}.json'.format(
                    dataset_name = args.dataset_name,
                    model_file = args.load_from
                )
                with open(model_out_fname, "w") as outfile:
                    json.dump(model_outputs_for_saving, outfile, indent = 2)
                    
        eval_accuracy = hits_at_k/total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))


        question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
        simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
        entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
        # for dictionary in [question_types_count]:
        for dictionary in [question_types_count, simple_complex_count, entity_time_count]:
        # for dictionary in [simple_complex_count, entity_time_count]:
            for key, value in dictionary.items():
                hits_at_k = sum(value)/len(value)
                s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
                    q_type = key,
                    hits_at_k = round(hits_at_k, 3),
                    num_questions = len(value)
                )
                if print_numbers_only:
                    s = str(round(hits_at_k, 3))
                eval_log.append(s)
            eval_log.append('')

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log, total_loss

################          End           ###################
###########################################################


###########################################################
##############         Training          ##################

def train(qa_model, dataset, valid_dataset, args,result_filename=None):
    num_workers = 0
    batch_size = args.batch_size
    learning_rate = args.lr * math.sqrt(batch_size)
    
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    
    if args.decay == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    max_eval_score = 0
    if args.save_to == '':
        args.save_to = 'temp'
    if result_filename is None:
        result_filename = 'results/{dataset_name}/{model_file}.log'.format(
            dataset_name = args.dataset_name,
            model_file = args.save_to
        )
    checkpoint_file_name = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
            dataset_name = args.dataset_name,
            model_file = args.save_to
        )
    
    early_checkpoint_file_name = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name = args.dataset_name,
        model_file = args.save_to + "_EARLY"
    )

    # if not loading from any previous file
    # we want to make new log file
    # also log the config ie. args to the file
    if args.load_from == '':
        print('Creating new log file')
        f = open(result_filename, 'a+')
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('Log time: %s\n' % dt_string)
        f.write('Config: \n')
        for key, value in vars(args).items():
            key = str(key)
            value = str(value)
            f.write('%s:\t%s\n' % (key, value))
        f.write('\n')
        f.close()


    max_eval_score = 0
    early_stopping = EarlyStopping(patience=args.patience, verbose=True) #, delta=30)       # Early Stopping

    print('Starting training')
    for epoch in range(args.max_epochs):
        if args.decay == 1:
            scheduler.step()        
        # qa_model.train()
        epoch_loss = 0
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0

        for i_batch, a in enumerate(loader):
            qa_model.zero_grad()
            qa_model.train()
            answers_single = a.answers_single 
            scores = qa_model.forward(a)
            
            loss = qa_model.loss(scores, answers_single.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)), Epoch=epoch) #*batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, args.max_epochs))
            loader.update()
        
        print('Epoch loss = ', epoch_loss)
        if (epoch + 1) % args.valid_freq == 0:
            print('Starting eval')
            eval_score, eval_log, ev_loss = eval(args,qa_model, valid_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k = args.eval_k)
            if eval_score > max_eval_score:
                print('Valid score increased') 
                save_model(qa_model, checkpoint_file_name)
                max_eval_score = eval_score
            # log each time, not max
            # can interpret max score from logs later
            append_log_to_file(eval_log, epoch, result_filename)
            
            # early_stopping(eval_score * 10000)
            early_stopping(ev_loss)  
                        
            if early_stopping.early_stop:
                print("Early stopping!")
                save_model(qa_model, early_checkpoint_file_name)
                print("Early Model Saved")
                break
 
 
###########################################################


def save_model(qa_model, filename):
    print('Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to ', filename)
    return
  
            
################          End           ###################
###########################################################

###########################################################
############         Args Processing         ##############


if 'complex' in args.tkbc_model_file and 'tcomplex' not in args.tkbc_model_file: #TODO this is a hack
    tkbc_model = loadTkbcModel_complex('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name = args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))
else:
    tkbc_model = loadTkbcModel('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name = args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))

if args.mode == 'test_kge':
    utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
    exit(0)

train_split = 'train'
if args.pct_train != '100':
    train_split = 'train_' + args.pct_train + 'pct'



###########################################################

dataset = QA_Dataset_TimeQuestions(split=train_split, dataset_name=args.dataset_name)
valid_dataset = QA_Dataset_TimeQuestions(split=args.eval_split, dataset_name=args.dataset_name)
test_dataset = QA_Dataset_TimeQuestions(split="test", dataset_name=args.dataset_name)
       
if args.model == 'twirgcn':
    qa_model = TwiRGCN(tkbc_model, args)
else:
    print('Model %s not implemented!' % args.model)
    exit(0)

print('Model is', args.model)

###########################################################


if args.load_from != '':
    filename = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.load_from
    )
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from ', filename)
else:
    print('Not loading from checkpoint. Starting fresh!')

qa_model = qa_model.cuda()

# For Eval
if args.mode == 'eval':
    score, log, e_loss = eval(args, qa_model, valid_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k = args.eval_k)
    exit(0)

result_filename = 'results/{dataset_name}/{model_file}.log'.format(
    dataset_name=args.dataset_name,
    model_file=args.save_to
)

################          End           ###################
###########################################################



###########################################################
#################      Run Train      #####################

train(qa_model, dataset, valid_dataset, args,result_filename=None)

print('Training finished')


################          End           ###################
###########################################################   
###########################################################
