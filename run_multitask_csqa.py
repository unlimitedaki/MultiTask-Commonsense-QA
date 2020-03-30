import argparse
import csv
import json
import logging
import os
import random
import sys 
if os.path.exists("/home/aistudio/work/external-libraries"):
    sys.path.append('/home/aistudio/work/external-libraries')
from io import open
import re
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

from tqdm import tqdm, trange


from transformers import XLNetForMultipleChoice,XLNetConfig,XLNetModel,BertModel
from transformers.modeling_utils import SequenceSummary # for MultipleChoice
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import XLNetTokenizer,BertTokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import pdb
from model import MultiTaskModel
from batcher import SingleTaskDataset,MultiTaskDataset,MultiTaskBatchSampler,Collater
from task_def import TaskType,TaskName
# import loss


BLANK_STR = "___"


def select_tokenizer(model_name):
    if 'bert' in model_name:
        return BertTokenizer.from_pretrained(model_name)
    elif 'xlnet' in model_name:
        return XLNetTokenizer.from_pretrained(model_name)
    else:
        return None

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(model,dev_dataloaders):
    res = []
    total_acc =0.0
    total_step = 0.0
    for task_id,task_type,dataloader in dev_dataloaders:
        total_loss = 0.0
        eval_examples = 0
        eval_steps = 0
        tmp_eval_accuracy = 0.0
        batch_meta = {"task_id":task_id,"task_type":task_type}
        for step,batch_data in enumerate(tqdm(dataloader,desc = "evaluate "+str(TaskName[task_id]))):
            batch_data = [item.cuda() for item in batch_data]
            loss,logits = model(batch_meta,batch_data)

            total_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            labels = batch_data[3].to('cpu').numpy()
            tmp_eval_accuracy += accuracy(logits, labels)
            eval_steps += 1
            eval_examples += batch_data[0].size(0)
        eval_loss = total_loss/eval_steps
        eval_accuracy = tmp_eval_accuracy / eval_examples
        total_acc += tmp_eval_accuracy
        total_step += eval_examples
        res.append((task_id,eval_loss,eval_accuracy))
    return res,total_acc/total_step

def train(args):
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logfilename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" "+args.save_model_name+".log.txt"
    fh = logging.FileHandler(os.path.join(output_dir,logfilename), mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)


    # load data
    logger.info("****** Loading Datasets ******")
    tokenizer = select_tokenizer(args.model)
    dataset_names = args.datasets.split(" ")
    with open(args.datasets_config) as f:
        datasets_config = json.load(f)
    train_datasets = []
    dev_datasets = []
    task_list = []
    for task_id,dataset in enumerate(dataset_names) :# 按照顺序分配task_id
        config = datasets_config[dataset]
        if args.do_preprocess:
            single_train_dataset = SingleTaskDataset(
              tokenizer = tokenizer,
              path = config["train_dir"],
              is_training = True,
              task_id = config['task_id'],
              is_pair = config["is_pair"],
              task_type = TaskType[config["task_type"]],
              batch_size= config["train_batch_size"],
              max_seq_length = config["max_seq_length"]
            )
            
            single_dev_dataset = SingleTaskDataset(
              tokenizer = tokenizer,
              path = config["dev_dir"],
              is_training = True,
              task_id = config['task_id'],
              is_pair = config["is_pair"],
              task_type = TaskType[config["task_type"]],
              batch_size= config["dev_batch_size"],
              max_seq_length = config["max_seq_length"]
            )
            # single_train_dataset.load_data()
            # single_dev_dataset.load_data
            torch.save(single_train_dataset,os.path.join(args.dataset_features_dir,dataset+".train"))
            torch.save(single_dev_dataset,os.path.join(args.dataset_features_dir,dataset+".dev"))
        else:
            single_train_dataset = torch.load(os.path.join(args.dataset_features_dir,dataset+".train"))
            single_dev_dataset = torch.load(os.path.join(args.dataset_features_dir,dataset+".dev"))
    # task_list.append(TaskType[config["task_type"]])
        task_list.append((config["task_id"],TaskType[config["task_type"]]))
        train_datasets.append(single_train_dataset)
        dev_datasets.append(single_dev_dataset)
    train_collater = Collater(dropout_w= args.collater_dropout)
    multi_task_datasets = MultiTaskDataset(train_datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets,mix_opt=0,extra_task_ratio=0)
    train_dataloader = DataLoader(multi_task_datasets,batch_sampler=multi_task_batch_sampler, collate_fn=train_collater.collate_fn, pin_memory=torch.cuda.is_available())

    dev_dataloaders = []
    for dataset in dev_datasets:
        all_input_ids = torch.tensor( [item['sample'].select_field("input_ids") for item in dataset ] ,dtype=torch.long)
        all_token_type_ids = torch.ByteTensor([item['sample'].select_field("segment_ids") for item in dataset ])
        all_attention_mask = torch.ByteTensor([item['sample'].select_field("input_mask") for item in dataset ])
        all_labels = torch.tensor([item['sample'].label for item in dataset],dtype=torch.long)
        dev_dataset = TensorDataset(all_input_ids, all_token_type_ids,all_attention_mask,  all_labels)
        sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset,sampler = sampler, batch_size=dataset.get_batch_size())
        dev_dataloaders.append((dataset.get_task_id(),dataset.get_task_type(),dev_dataloader))





    # prepare model
    status = {}
    tasklist = list(set(task_list))
    if args.do_finetune:
        model_dir = os.path.join(args.output_dir,args.save_model_name)
        status = json.load(open(os.path.join(model_dir,'status.json')))
        epoch =  status["current_epoch"]
        current_model = os.path.join(model_dir, 'checkpoint-{}.model'.format(epoch))
        # model = XLNetForMultipleChoice.from_pretrained(current_model)
        model = MultiTaskModel(args.model,cache_dir=args.cache_dir,task_list=task_list)
        model.load_state_dict(torch.load(current_model))
        model.cuda()
        # dev_result,dev_accuracy= evaluate(model,dev_dataloaders)
        # for res in dev_result:
        #     logger.info("%s dev loss: %s, dev acc: %s",TaskName[res[0]],str(res[1]),str(res[2]))
        # logger.info("total dev acc: %s",str(dev_accuracy))
    # elif args.do_finetune_best:
    #     return 
    else :
        # first time
        model = MultiTaskModel(args.model,cache_dir=args.cache_dir,task_list=task_list)
        status["best_epoch"] = 0
        status["current_epoch"] = 0
        status['best_dev_accuracy'] = 0
        model.cuda()

    num_train_optimization_steps = len(train_dataloader)*args.num_train_epochs
    optimizer = AdamW(model.parameters(), eps =args.eps, lr=args.learning_rate, correct_bias=False)
    if args.do_finetune:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps) 
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_train_optimization_steps) 
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=num_train_optimization_steps) 

    # train
    logger.info("***** Running training *****")

    best_dev_accuracy = 0
    best_dev_epoch = 0
    no_up = 0
    global_step = 0

    model.cuda()
    epoch_tqdm = trange(status["current_epoch"],int(args.num_train_epochs), desc="Epoch")
    for epoch in epoch_tqdm:
        model.train()
        train_loss = 0.0
        avg_loss = 0.0

        for step, (batch_meta, batch_data) in enumerate(tqdm(train_dataloader,desc = "Iteration")):

            #
            batch_meta,batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
            # pdb.set_trace()
            batch_data[1] = batch_data[1].byte()
            batch_data[2] = batch_data[2].byte()
            
            batch_data = [item.cuda() for item in batch_data]
            # print([item.shape for item in batch_data])
            loss, logits = model(batch_meta,batch_data)
            # pdb.set_trace()
            loss.backward()

            train_loss += loss.item()

            global_step += 1
            if step% 500 == 0 or step == len(train_dataloader)-1:# print step average loss every 500 step
                    avg_loss = train_loss/(step+1)
                    logger.info("\t average_step_loss=%s @ step = %s on epoch = %s",str(avg_loss),str(global_step),str(epoch+1))

            optimizer.step()
            scheduler.step()  # Update scheduler
            model.zero_grad()

        # evalute
        dev_result,dev_accuracy= evaluate(model,dev_dataloaders)
        for res in dev_result:
            logger.info("%s dev loss: %s, dev acc: %s",TaskName[res[0]],str(res[1]),str(res[2]))
        logger.info("total dev acc: %s",str(dev_accuracy))

        if dev_accuracy > best_dev_accuracy:
            # New best model.
            status['best_dev_accuracy'] = dev_accuracy
            best_dev_accuracy = dev_accuracy
            best_dev_epoch = epoch + 1
            status["best_epoch"] = epoch + 1
            no_up = 0
        else:
            no_up += 1

        torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint-{}.model'.format(epoch + 1)))
        logger.info("\t epoch %s saved to %s",str(epoch+1),output_dir)

        status["current_epoch"] = epoch+1
        with open(os.path.join(output_dir,'status.json'),"w") as fs:
            json.dump(status,fs)
        if no_up >= args.patience:
            epoch_tqdm.close()
            break


# main
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--datasets",default = "csqa",type = str)
parser.add_argument("--datasets_config",default = "datasets.json",type = str,help = 'config of batch_size, max_length,dataset_path of each dataset')
parser.add_argument("--do_train", default = True,action='store_true',help="Whether to run training.")
parser.add_argument("--do_preprocess", default = False, action='store_true',help = "Whether to run preprocess")

parser.add_argument("--collater_dropout",default = 0,type = float,help = "I don't know what's this")
parser.add_argument("--learning_rate",default = 2e-5,type = float,help = "learning_rate")
parser.add_argument("--eps",default = 1e-8,type = float,help = "adam epsilon")
parser.add_argument("--num_warmup_steps",default = 1000,type = float,help = "num_warmup_steps")
parser.add_argument("--num_train_epochs",default = 10,type = int)
parser.add_argument("--save_model_name",default = "xlnet_base_cased_st_csqa_nosan",type = str)
parser.add_argument("--log_dir",default = "log",type=str)
parser.add_argument("--patience",default = 5,type=int)


parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
parser.add_argument("--model",default = "xlnet-base-cased",type = str,help="pretrained model")
parser.add_argument("--cache_dir",default = "model/cache",type=str)
parser.add_argument("--output_dir",default = "model",type =str)
parser.add_argument("--dataset_features_dir",default = "datasets/features")
parser.add_argument("--do_finetune",action="store_true",default = False,help = "Whether to finetune")
parser.add_argument("--do_finetune_best",action="store_true",default = False,help="Whether to finetune the best model")

# for notebook parser bug
args = parser.parse_args()
if args.do_train:
    train(args)


