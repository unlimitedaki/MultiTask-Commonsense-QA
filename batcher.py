# -*- coding: utf-8 -*-
import json
import re
import pdb
import random
from transformers import XLNetTokenizer
import torch
import xml.dom.minidom
from xml.dom.minidom import parse
from processor import CSQAProcessor,CosmosQAProcessor,MCScriptProcessor
from processor import convert_examples_to_features,InputFeatures
from torch.utils.data import Dataset, DataLoader, BatchSampler
from enum import IntEnum
from task_def import TaskType,TaskName,TaskID

# TaskType={
#     "classification":0
# }

class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Seqence = 5
    MLM = 6
# TaskName = ["csqa","mcscript2","cosmosqa"]
TaskProcessor = [CSQAProcessor,MCScriptProcessor,CosmosQAProcessor]

# dataset class for each single dataset
class SingleTaskDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        path,
        is_training,# for train and dev data
        task_id,
        is_pair,
        task_type,
        batch_size,
        max_seq_length):
    
        processor = TaskProcessor[task_id]()
        examples = processor.read_examples(path)
        if task_id == TaskID['csqa']:
            self._data = processor.convert_examples_to_features(examples,tokenizer,max_seq_length,is_training)
            print("special convert method for csqa")
        else:
            self._data = convert_examples_to_features(examples,tokenizer,max_seq_length,is_training)
        self._path = path
        self._task_id = task_id
        self._task_type = task_type
        self._batch_size = batch_size
        self._is_pair = is_pair
        # self._data = None
    
    # def load_data(self,path):
    #     examples = processor.read_examples(path)
    #     if task_id == TaskID['csqa']:
    #         self._data = processor.convert_examples_to_features(examples,tokenizer,max_seq_length,is_training)
    #         print("special convert method for csqa")
    #     else:
    #         self._data = convert_examples_to_features(examples,tokenizer,max_seq_length,is_training)

    def __len__(self):
        return len(self._data)

    def get_task_id(self):
        return self._task_id
    
    def get_batch_size(self):
        return self._batch_size
        
    def get_task_type(self):
        return self._task_type
    
    def __getitem__(self,idx): # return task infomation and sample
        return {"task":{"task_id":self._task_id,"task_type":self._task_type,"is_pair":self._is_pair},"sample":self._data[idx]}


# dataset class for multi task , rewrite __getitem__ for compliex index
class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]

# sampler, randomly spilt each dataset
class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, mix_opt, extra_task_ratio):
        self._datasets = datasets
        # self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), dataset.get_batch_size()))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i+batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list, self._mix_opt, self._extra_task_ratio)
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            # pdb.set_trace()
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices

class Collater:
    def __init__(self, 
                 is_train=True,
                 dropout_w=0.005,
                 soft_label=False):
                #  encoder_type=EncoderModelType.BERT):
        self.is_train = is_train
        self.dropout_w = dropout_w
        self.soft_label_on = soft_label
        # self.encoder_type = encoder_type
        self.pairwise_size = 1

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    @staticmethod
    def patch_data(gpu, batch_info, batch_data):
        if gpu:
            for i, part in enumerate(batch_data):
                if isinstance(part, torch.Tensor):
                    batch_data[i] = part.pin_memory().cuda(non_blocking=True)
                elif isinstance(part, tuple):
                    batch_data[i] = tuple(sub_part.pin_memory().cuda(non_blocking=True) for sub_part in part)
                elif isinstance(part, list):
                    batch_data[i] = [sub_part.pin_memory().cuda(non_blocking=True) for sub_part in part]
                else:
                    raise TypeError("unknown batch data type at %s: %s" % (i, part))
                    
            if "soft_label" in batch_info:
                batch_info["soft_label"] = batch_info["soft_label"].pin_memory().cuda(non_blocking=True)

        return batch_info, batch_data

    def rebatch(self, batch):
        newbatch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(0, size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['ruid'][idx]
                olab = sample['olabel'][idx]
                newbatch.append({'uid': uid, 'token_id': token_id, 'type_id': type_id, 'label':sample['label'], 'true_label': olab})
        return newbatch

    def __if_pair__(self, data_type):
        return data_type in [DataFormat.PremiseAndOneHypothesis, DataFormat.PremiseAndMultiHypothesis]


    def collate_fn(self, batch):
        task_id = batch[0]["task"]["task_id"]
        task_type = batch[0]["task"]["task_type"]
        is_pair = batch[0]["task"]["is_pair"] # premise and hypothesis pair
        # data_type = batch[0]["task"]["data_type"]
        new_batch = []
        for sample in batch:
            assert sample["task"]["task_id"] == task_id
            assert sample["task"]["task_type"] == task_type
            # assert sample["task"]["data_type"] == data_type
            new_batch.append(sample["sample"])
        batch = new_batch

        batch_info, batch_data = self._prepare_model_input(batch)
        batch_info['task_id'] = task_id  # used for select correct decoding head
        batch_info['input_len'] = len(batch_data)  # used to select model inputs
        batch_info['task_type'] = task_type
        
        return batch_info, batch_data

    def _get_max_len(self, batch, key='token_id'):
        tok_len = max(len(x[key]) for x in batch)
        return tok_len

    def _get_batch_size(self, batch):
        return len(batch)

    def _prepare_model_input(self, batch):
        batch_size = self._get_batch_size(batch)
        # tok_len = self._get_max_len(batch, key='token_id')
        batch_info = {} # useless now 
        #tok_len = max(len(x['token_id']) for x in batch)
        # pdb.set_trace()
        token_ids = torch.tensor( [item.select_field("input_ids") for item in batch ] ,dtype=torch.long)
        segment_ids = torch.tensor(  [item.select_field("segment_ids") for item in batch ],dtype=torch.long)
        input_mask = torch.tensor([item.select_field("input_mask") for item in batch ],dtype=torch.long)
        labels = torch.tensor([item.label for item in batch],dtype=torch.long)
        # if is_pair: 已修改，无用
        #     import pdb
        #     pdb.set_trace()
        #     premise_mask = torch.tensor( [item.select_field("premise_mask") for item in batch ] ,dtype=torch.long)
        #     hypothesis_mask =torch.tensor( [item.select_field("hypothesis_mask") for item in batch ] ,dtype=torch.long)
        #     batch_data = [token_ids,segment_ids,input_mask,premise_mask,hypothesis_mask,labels]
        # else:
        batch_data = [token_ids,segment_ids,input_mask,labels]
        return batch_info, batch_data

# test  code
# tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
# csqa_dataset = SingleTaskDataset(
#     tokenizer = tokenizer,
#     path = "datasets/CSQA/train_rand_split.jsonl",
#     is_training=True,
#     task_id = 0,
#     task_type = 0,
#     max_seq_length = 80,
#     batch_size = 24)
# mcscript_dataset = SingleTaskDataset(
#     tokenizer = tokenizer,
#     path = "datasets/MCScript2.0/train-data.xml",
#     is_training=True,
#     task_id = 1,
#     task_type = 0,
#     max_seq_length = 128,
#     batch_size = 10)
# cosmos_dataset = SingleTaskDataset(
#     tokenizer = tokenizer,
#     path = "datasets/cosmosqa-data/train.jsonl",
#     is_training=True,
#     task_id = 2,
#     task_type = 0,
#     max_seq_length = 256,
#     batch_size = 5)

# multi_task_Dataset = MultiTaskDataset([csqa_dataset,mcscript_dataset,cosmos_dataset])

# batch_size = 5
# multi_task_batch_sampler = MultiTaskBatchSampler([csqa_dataset,mcscript_dataset,cosmos_dataset], batch_size, 0, 0)

# train_collater = Collater(dropout_w= 0 )
# multi_task_train_data = DataLoader(multi_task_Dataset, batch_sampler=multi_task_batch_sampler, collate_fn=train_collater.collate_fn, pin_memory=torch.cuda.is_available())
# for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):
#     # pdb.set_trace()
#     print(batch_data[0].shape)
#     # batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)

