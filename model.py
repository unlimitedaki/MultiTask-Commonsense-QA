from module.san import SANClassifier
from module.dropout_wrapper import DropoutWrapper
from transformers import XLNetForMultipleChoice,XLNetConfig,XLNetModel,BertModel
from transformers.modeling_utils import SequenceSummary # for MultipleChoice
from transformers import XLNetTokenizer,BartTokenizer
import os
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from task_def import TaskType,TaskName

def select_pretrained(model_name,cache_dir):
    cache = os.path.join(cache_dir,model_name)
    if 'bert' in model_name:
        return BertModel.from_pretrained(model_name,cache_dir = cache)
    elif 'xlnet' in model_name:
        return XLNetModel.from_pretrained(model_name,cache_dir = cache)
    else:
        return None


class Classification(nn.Module):
    def __init__(self,config):
        super(Classification,self).__init__()
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model,1)
    def forward(self,batch_data,encoder_outputs):
        input  = encoder_outputs[0]
        num_choices = batch_data[0].shape[1]
        labels = batch_data[3]
        seq_sum = self.sequence_summary(input)
        logits = self.logits_proj(seq_sum)
        reshaped_logits = logits.view(-1, num_choices)
        # outputs = reshaped_logits
        loss_fct = CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(reshaped_logits, labels.view(-1))
            outputs = (loss,reshaped_logits)
        return outputs

Decoders = [Classification]

class MultiTaskModel(nn.Module):
    def __init__(self,model_name,cache_dir,task_list):
        super(MultiTaskModel,self).__init__()
        cache = os.path.join(cache_dir,model_name)
        self.transformer = XLNetModel.from_pretrained(model_name,cache_dir = cache)
        self.transformer_config = self.transformer.config
        self.dropout = DropoutWrapper(self.transformer_config.dropout)
        self.decoderID = {} #模型内部的task_id与decoder_id的映射
        # self.decoder = {}
        self.decoder_list = nn.ModuleList()
        for innerid,task in enumerate(task_list):
            if task[1] == TaskType["classification"]:# task[1] = tasktype 
                classifier = Classification(self.transformer_config)
                # classifier = Classification(self.transformer_config)
                print("use simple classification")
                self.decoder_list.append(classifier) 
            elif task[1] == TaskType["SANclassification"]:
                classifier = SANClassifier(self.transformer_config.hidden_size,self.transformer_config.hidden_size,label_size = 1,dropout = self.dropout)
                print("use SANClassifier")
                self.decoder_list.append(classifier) 
            else:
                pass
            self.decoderID[task[0]] = innerid
            # self.decoders[task] = decoder
        # self.init_weights()
            
    def forward(self,batch_meta,batch_data):
        task = batch_meta['task_type']
        task_id = batch_meta['task_id']
        input_ids = batch_data[0]
        token_type_ids = batch_data[1]
        attention_mask = batch_data[2]
        labels = batch_data[3]
        # shared encoder 
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
        )
        # decoder
        if task == TaskType["SANclassification"]:
            num_choices = input_ids.shape[1]
            # 直接将 attention_mask 和 token_type_ids 改为 premise_mask 和 hypothesis_mask 来节省显存
            for i,atten in enumerate(flat_attention_mask):
                # for xlnet left_padding
                seq_len = len(atten)
                padding_len = seq_len-sum(atten) # attention 中0的个数为padding长度
                flat_token_type_ids[i][:padding_len] = torch.ByteTensor([0]*padding_len)# xlnet的toke_type_ids的padding为3，需要改成1来计算
                flat_token_type_ids[i][-1:] = torch.ByteTensor([0]*1)# xlnet的cls的token_type_id = 2
                premise_len = seq_len-sum(flat_token_type_ids[i])-1
                flat_attention_mask[i][premise_len:] = torch.ByteTensor([0]*(seq_len-premise_len))
            flat_token_type_ids = flat_token_type_ids.bool()
            flat_attention_mask = flat_attention_mask.bool()
            
            outputs = self.decoder_list[self.decoderID[task_id]](transformer_outputs[0],transformer_outputs[0],flat_attention_mask,flat_token_type_ids)
            reshaped_logits = outputs.view(-1, num_choices)
            loss_fct = CrossEntropyLoss()
            if labels is not None:
                loss = loss_fct(reshaped_logits, labels.view(-1))
                outputs = (loss,reshaped_logits)
        elif task == TaskType["classification"]:
            outputs = self.decoder_list[self.decoderID[task_id]](batch_data,transformer_outputs)
        else:
            raise NotImplementedError
        return outputs

