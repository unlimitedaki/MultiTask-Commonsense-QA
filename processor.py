# -*- coding: utf-8 -*-
import json
import re
import pdb
from transformers import XLNetTokenizer
import xml.dom.minidom
from xml.dom.minidom import parse

BLANK_STR = "___"
class MultipleChoiceExample(object): # examples for all kind of dataset s
    # including 
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 id,
                 context,
                 question,
                 endings,
                 label=None):
        self.id = id,   
        self.context = context
        self.question = question
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 id,
                 features,
                 label

    ):
        self.id = id
        # if len(features)>3: # have premise and hypothesis for SAN
        # self.features = [
        #     {
        #         'input_ids': input_ids,
        #         'input_mask': input_mask,
        #         'segment_ids': segment_ids,
        #         'premise_mask':premise_mask,
        #         'hypothesis_mask':hypothesis_mask
        #     }
        #     for input_ids, input_mask, segment_ids, premise_mask, hypothesis_mask in features
        # ]
        # else:
        self.features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for input_ids, segment_ids, input_mask in features
        ]
        
        self.label = label

    def select_field(self, field):
        return [
            item[field] for item in self.features
        ]




def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class CSQAProcessor():

    # Identify the wh-word in the question and replace with a blank
    def replace_wh_word_with_blank(self,question_str: str):
        if "What is the name of the government building that houses the U.S. Congress?" in question_str:
            print()
        question_str = question_str.replace("What's", "What is")
        question_str = question_str.replace("whats", "what")
        question_str = question_str.replace("U.S.", "US")
        wh_word_offset_matches = []
        wh_words = ["which", "what", "where", "when", "how", "who", "why"]
        for wh in wh_words:
            # Some Turk-authored SciQ questions end with wh-word
            # E.g. The passing of traits from parents to offspring is done through what?

            if wh == "who" and "people who" in question_str:
                continue

            m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches = [(wh, m.start())]
                break
            else:
                # Otherwise, find the wh-word in the last sentence
                m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
                if m:
                    wh_word_offset_matches.append((wh, m.start()))
                # else:
                #     wh_word_offset_matches.append((wh, question_str.index(wh)))

        # If a wh-word is found
        if len(wh_word_offset_matches):
            # Pick the first wh-word as the word to be replaced with BLANK
            # E.g. Which is most likely needed when describing the change in position of an object?
            wh_word_offset_matches.sort(key=lambda x: x[1])
            wh_word_found = wh_word_offset_matches[0][0]
            wh_word_start_offset = wh_word_offset_matches[0][1]
            # Replace the last question mark with period.
            question_str = re.sub("\?$", ".", question_str.strip())
            # Introduce the blank in place of the wh-word
            fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                            question_str[wh_word_start_offset + len(wh_word_found):])
            # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
            # question. E.g. "Which of the following force ..." -> "___ force ..."
            final = fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
            final = final.replace(BLANK_STR + " of these", BLANK_STR)
            return final

        elif " them called?" in question_str:
            return question_str.replace(" them called?", " " + BLANK_STR+".")
        elif " meaning he was not?" in question_str:
            return question_str.replace(" meaning he was not?", " he was not " + BLANK_STR+".")
        elif " one of these?" in question_str:
            return question_str.replace(" one of these?", " " + BLANK_STR+".")
        elif re.match(".*[^\.\?] *$", question_str):
            # If no wh-word is found and the question ends without a period/question, introduce a
            # blank at the end. e.g. The gravitational force exerted by an object depends on its
            return question_str + " " + BLANK_STR
        else:
            # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
            # e.g. Virtually every task performed by living organisms requires this?
            return re.sub(" this[ \?]", " ___ ", question_str)
    # the original one



    def read_examples(self,input_file, have_answer=True):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                csqa_json = json.loads(line)
                if have_answer:
                    label = ord(csqa_json["answerKey"]) - ord("A")
                else:
                    label = 0  # just as placeholder here for the test data
                examples.append(
                    MultipleChoiceExample(
                        id = csqa_json["id"], 
                        question=csqa_json["question"]["stem"],
                        context = "",
                        endings  = [csqa_json["question"]["choices"][i]["text"]  for i in range(5) ],
                        label = label
                    ))
        return examples


    def get_fitb_from_question(self,question_text: str) -> str:
        fitb = self.replace_wh_word_with_blank(question_text)
        if not re.match(".*_+.*", fitb):
            # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
            # Strip space, period and question mark at the end of the question and add a blank
            fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + " "+ BLANK_STR
        return fitb


    # Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
    def create_hypothesis(self,fitb: str, choice: str) -> str:
        if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
            choice = choice[0].upper() + choice[1:]
        else:
            choice = choice.lower()
        # Remove period from the answer choice, if the question doesn't end with the blank
        if not fitb.endswith(BLANK_STR):
            choice = choice.rstrip(".")
        # Some questions already have blanks indicated with 2+ underscores
        hypothesis = re.sub("__+", choice, fitb)
        return hypothesis

    #  convert csqa examples(MultipleChoiceExamples) to InputFeatures
    def convert_examples_to_features(self,examples, tokenizer, max_seq_length,
                                    is_training):
        """Loads a data file into a list of `InputBatch`s."""

        # CSQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # - [CLS] context [SEP] choice_5 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        features = []
        for example_index, example in enumerate(examples):

            # start_ending_tokens = tokenizer.tokenize(example.start_ending)

            choices_features = []
            for ending_index, ending in enumerate(example.endings):
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens

                statement = self.create_hypothesis(self.get_fitb_from_question(example.question), ending)

                statement = example.question

                context_tokens = tokenizer.tokenize(statement)
                context_tokens_choice = context_tokens[:]

                ending_tokens = tokenizer.tokenize(ending)
                _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
                # pdb.set_trace()
                tokens = context_tokens_choice + ["<sep>"] + ending_tokens + ["<sep>"]+["<cls>"] 
                # segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)
                inputs = tokenizer.encode_plus(
                    statement,
                    ending,
                    add_special_tokens=True,
                    max_length = max_seq_length
                )
                input_ids,segment_ids,input_mask = inputs['input_ids'],inputs['token_type_ids'],inputs['attention_mask']
                # input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                padding_segment = [4] * (max_seq_length - len(input_ids))# xlnetâ€˜s segment padding id is 4!

                input_ids = padding + input_ids
                input_mask = padding + input_mask
                segment_ids = padding_segment  + segment_ids
                # pdb.set_trace()

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((input_ids, segment_ids,input_mask))

            label = example.label

            features.append(
                InputFeatures(
                    id = example.id,
                    features = choices_features,
                    label = label
                )
            )

        return features

class MCScriptProcessor():
    def read_examples(self,input_file,have_answer=True):
        DomTree = xml.dom.minidom.parse(input_file)
        collection = DomTree.documentElement
        # each instance has a text(context),several questions, each have 2 answers
        instances = collection.getElementsByTagName("instance")
        examples = []
        for instance in instances:
            instance_id = instance.getAttribute("id")
            context = instance.getElementsByTagName("text")[0].childNodes[0].data
            questions = instance.getElementsByTagName("questions")[0].getElementsByTagName("question")
            for question in questions:
                question_text = question.getAttribute("text")
                endings = []
        
                for answer in question.getElementsByTagName("answer"):
                    endings.append(answer.getAttribute("text"))
                if question.getElementsByTagName("answer")[0].getAttribute("correct") == "True":
                    label = 0
                else:
                    label = 1
                # pdb.set_trace()
                examples.append(
                    MultipleChoiceExample(
                        id = instance_id+"-"+question.getAttribute("id"),
                        question = question_text,
                        context = context,
                        endings  = endings,
                        label = label
                    ))
        return examples        

class CosmosQAProcessor():
    def read_examples(self,input_file,have_answer=True):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                example_json = json.loads(line)
                # pdb.set_trace()
                if have_answer:
                    label = ord(example_json["label"]) - ord("0")
                else:
                    label = 0  # just as placeholder here for the test data
                examples.append(
                    MultipleChoiceExample(
                        id = example_json["id"], 
                        question=example_json["question"],
                        context = example_json["context"],
                        endings  = [example_json["answer"+str(i)]  for i in range(4) ],
                        label = label
                    ))
        return examples
    
def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training=True,is_pair = False):
    features = []
    for example_index,example in enumerate(examples):
        context = example.context
        question = example.question
        context_token = tokenizer.tokenize(context)
        question_token = tokenizer.tokenize(question)
        choices_features = []
        for ending_index,ending in enumerate(example.endings):
            ending_token = tokenizer.tokenize(ending)
            if context != "":
                seg1_tokens = context_token+["<sep>"]+question_token
            else:
                seg1_tokens = question_token
            inputs = tokenizer.encode_plus(
                seg1_tokens,
                ending_token,
                add_special_tokens=True,
                max_length = max_seq_length,
                pad_to_max_length=True,
                )
            input_ids,segment_ids,input_mask = inputs['input_ids'],inputs['token_type_ids'],inputs['attention_mask']
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # if  is_pair: # 废弃，以后删，找到了更好的方法
            #     premise_len = len(segment_ids)-sum(segment_ids)# 统计第一句，也就是0的长度
            #     premise_mask = input_mask[:premise_len]+[0]*(sum(segment_ids))
            #     hypothesis_mask = segment_ids

            choices_features.append((input_ids,segment_ids,input_mask))
        label = example.label
        # pdb.set_trace()
        features.append(
            InputFeatures(
                id = example.id,
                features = choices_features,
                label = label
            )
        )
    return features

