'''Combine using >>>> ls . -v | xargs cat > combi.dev.rels'''

import argparse
import re
import torch
import csv
from spacytokenizer import SpacyTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device('cuda:5')


def remove_punkt_from_input(align_input):
    if isinstance(align_input, str):
        align_input = re.sub(r'[^\w\s]','',align_input)
    else:
        align_input = [re.sub(r'[^\w\s]','',s) for s in align_input]
        align_input = ' '.join(align_input).split()
        align_input = ' '.join(align_input)
    return align_input

class HFTranslator:

    '''
    ************Usage:***********
    src_text = 'this is a sentence in english that we want to translate to french',
    Hft = HFTranslator(src=src_lang, dest=target_lang, model_version='wmt2021', device=device)
    result = Hft.translate_sentences(src_txt) if version=='v1' else Hft.translate_combined_args_with_failback(src_txt)
    '''

    def __init__(self, src, dest, model_version, device):
        if model_version=='wmt2016':
            self.tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de").to(device)
        elif model_version=='wmt2021':
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x").to(device)
        self.src = src
        self.dest = dest
        self.device = device
        self.model_version = model_version
        assert src in ['en']
        assert dest in ['de']

    def translate_model_call(self, src_text):
        if self.model_version=='wmt2016':
            input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device).long()
            try:
                output_ids = self.model.generate(input_ids, max_new_tokens=1024)[0]
            except:
                print('skipping....')
                return '##########', input_ids.shape
            translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        elif self.model_version=='wmt2021':
            inputs = self.tokenizer(src_text, return_tensors="pt").to(self.device)
            generated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.get_lang_id(self.dest))
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    def translate_sentences(self, src_text):
        if isinstance(src_text, str):
            translation = self.translate_model_call(src_text)
        else:
            translation = []
            for sentence in src_text:
                translation_item = self.translate_model_call(sentence)
                translation.append(translation_item)
        return translation

    def strip_args(self, arg1, arg2):
        arg1 = arg1.strip('~').lstrip()
        arg2 = arg2.strip('~').lstrip()
        return arg1, arg2
    
    def translate_combined_args_with_failback(self, src_text, special_character_string):
        #within the limit translate the combined args otherwise split the args. if empty result is return then split the args.
        input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device).long()
        if input_ids.shape[1]>70:
            arg1, arg2 = src_text.split(special_character_string)
            arg1, arg2 = self.strip_args(arg1, arg2)
            arg1 = self.translate_sentences(arg1)
            arg2 = self.translate_sentences(arg2)
        else:   
            try:
                output_ids = self.model.generate(input_ids, max_new_tokens=1024)[0]
                translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                arg1, arg2 = self.split_arg_from_special_chars(translation)
            except:
                arg1, arg2 = src_text.split(special_character_string)
                arg1, arg2 = self.strip_args(arg1, arg2)
                arg1 = self.translate_sentences(arg1)
                arg2 = self.translate_sentences(arg2)
            if arg1=='' or arg2=='':
                arg1, arg2 = self.split_arg_from_special_chars(src_text)
                arg1, arg2 = self.strip_args(arg1, arg2)
                arg1 = self.translate_sentences(arg1)
                arg2 = self.translate_sentences(arg2)

        arg1, arg2 = self.strip_args(arg1, arg2)
        return arg1, arg2

    def split_arg_from_special_chars(self, arg3):
        #designed specifically for specialcharacter "~#~"
        arg1 = ''
        arg2 = ''
        flag = 0
        for c in arg3:
            if flag==2: arg2=arg2+c
            elif flag==0:
                if c=='~' or c=='#':
                    flag=1
                else:
                    arg1=arg1+c
            elif flag==1:
                if c=='~':
                    flag =2
        arg1 = arg1.strip('~').lstrip()
        arg2 = arg2.strip('~').lstrip()
        return arg1, arg2

def replace_special_chars(src_text, Stok):
    # no way to handle double escaping "" by csv module. manually remove
    src_text = [_.replace('&lt;*&gt;', '<*>').replace('&quot;', '" ').replace('&#39;', '’') for _ in src_text]
    src_text = [_.replace('&gt;', '>').replace('&lt;', '<') for _ in src_text]
    
    src_text = [Stok.spacify_punkts(_) for _ in src_text]
    src_text = [Stok.spacify_inverted_commas(_) for _ in src_text]
    src_text = ''.join(src_text)
    
    return src_text


def process_chunk(reader, target_csv, src_lang, target_lang, Hft, Stok, version):
    writer = csv.writer(open(target_csv, 'w'), delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')#, escapechar='\\')
    print("Translating chunk...")
    for count, line in enumerate(reader):
        print('Processing line ', count)
        # if count<10299:
        #     continue

        if version=='v1':
            print('translation 1...')
            arg1 = Hft.translate_sentences(replace_special_chars(line[3], Stok)).replace('\t', ' ')
            print('translation 2...')
            arg2 = Hft.translate_sentences(replace_special_chars(line[4], Stok)).replace('\t', ' ')
        elif version=='v2':
            print('translation...')
            arg1, arg2 = Hft.translate_combined_args_with_failback(line[3]+" ~#~ "+line[4], special_character_string="~#~")

        line[3] = arg1
        line[4] = arg2
        if line[7]==line[3]:
            line[7] = arg1
        else:
            line[7] = Hft.translate_sentences(replace_special_chars(line[7], Stok))

        if line[8]==line[4]:
            line[8] = arg2
        else:
            line[8] = Hft.translate_sentences(replace_special_chars(line[8], Stok))
        writer.writerow(line)
        print('line: ', line)
    print("Translated chunk")
    

def translate_disrpt_csv(src_csv, target_csv, src_lang, target_lang):
    print('Starting init')
    reader = csv.reader(open(src_csv), quoting=csv.QUOTE_NONE, delimiter='\t')
    Stok = SpacyTokenizer()
    Hft = HFTranslator(src=src_lang, dest=target_lang, model_version='wmt2021', device=device)
    print('Init complete')
    
    process_chunk(reader, target_csv, src_lang, target_lang, Hft=Hft, Stok=Stok, version='v2')

def unit_tests():
    # ### check if lenght is an issue: yes
    # def translate(arg1, arg2, Hft=Hft):
    #     print('E1: ', arg1)
    #     print('E2: ', arg2)
    #     arg1, arg2 = Hft.translate_combined_args_with_failback(arg1+" ~#~ "+arg2, special_character_string="~#~")

    #     print('D1: ', arg1)
    #     print('D2: ', arg2)
    #     print("----------------------------------------")

    # Hft = HFTranslator(src='en', dest='de', device=device)
    # translate("Ajinomoto Co. , a Tokyo-based food-processing concern , said net income in its first half rose 8.9 % to 8.2 billion yen <*> from 7.54 billion yen a year earlier .",
    #           "Ajinomoto predicted sales in the current fiscal year <*> of 480 billion yen ,")
    
    # translate("to decide", "whether he has the right .")

    # translate("Now the White House is declaring that he might not rely on Congress <*> to pass the line-item veto law decided in the House of California in the last few months of the year", 
    #           "White House spokesmen last week said <*> that the Constitution gives him the power , exercising a line-item veto and inviting a court challenge") #so it is an issue of length. we also tried shorter versions of the sentence and found an uppper vound on sentence length that can be successfully translated.
    # #when we translate when arguments are combined versus spearated; there is better accuracy when arguments are split
    
    # translate("Now the White House is declaring that he might not rely on Congress <*> to pass the line-item veto law decided in the House of California in the last few months of the year", 
    #           "White House spokesmen last week said <*> that the Constitution gives him the power , exercising a line-item veto and inviting a court challenge")


    # check if wmt2021 news dataset model is better: yes, even better than t5 for n=2
    # wmt2021
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
    inputs = tokenizer("it might n't be effective in cutting the deficit .", return_tensors="pt").to(device)
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("de"))
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]) # Es könnte nicht wirksam sein, das Defizit zu senken.
    inputs = tokenizer("And among the issues <*> were Detroit Edison Co. and Niagara Mohawk Power Corp.", return_tensors="pt").to(device)
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("de"))
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]) # Und unter den Problemen <*> waren Detroit Edison Co. und Niagara Mohawk Power Corp.

    # # wmt2016
    # tokenizer2 = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
    # model2 = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de").to(device)
    # input_ids = tokenizer2("it might n't be effective in cutting the deficit .", return_tensors="pt", add_special_tokens=False).input_ids.to(device).long()
    # output_ids = model2.generate(input_ids, max_new_tokens=1024)[0]
    # print(tokenizer2.decode(output_ids, skip_special_tokens=True))# The room was very small and the bed was very comfortable.
    # input_ids = tokenizer2("And among the issues <*> were Detroit Edison Co. and Niagara Mohawk Power Corp.", return_tensors="pt", add_special_tokens=False).input_ids.to(device).long()
    # output_ids = model2.generate(input_ids, max_new_tokens=1024)[0]
    # print(tokenizer2.decode(output_ids, skip_special_tokens=True))# und unter den Problemen waren detroit edison co.

    # #t5
    # from transformers import T5Tokenizer, T5ForConditionalGeneration
    # tokenizer3 = T5Tokenizer.from_pretrained("t5-base")
    # model3 = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
    # input_ids = tokenizer3("translate English to German: "+"it might n't be effective in cutting the deficit .", return_tensors="pt", add_special_tokens=False).input_ids
    # outputs = model3.generate(input_ids)
    # print(tokenizer3.decode(outputs[0], skip_special_tokens=True)) # es könnte nicht wirksam sein, das Defizit zu senken
    # input_ids = tokenizer3("translate English to German: "+"And among the issues <*> were Detroit Edison Co. and Niagara Mohawk Power Corp.", return_tensors="pt", add_special_tokens=False).input_ids
    # outputs = model3.generate(input_ids)
    # print(tokenizer3.decode(outputs[0], skip_special_tokens=True)) # Und unter den Themen *> waren Detroit Edison Co. und Niagara Mohawk Power Corp



if __name__ == "__main__":

    skip_chunk_list = [] 

    parser = argparse.ArgumentParser()

    # Raw Data
    parser.add_argument("--src_csv", type=str, default="",
                        help="pdtb2.csv's location")
    # Processed Data
    parser.add_argument("--target_csv", type=str, default="",
                        help="Output pdtb2_translated.csv")

    # Source language
    parser.add_argument("--src_lang", type=str, default="",
                        help="source language")    
    # Target language
    parser.add_argument("--target_lang", type=str, default="",
                        help="target language")
    
    args = parser.parse_args()
    translate_disrpt_csv(args.src_csv, args.target_csv, args.src_lang, args.target_lang)

# wmt2016
# nohup python dranslate_files2021.py --src_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/eng.rst.rstdt_train_enriched.rels --target_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/translated_combi/eng.rst.rstdt_train_enriched_translated3.rels --src_lang=en --target_lang=de > runtime_train_combi3.out 2> runtime_train_combi3.err &

# wmt2021
# nohup python dranslate_files2021.py --src_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/eng.rst.rstdt_train_enriched.rels --target_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/translated_wmt2021_combi/eng.rst.rstdt_train_enriched_translated.rels --src_lang=en --target_lang=de > runtime_train_wmt_combi.out 2> runtime_train_wmt_combi.err &
# nohup python dranslate_files2021_copy.py --src_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/eng.rst.rstdt_test_enriched.rels --target_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/translated_wmt2021_combi/eng.rst.rstdt_test_enriched_translated.rels --src_lang=en --target_lang=de > runtime_test_wmt_combi.out 2> runtime_test_wmt_combi.err &
# nohup python dranslate_files2021_copy.py --src_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/eng.rst.rstdt_dev_enriched.rels --target_csv=/home/VD/kaveri/bert_categorical_tutorial/allennlp_repro/processed/translated_wmt2021_combi/eng.rst.rstdt_dev_enriched_translated.rels --src_lang=en --target_lang=de > runtime_dev_wmt_combi.out 2> runtime_dev_wmt_combi.err &


# nohup jupyter nbconvert --execute --to notebook --inplace 07_Cotraining_en1000_featureless_balance_seed_sweep3.ipynb > nologs/jupyter.out 2> nologs/jupyter.err &
