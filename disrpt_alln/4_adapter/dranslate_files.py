'''Combine using >>>> ls . -v | xargs cat > combi.dev.rels'''

import argparse
import re
import torch
import csv
from spacytokenizer import SpacyTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device('cuda:6')


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
    src_text = [
    'this is a sentence in english that we want to translate to french',
    'This should go to portuguese',
    'And this to Spanish'
    ]
    Hft = HFTranslator(src="en", dest="de")
    for x in src_text:
        result = Hft.translate_sentences(x)
        print(result)
    '''

    def __init__(self, src, dest, device):
        self.tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
        print(device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de").to(device)
        self.src = src
        self.dest = dest
        self.device = device
        assert src in ['en']
        assert dest in ['de']

    def translate_sentences(self, src_text):
        if isinstance(src_text, str):
            input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device).long()
            output_ids = self.model.generate(input_ids, max_new_tokens=512)[0]
            translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            translation = []
            for sentence in src_text:
                input_ids = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device).long()
                output_ids = self.model.generate(input_ids, max_new_tokens=512)[0]
                translation_item = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                translation.append(translation_item)
        return translation

def replace_special_chars(src_text, Stok):
    # no way to handle double escaping "" by csv module. manually remove
    src_text = [_.replace('&lt;*&gt;', '<*>').replace('&quot;', '" ').replace('&#39;', '’') for _ in src_text]
    src_text = [_.replace('&gt;', '>').replace('&lt;', '<') for _ in src_text]
    
    src_text = [Stok.spacify_punkts(_) for _ in src_text]
    src_text = [Stok.spacify_inverted_commas(_) for _ in src_text]
    src_text = ''.join(src_text)
    
    return src_text


def process_chunk(reader, target_csv, src_lang, target_lang, Hft, Stok):
    writer = csv.writer(open(target_csv, 'w'), delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')#, escapechar='\\')
    print("Translating chunk...")
    for count, line in enumerate(reader):

        print('translation 1...')
        # print(replace_special_chars(line[3], Stok))
        arg1 = Hft.translate_sentences(replace_special_chars(line[3], Stok)).replace('\t', ' ')
        print('translation 2...')
        arg2 = Hft.translate_sentences(replace_special_chars(line[4], Stok)).replace('\t', ' ')

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
        print(count)
    print("Translated chunk")
    

def translate_disrpt_csv(src_csv, target_csv, src_lang, target_lang):
    reader = csv.reader(open(src_csv), quoting=csv.QUOTE_NONE, delimiter='\t')
    Stok = SpacyTokenizer()
    Hft = HFTranslator(src=src_lang, dest=target_lang)
    
    process_chunk(reader, target_csv, src_lang, target_lang, Hft=Hft, Stok=Stok)


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