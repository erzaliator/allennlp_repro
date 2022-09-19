#!/usr/bin/env python
# coding: utf-8


#seeding for comparing experiment in part 2
import torch
import json
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:1')


# define macros
BERT_MODEL = 'bert-base-german-cased' #'dbmdz/bert-base-german-cased'
MODEL = 'bert'
# 'bert-base-multilingual-cased'
# 'bert-base-uncased'

batch_size = 4
batches_per_epoch = 541


# # Prepare data

# ## load the dataset

import pandas as pd

# custom reader needed to handle quotechars
def read_df_custom(file):
    header = 'doc     unit1_toks      unit2_toks      unit1_txt       unit2_txt       s1_toks s2_toks unit1_sent      unit2_sent      dir     nuc_children    sat_children    genre   u1_discontinuous        u2_discontinuous       u1_issent        u2_issent       u1_length       u2_length       length_ratio    u1_speaker      u2_speaker      same_speaker    u1_func u1_pos  u1_depdir       u2_func u2_pos  u2_depdir       doclen  u1_position      u2_position     percent_distance        distance        lex_overlap_words       lex_overlap_length      unit1_case      unit2_case      label'
    extracted_columns = ['unit1_txt', 'unit1_sent', 'unit2_txt', 'unit2_sent', 'dir', 'label', 'distance', 'u1_depdir', 'u2_depdir', 'u2_func', 'u1_position', 'u2_position', 'sat_children', 'nuc_children']
    header = header.split()
    df = pd.DataFrame(columns=extracted_columns)
    file = open(file, 'r')

    rows = []
    count = 0 
    for line in file:
        line = line[:-1].split('\t')
        count+=1
        if count ==1: continue
        row = {}
        for column in extracted_columns:
            index = header.index(column)
            row[column] = line[index]
        rows.append(row)

    df = pd.concat([df, pd.DataFrame.from_records(rows)])
    return df

# we only need specific columns
# train_df = read_df_custom('../processed/nld.rst.nldt_train_enriched.rels')
# test_df = read_df_custom('../processed/nld.rst.nldt_test_enriched.rels')
# val_df = read_df_custom('../processed/nld.rst.nldt_dev_enriched.rels')
# train_df = read_df_custom('../processed/fas.rst.prstc_train_enriched.rels')
# test_df = read_df_custom('../processed/fas.rst.prstc_test_enriched.rels')
# val_df = read_df_custom('../processed/fas.rst.prstc_dev_enriched.rels')
train_df = read_df_custom('../processed/deu.rst.pcc_train_enriched.rels')
test_df = read_df_custom('../processed/deu.rst.pcc_test_enriched.rels')
val_df = read_df_custom('../processed/deu.rst.pcc_dev_enriched.rels')


# ## Clean the data


#dropping any empty values
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# train_df = train_df[:500]
# val_df = val_df[:50]
# test_df = test_df[:50]


# ## Prepare a dataset handler class

from multiprocessing.sharedctypes import Value
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sys import path
path.append('/home/VD/kaveri/anaconda3/envs/py310/lib/python3.10/site-packages/allennlp/data/data_loaders/')
from allennlp.data import allennlp_collate#, DataLoader
# from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, AutoTokenizer, XLMRobertaTokenizer
import pandas as pd

class MNLIDataBert(Dataset):

  def __init__(self, train_df, val_df, test_df):
    self.num_labels = -1
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    if MODEL=='bert': self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True) # Using a pre-trained BERT tokenizer to encode sentences
    elif MODEL=='xlmr': self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base") # Using a pre-trained BERT tokenizer to encode sentences
    else: self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base") # Using a pre-trained BERT tokenizer to encode sentences
    
    self.train_data = None
    self.val_data = None
    self.test_data = None
    self.train_idx = None
    self.val_idx = None
    self.test_idx = None
    self.init_data()

  def init_data(self):
    self.get_label_mapping()
    self.train_data, self.train_idx = self.load_data(self.train_df)
    self.val_data, self.val_idx = self.load_data(self.val_df)
    self.test_data, self.test_idx = self.load_data(self.test_df)

  def get_label_mapping(self):
    labels = {}
    labels_list = list(set(list(self.train_df['label'].unique()) + list(self.test_df['label'].unique()) + list(self.val_df['label'].unique())))
    for i in range(len(labels_list)):
        labels[labels_list[i]] = i
    self.label_dict = labels# {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    # needed later for classification report object to generate precision and recall on test dataset
    self.rev_label_dict = {self.label_dict[k]:k for k in self.label_dict.keys()} 
  
  def add_directionality(self, premise, hypothesis, dir):
    if dir == "1<2":
        hypothesis = 'left ' + hypothesis + ' {'
        # hypothesis = '< ' + hypothesis + ' {'
        # hypothesis = '{ ' + hypothesis + ' {'
    else:
        premise = '} ' + premise + ' right'
        # premise = '} ' + premise + ' >'
        # premise = '} ' + premise + ' }'
    return premise, hypothesis

  def get_distance(self, d):
    if d<-8: return -2
    elif d>=-8 and d<-2: return -1
    elif d>=-2 and d<0: return 0
    elif d>=0 and d<2: return 1
    elif d>=2 and d<8: return 2
    elif d>=8: return 3

  def get_dep(self, d):
    if d=='ROOT': return 0
    elif d=='RIGHT': return 1
    elif d=='LEFT': return -1
    else: raise ValueError()

  def get_u2_func(self, u):
    u2_dict = {'root':0, 'conj':1, 'advcl':2, 'acl':3, 'xcomp':4, 'obl':5, 'ccomp':6,
       'parataxis':7, 'advmod':8, 'dep':9, 'csubj':10, 'nmod':11, 'punct':12, 'cc':13,
       'appos':14, 'aux':15, 'obj':16, 'iobj':17, 'nsubj':18, 'nsubj:pass':19, 'csubj:pass':20}
    return u2_dict[u]

  def get_u_position(self, u):
    if u>=0.0 and u<0.1: return -5
    elif u>=0.1 and u<0.2: return -4
    elif u>=0.2 and u<0.3: return -3
    elif u<=0.3 and u<0.4: return -2
    elif u<=0.4 and u<0.5: return -1
    elif u<=0.5 and u<0.6: return 0
    elif u<=0.6 and u<0.7: return 1
    elif u<=0.7 and u<0.8: return 2
    elif u<=0.8 and u<0.9: return 3
    elif u<=0.9 and u<1.0: return 4
    elif u<=1.0 and u<1e9: return 5

  def get_feature(self, features):
    distance = self.get_distance(int(features[0]))
    u1_depdir = self.get_dep(features[1])
    u2_depdir = self.get_dep(features[2])
    u2_func = self.get_u2_func(features[3])
    u1_position = self.get_u_position(float(features[4]))
    u2_position = self.get_u_position(float(features[5]))
    sat_children = int(features[6])
    nuc_children = int(features[6])
    return [distance, u1_depdir, u2_depdir, u2_func, u1_position, u2_position, sat_children, nuc_children]
    

  def load_data(self, df):
    MAX_LEN = 256 # dont need to enforce this now because snli is a sanitized dataset where sentence lenghts are reasonable. otherwise the beert model doesn't have enough parameters to handle long length sentences
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []
    feats = []
    idx = []
    idx_map = {}

    self.num_labels = max(self.num_labels, len(df['label'].unique()))

    count=0
    for row in df.iterrows():
      row = row[1]
      premise = row['unit1_txt']
      hypothesis = row['unit2_txt']
      label = row['label']
      dir = row['dir']

      distance = row['distance']
      u1_depdir = row['u1_depdir']
      u2_depdir = row['u2_depdir']
      u2_func = row['u2_func']
      u1_position = row['u1_position']
      u2_position = row['u2_position']
      sat_children = row['sat_children']
      nuc_children = row['nuc_children']
      features = [distance, u1_depdir, u2_depdir, u2_func, u1_position, u2_position, sat_children, nuc_children]

      # print(self.tokenizer.encode("< "))
      # print(self.tokenizer.encode("< this"))
      # print(self.tokenizer.encode("this ."))
      # print(self.tokenizer.encode("this } "))
      # print(self.tokenizer.encode("fffffff } "))

      premise, hypothesis = self.add_directionality(premise, hypothesis, dir)

      if MODEL=='bert':
        premise_id = self.tokenizer.encode(premise, add_special_tokens = False, max_length=MAX_LEN, truncation=True)
        hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False, max_length=MAX_LEN, truncation=True)
        pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
        # pair_token_ids = self.tokenizer.encode(premise, hypothesis, add_special_tokens = True, max_length=MAX_LEN*2, truncation=True)
      elif MODEL=='xlmr':
        pair_token_ids = self.tokenizer(premise, hypothesis)
        attention_mask_ids = pair_token_ids['attention_mask']
        pair_token_ids = pair_token_ids['input_ids']
      
      premise_len = len(premise_id) if MODEL=='bert' else 10
      hypothesis_len = len(hypothesis_id) if MODEL=='bert' else 10

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      if MODEL=='bert': attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(torch.tensor(attention_mask_ids))
      y.append(self.label_dict[label])
      feats.append(self.get_feature(features))

      idx_map[count] = [premise, hypothesis]
      idx.append(count)
      count+=1
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)

    y = torch.tensor(y)
    idx = torch.tensor(idx)
    feats = torch.tensor(feats)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, feats, y, idx)
    return dataset, idx_map

  def get_data_loaders(self, batch_size=4, batches_per_epoch=402, shuffle=True): #1609 samples / 64:25=1600 / 402:4=1608
    train_loader_torch = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size,
    )

    val_loader_torch = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size,
    )

    test_loader_torch = DataLoader(
      self.test_data,
      shuffle=False,#shuffle,
      batch_size=batch_size,
    )
    
    train_loader = LoaderWrapper(train_loader_torch, n_step=batches_per_epoch)
    val_loader = LoaderWrapper(val_loader_torch, n_step=batches_per_epoch)
    test_loader = LoaderWrapper(test_loader_torch, n_step=batches_per_epoch)

    return train_loader, val_loader_torch, test_loader_torch


# In[101]:


class LoaderWrapper:
    def __init__(self, loader, n_step):
        self.step = n_step
        self.idx = 0
        self.iter_loader = iter(loader)
        self.loader = loader
    
    def __iter__(self):
        return self

    def __len__(self):
        return self.step

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)


# In[102]:


mnli_dataset = MNLIDataBert(train_df, val_df, test_df)

train_loader, val_loader, test_loader = mnli_dataset.get_data_loaders(batch_size=batch_size, batches_per_epoch=batches_per_epoch) #64X250
label_dict = mnli_dataset.label_dict # required by custom func to calculate accuracy, bert model
rev_label_dict = mnli_dataset.rev_label_dict # required by custom func to calculate accuracy


from CategoricalAccuracy import CategoricalAccuracy as CA
import numpy as np

ca = CA()

# to evaluate model for train and test. And also use classification report for testing
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# helper function to calculate the batch accuracy
def multi_acc(y_pred, y_test, allennlp=False):
  if allennlp==False:
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc

def save_cm(y_true, y_pred, target_names, display_zeros=True):
  confusion_matrix = pd.crosstab(y_true, y_pred)
  print(confusion_matrix)
  target_namesx = confusion_matrix.columns
  target_namesy = confusion_matrix.index.values

  fig, ax = plt.subplots(figsize=(10,10))
  s = sns.heatmap(confusion_matrix, xticklabels=target_namesx, yticklabels=target_namesy, annot=True, fmt = '.5g')

  plt.title('CM predicted v actual values')
  plt.xlabel('Pred')
  plt.ylabel('True')
  plt.tight_layout()
  # plt.show()
  # plt.savefig(image_file+exp+'.png')
  # print(image_file+exp+'.png')

# freeze model weights and measure validation / test 
def evaluate_accuracy(model, optimizer, data_loader, rev_label_dict, label_dict, save_path, is_training=True):
  model.eval()
  total_val_acc  = 0
  total_val_loss = 0
  
  #for classification report
  y_true = []
  y_pred = []
  idx_list = []
  premise_list = []
  hypo_list = []
  idx_map = mnli_dataset.val_idx if is_training else mnli_dataset.test_idx

  with torch.no_grad():
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, feat, y, idx) in enumerate(data_loader):      
      optimizer.zero_grad()
      # labels = y
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      feat = feat.to(device)
      
      outputs = model(pair_token_ids, 
                            token_type_ids=seg_ids, 
                            attention_mask=mask_ids, 
                            feat=feat)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(outputs, labels)
      acc = multi_acc(outputs, labels)

      total_val_loss += loss.item()
      total_val_acc  += acc.item()

      # log predictions for classification report
      argmax_predictions = torch.argmax(outputs,dim=1).tolist()
      labels_list = labels.tolist()
      assert(len(labels_list)==len(argmax_predictions))
      for p in argmax_predictions: y_pred.append(rev_label_dict[int(p)])
      for l in labels_list: y_true.append(rev_label_dict[l])
      for i in idx.tolist():
        idx_list.append(i)
        premise_list.append(idx_map[i][0])
        hypo_list.append(idx_map[i][1])

  val_acc  = total_val_acc/len(data_loader)
  val_loss = total_val_loss/len(data_loader)
  cr = classification_report(y_true, y_pred)

  idx_json = {'idx': idx_list, 'gold_label': y_true, 'pred_label': y_pred, 'premise': premise_list, 'hypothesis': hypo_list}
  # if not is_training: json.dump(idx_json, open(save_path, 'w', encoding='utf8'), ensure_ascii=False)

  # if not is_training:
  #   save_cm(y_true, y_pred, rev_label_dict)
  
  return val_acc, val_loss, cr, model, optimizer


# ## define bert custom model

# In[111]:


from transformers import BertModel, AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModelForMaskedLM, XLMRobertaModel
import torch.nn as nn
class CustomBERTModel(nn.Module):
    #https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
    def __init__(self, num_labels):
          super(CustomBERTModel, self).__init__()
          self.num_classes = num_labels+1 # zero indexed classes
          print('ASSIGN:', self.num_classes)

          if MODEL=='bert': self.bert = BertModel.from_pretrained(BERT_MODEL)
          elif MODEL=='xlmr': self.bert = XLMRobertaModel.from_pretrained("xlm-roberta-base")
          else: self.bert = AutoModel.from_pretrained("xlm-roberta-base", config=AutoConfig.from_pretrained("xlm-roberta-base", output_attentions=True,output_hidden_states=True)) 
          
          ### New layers:
          self.linear1 = nn.Linear(776, 512)
          self.linear2 = nn.Linear(512, 256)
          self.linear3 = nn.Linear(256, 128)
          self.linear4 = nn.Linear(128, self.num_classes)
          self.act1 = nn.ReLU() # can i use the same activation object everywhere?
          self.act2 = nn.ReLU()
          self.act3 = nn.ReLU()
          self.drop = nn.Dropout(0.1) 

    def forward(self, pair_token_ids, token_type_ids, attention_mask, feat):
        if MODEL=='bert':
          sequence_output, pooled_output = self.bert(input_ids=pair_token_ids, 
                          token_type_ids=token_type_ids, 
                          attention_mask=attention_mask).values()
        elif MODEL=='xlmr':
          pair_token_ids = {'input_ids': pair_token_ids, 'attention_mask': attention_mask}
          sequence_output = self.bert(**pair_token_ids)['last_hidden_state']

        feat_concat = torch.concat((sequence_output[:,0,:].view(-1,768), feat),-1)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(feat_concat) ## extract the 1st token's embeddings
        linear1_output = self.act1(linear1_output)
        linear2_output = self.linear2(linear1_output)
        linear2_output = self.act2(linear2_output)
        # drop_output = self.drop(linear2_output)
        linear3_output = self.linear3(linear2_output)
        linear3_output = self.act3(linear3_output)
        linear4_output = self.linear4(linear3_output)
        drop_output = self.drop(linear4_output)
        return linear4_output# loss, outputs

# tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = CustomBERTModel(mnli_dataset.num_labels) # You can pass the parameters if required to have more flexible model
model.to(device) ## can be gpu
# optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)



from transformers import BertForSequenceClassification, AdamW
from torch import optim

# model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(label_dict)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, mode='max', patience=2, min_lr=5e-7, verbose=True)




### MODIFIED
import time
import traceback
import torch.nn.functional as F
from typing import Optional, Iterable, Dict, Any
from EarlyStopperUtil import MetricTracker


EPOCHS = 100
best_epoch = 'N'

def train(model, train_loader, val_loader, optimizer, scheduler, rev_label_dict):  
  EarlyStopper = MetricTracker(patience=12, metric_name='+accuracy')
  best_acc = -1

  for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0

    # logging for scheduler
    losses = []
    accuracies= []

    train_size = 0

    for batch_idx, (pair_token_ids, mask_ids, seg_ids, feat, y, idx) in enumerate(train_loader):
      train_size+=1
      optimizer.zero_grad()
      # labels = y
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      feat = feat.to(device)

      ############new code#####################

      outputs = model(pair_token_ids, 
                            token_type_ids=seg_ids, 
                            attention_mask=mask_ids,
                            feat=feat)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(outputs, labels)
      loss.backward()
      acc = multi_acc(outputs, labels)
      optimizer.step()
      ################old code#################

      # loss, prediction = model(pair_token_ids, 
      #                       token_type_ids=seg_ids, 
      #                       attention_mask=mask_ids, 
      #                       labels=labels).values()

      # acc = multi_acc(prediction, labels)
      # loss.backward()
      # optimizer.step()

      ########################################
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      # log losses for scheduler
      losses.append(loss)
      mean_loss = sum(losses)/len(losses)
      scheduler.step(mean_loss)

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)

    val_acc, val_loss, cr, model, optimizer = evaluate_accuracy(model, optimizer, val_loader, rev_label_dict, label_dict, None)
    
    if val_acc>=best_acc:
        if epoch>4:
          torch.save(model.state_dict(), 'best_debug/deu_debugXlmr_best_'+str(epoch)+'.pt')
          print('Saving at.... deu_debugXlmr_best_'+str(epoch)+'.pt')

        global best_epoch
        best_acc = val_acc
        best_epoch = epoch
    
    EarlyStopper.add_metric(val_acc)
    if EarlyStopper.should_stop_early(): break

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print(f'train_size: {train_size}')


import warnings
from sklearn.exceptions import DataConversionWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    train(model, train_loader, val_loader, optimizer, scheduler, rev_label_dict)


save_path = 'debugXlmr_last.pt'
torch.save(model.state_dict(), save_path)

def validate(model, test_loader, optimizer, rev_label_dict, label_dict, save_path):
  start = time.time()
  test_acc, test_loss, cr, model, optimizer = evaluate_accuracy(model, optimizer, test_loader, rev_label_dict, label_dict, save_path.replace('.pt', '.json'), is_training=False)
  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)

  print(f'Test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}')
  print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
  print(cr)

  return test_loss, test_acc

print('LAST ACC')
test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict, save_path)
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

print('BEST ACC')
print('Loading data from epoch ', best_epoch)
model.load_state_dict(torch.load('best_debug/deu_debugXlmr_best_'+str(best_epoch)+'.pt'))
test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict, save_path)
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')