#!/usr/bin/env python
# coding: utf-8

# In[76]:


#seeding for comparing experiment in part 2
import torch
import json
SEED = 2003
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:7')


# # SNLI Bert
# ## Second Tutorial
# https://towardsdatascience.com/fine-tuning-pre-trained-transformer-models-for-sentence-entailment-d87caf9ec9db
# Check his Github code for complete notebook. I never referred to it. Medium was enough.
# BERT in keras-tf: https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b

# In[77]:


# define macros
BERT_MODEL = 'bert-base-german-cased'

batch_size = 4
batches_per_epoch = 541

save_path_suffix = '17_refactor_'


# # Prepare data

# ## load the dataset

# In[78]:


import pandas as pd

# custom reader needed to handle quotechars
def read_df_custom(file):
    header = 'doc     unit1_toks      unit2_toks      unit1_txt       unit2_txt       s1_toks s2_toks unit1_sent      unit2_sent      dir     nuc_children    sat_children    genre   u1_discontinuous        u2_discontinuous       u1_issent        u2_issent       u1_length       u2_length       length_ratio    u1_speaker      u2_speaker      same_speaker    u1_func u1_pos  u1_depdir       u2_func u2_pos  u2_depdir       doclen  u1_position      u2_position     percent_distance        distance        lex_overlap_words       lex_overlap_length      unit1_case      unit2_case      label'
    extracted_columns = ['unit1_txt', 'unit1_sent', 'unit2_txt', 'unit2_sent', 'dir', 'label', 'distance', 'u1_depdir', 'u2_depdir', 'u2_func', 'u1_position', 'u2_position', 'sat_children', 'nuc_children', 'genre', 'unit1_case', 'unit2_case',
                            'u1_discontinuous', 'u2_discontinuous', 'same_speaker', 'lex_overlap_length', 'u1_func']
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

train_df = read_df_custom('../../processed/deu.rst.pcc_train_enriched.rels')
test_df = read_df_custom('../../processed/deu.rst.pcc_test_enriched.rels')
val_df = read_df_custom('../../processed/deu.rst.pcc_dev_enriched.rels')
lang = 'deu'


# ## Clean the data

# In[79]:


#dropping any empty values
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
test_df.dropna(inplace=True)


# ## Prepare a dataset handler class

# In[80]:


train_df.head()


# In[81]:


def remove_classes_not_in_test(train_df, val_df, test_df):
    test_labels = list(test_df['label'].unique())
    train_df = train_df[train_df['label'].isin(test_labels)]
    val_df = val_df[val_df['label'].isin(test_labels)]
    return train_df, val_df, test_df

train_df, val_df, test_df = remove_classes_not_in_test(train_df, val_df, test_df)


# In[82]:


a = torch.Tensor([3, 26996, 20971])
c = torch.Tensor([3, 26996, 20971, 1, 2,34,5,6,7,8,89])
if any([(a == c_).all() for c_ in c]):
    print('a in c')


# In[93]:


from multiprocessing.sharedctypes import Value
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sys import path
path.append('/home/VD/kaveri/anaconda3/envs/py310/lib/python3.10/site-packages/allennlp/data/data_loaders/')
from allennlp.data import allennlp_collate#, DataLoader
# from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST, BertTokenizer
import pandas as pd

class MNLIDataBert(Dataset):

  def __init__(self, train_df, val_df, test_df):
    self.lang = lang
    self.num_labels = set()
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True) # Using a pre-trained BERT tokenizer to encode sentences
    self.train_data = None
    self.val_data = None
    self.test_data = None
    self.train_idx = None
    self.val_idx = None
    self.test_idx = None
    self.init_data()

  def init_data(self):
    self.get_label_mapping()
    self.get_feature_mappings()
    self.train_data, self.train_idx = self.load_data(self.train_df)
    self.val_data, self.val_idx = self.load_data(self.val_df)
    self.test_data, self.test_idx = self.load_data(self.test_df)

  def combine_unique_column_values_to_dict(self, column_name):
    ini_set = set([*self.train_df[column_name].unique(), *self.test_df[column_name].unique(), *self.val_df[column_name].unique()])
    res = dict.fromkeys(ini_set, 0)
    return res

  def get_label_mapping(self):
    labels = {}
    labels_list = list(set(list(self.train_df['label'].unique()) + list(self.test_df['label'].unique()) + list(self.val_df['label'].unique())))
    for i in range(len(labels_list)):
        labels[labels_list[i]] = i
    self.label_dict = labels# {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    # needed later for classification report object to generate precision and recall on test dataset
    self.rev_label_dict = {self.label_dict[k]:k for k in self.label_dict.keys()} 

  def get_feature_mappings(self):
    self.feature_maps = { 'genre': self.combine_unique_column_values_to_dict('genre'),
                          'unit1_case': self.combine_unique_column_values_to_dict('unit1_case'),
                          'unit2_case': self.combine_unique_column_values_to_dict('unit2_case'),
                          'u1_func': self.combine_unique_column_values_to_dict('u1_func') }

  def add_directionality(self, premise, hypothesis, dir):
    if dir == "1<2":
        hypothesis = '< ' + hypothesis + ' {'
    else:
        premise = '} ' + premise + ' >'
    return premise, hypothesis

  def get_distance(self, d):
    if d<-8: return -2.0
    elif d>=-8 and d<-2: return -1.0
    elif d>=-2 and d<0: return 0.0
    elif d>=0 and d<2: return 1.0
    elif d>=2 and d<8: return 2.0
    elif d>=8: return 3.0

  def get_dep(self, d):
    if d=='ROOT': return 0.0
    elif d=='RIGHT': return 1.0
    elif d=='LEFT': return -1.0
    else: raise ValueError()

  def get_u2_func(self, u):
    u2_dict = {'root':0.0, 'conj':1.0, 'advcl':2.0, 'acl':3.0, 'xcomp':4.0, 'obl':5.0, 'ccomp':6.0,
       'parataxis':7.0, 'advmod':8.0, 'dep':9.0, 'csubj':10.0, 'nmod':11.0, 'punct':12.0, 'cc':13.0,
       'appos':14.0, 'aux':15.0, 'obj':16.0, 'iobj':17.0, 'nsubj':18.0, 'nsubj:pass':19.0, 'csubj:pass':20.0}
    return u2_dict[u]

  def get_u_position(self, u):
    if u>=0.0 and u<0.1: return -5.0
    elif u>=0.1 and u<0.2: return -4.0
    elif u>=0.2 and u<0.3: return -3.0
    elif u<=0.3 and u<0.4: return -2.0
    elif u<=0.4 and u<0.5: return -1.0
    elif u<=0.5 and u<0.6: return 0.0
    elif u<=0.6 and u<0.7: return 1.0
    elif u<=0.7 and u<0.8: return 2.0
    elif u<=0.8 and u<0.9: return 3.0
    elif u<=0.9 and u<1.0: return 4.0
    elif u<=1.0 and u<1e9: return 5.0

  def get_lex_overlap_length(self,l):
    if l>=0.0 and l<2.0: return -1
    elif l>=2.0 and l<7.0: return 0
    elif l>=7.0 and l<1e9: return 1

  def get_boolean(self, u):
    if u=='False': return 0.0
    elif u=='True': return 1.0

  def get_mapping_from_dictionary(self, column_name, dict_val):
    return self.feature_maps[column_name][dict_val]

  def get_allen_features(self, features):
    return None

  def get_feature(self, features):
    assert len(features)==16
    distance = self.get_distance(float(features[0]))
    u1_depdir = self.get_dep(features[1])
    u2_depdir = self.get_dep(features[2])
    u2_func = self.get_u2_func(features[3])
    u1_position = self.get_u_position(float(features[4]))
    u2_position = self.get_u_position(float(features[5]))
    sat_children = float(features[6])
    nuc_children = float(features[7])
    genre = self.get_mapping_from_dictionary(column_name='genre', dict_val=features[8])
    unit1_case = self.get_mapping_from_dictionary(column_name='unit1_case', dict_val=features[9])
    unit2_case = self.get_mapping_from_dictionary(column_name='unit2_case', dict_val=features[10])
    u1_discontinuous = self.get_boolean(features[11])
    u2_discontinuous = self.get_boolean(features[12])
    same_speaker = self.get_boolean(features[13])
    lex_overlap_length = self.get_lex_overlap_length(float(features[14]))
    u1_func = self.get_mapping_from_dictionary(column_name='u1_func', dict_val=features[15])
    
    # features2 = self.get_allen_features(features)
    # print(features2)
    # raise ValueError()
    
    if self.lang=='nld':
      return [distance, u1_depdir, sat_children, genre, u1_position]
    elif self.lang=='deu':
      return [distance, u1_depdir, u2_depdir, u2_func, u1_position, u2_position, sat_children, nuc_children]
      #[-0.7413,  0.3142, -1.8323,  
      # 1.7977, -0.4943,  0.3311, 
      # -0.8275,  0.1263, 0.5957,  
      # 0.1327,  0.9722,  0.5304, -0.5750, 
      # -2.1468, -0.8662,  0.1737,  1.7724,  
      # 0.4561,  0.3935, -0.2166, -0.3138,  
          # 0.0000,  1.0000]
      # (distance): Embedding(5, 3, padding_idx=0)
      # (u1_depdir): Embedding(5, 3, padding_idx=0)
      # (u2_depdir): Embedding(5, 3, padding_idx=0)
      # (u2_func): Embedding(14, 4, padding_idx=0)
      # (u1_position): Embedding(12, 4, padding_idx=0)
      # (u2_position): Embedding(12, 4, padding_idx=0)
      # (sat_children): Identity()
      # (nuc_children): Identity()
    elif self.lang=='eng.rst.gum':
      return [distance, same_speaker, u2_func, u2_depdir, unit1_case, unit2_case, nuc_children,
                    sat_children, genre, lex_overlap_length, u2_discontinuous, u1_discontinuous,
                    u1_position, u2_position]
    elif self.lang=='fas':
      return [distance, nuc_children, sat_children, u2_discontinuous, genre]
    elif self.lang=='spa.rst.sctb':
      return [distance, u1_position, sat_children]
    elif self.lang=='zho.rst.sctb':
      return [sat_children, nuc_children, genre, u2_discontinuous, u1_discontinuous, u1_depdir, u1_func]

  def set_labels(self):
    self.num_labels = len(self.num_labels)
    
  def load_data(self, df):
    MAX_LEN = 512 
    token_ids = [] 
    mask_ids = []
    seg_ids = []
    y = []
    feats = []
    idx = []
    idx_map = {}

    self.num_labels.update(df['label'].unique())

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
      genre = row['genre']
      unit1_case = row['unit1_case']
      unit2_case = row['unit2_case']
      u1_discontinuous = row['u1_discontinuous']
      u2_discontinuous = row['u2_discontinuous']
      same_speaker = row['same_speaker']
      lex_overlap_length = row['lex_overlap_length']
      u1_func = row['u1_func']
      features = [distance, u1_depdir, u2_depdir, u2_func, u1_position, u2_position, sat_children, nuc_children, genre, unit1_case, unit2_case, u1_discontinuous, u2_discontinuous, same_speaker,
                  lex_overlap_length, u1_func]

      premise, hypothesis = self.add_directionality(premise, hypothesis, dir)
      # premise_id = self.tokenizer.encode(premise, add_special_tokens = False, max_length=MAX_LEN, truncation=True)
      # hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False, max_length=MAX_LEN, truncation=True)
      encoded = self.tokenizer.encode_plus(premise, hypothesis, add_special_tokens = True, max_length=MAX_LEN, truncation=True, padding='max_length')
      pair_token_ids = torch.tensor(encoded['input_ids'])

      checker = torch.Tensor([3, 26996, 20971,  1456, 13402,  8849,  8268,    50,    21,  8447])
      if (checker==pair_token_ids[:len(checker)]).all():
        print(pair_token_ids)
        raise ValueError()
      # raise ValueError()
      # if pair_token_ids[:1]==[3]:
      #   print(pair_token_ids)
      #   raise ValueError()
      # premise_len = len(premise_id)
      # hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor(encoded['token_type_ids'])
      attention_mask_ids = torch.tensor(encoded['attention_mask'])
      assert len(pair_token_ids)==len(attention_mask_ids)

      token_ids.append(pair_token_ids)
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
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
    self.set_labels()
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


# In[94]:


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


# In[95]:


mnli_dataset = MNLIDataBert(train_df, val_df, test_df)

train_loader, val_loader, test_loader = mnli_dataset.get_data_loaders(batch_size=batch_size, batches_per_epoch=batches_per_epoch) #64X250
label_dict = mnli_dataset.label_dict # required by custom func to calculate accuracy, bert model
rev_label_dict = mnli_dataset.rev_label_dict # required by custom func to calculate accuracy


# In[96]:


for batch_idx, (pair_token_ids, mask_ids, seg_ids, feat, y, idx) in enumerate(train_loader):
    assert pair_token_ids.shape[-1]==512 #torch.Size([4, 512])
    assert mask_ids.shape[-1]==512
    assert seg_ids.shape[-1]==512
    assert feat.shape[-1]==8
    # y.shape==torch.Size([4])
    # idx.shape==torch.Size([4])


# # Define the model

# ## load pretrained model

# In[97]:


from transformers import BertForSequenceClassification, AdamW
from torch import optim
import os
path.append(os.path.join(os.getcwd(), '../utils/'))
from CategoricalAccuracy import CategoricalAccuracy as CA
import numpy as np

ca = CA()

x = torch.tensor(np.array([[[1,0,0], [1,0,0], [1,0,0]]]))
y1 = torch.tensor(np.array([[0], [1], [1]]))
y2 = torch.tensor(np.array([[0], [0], [0]]))

ca(x,y1)
print(ca.get_metric(reset=True))
ca(x,y2)
print(ca.get_metric(reset=True))


# ## define evaulation metric

# In[98]:


# to evaluate model for train and test. And also use classification report for testing
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# helper function to calculate the batch accuracy
def multi_acc(y_pred, y_test, allennlp=False):
  if allennlp==False:
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc

# freeze model weights and measure validation / test 
def evaluate_accuracy(model, optimizer, data_loader, rev_label_dict, label_dict, is_training=True):
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
  
  return val_acc, val_loss, cr, model, optimizer


# ## define custom bert model

# In[99]:


from transformers import BertModel, AutoTokenizer
import torch.nn as nn
from torch import eq
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
          super(CustomBERTModel, self).__init__()
          self.num_classes = num_labels
          print('ASSIGN:', self.num_classes)
          self.bert = BertModel.from_pretrained(BERT_MODEL)
          self.linear1 = nn.Linear(776, self.num_classes)
          self.act1 = nn.Softmax(dim=-1)

    def forward(self, pair_token_ids, token_type_ids, attention_mask, feat):
        sequence_output, pooled_output = self.bert(input_ids=pair_token_ids, 
                        token_type_ids=token_type_ids, 
                        attention_mask=attention_mask).values()
        feat_concat = torch.concat((pooled_output, feat),-1)
        assert feat_concat.shape[-1] == 776

        linear1_output = self.linear1(feat_concat) ## extract the 1st token's embeddingsp
        # linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
        linear1_output = self.act1(linear1_output)
        assert linear1_output[0].sum()>0.9 and linear1_output[0].sum()<1.1
        return linear1_output

model = CustomBERTModel(mnli_dataset.num_labels) 
model.to(device)
optimizer = AdamW(model.parameters(), lr=4e-6, correct_bias=False)#original 2e-5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, mode='max', patience=7, min_lr=5e-7, verbose=True)#original factor=0.6, min_lr=5e-7


# ## define training regime

# In[100]:


from torch.utils.tensorboard import SummaryWriter

def writer_init(save_path_suffix):
    writer_path = 'run1/'+save_path_suffix[:-1]+'/'
    if os.path.isdir(writer_path):
        filelist = [ f for f in os.listdir(writer_path) if 'events.out' in f ]
        print(filelist)
        for f in filelist:
            os.remove(os.path.join(writer_path, f))
    else:
        os.mkdir(writer_path)
    writer = SummaryWriter(log_dir=writer_path)
    return writer

writer = writer_init(save_path_suffix)


# In[101]:


### MODIFIED
import time
import traceback
import torch.nn.functional as F

from typing import Optional, Iterable, Dict, Any
from EarlyStopperUtil import MetricTracker
from sklearn.metrics import classification_report

EPOCHS = 100


def train(model, train_loader, val_loader, optimizer, scheduler, rev_label_dict):  
  EarlyStopper = MetricTracker(patience=12, metric_name='+accuracy')
  best_val_acc = 0


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
      loss.backward()
      acc = multi_acc(outputs, labels)
      optimizer.step()
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      losses.append(loss)
      accuracies.append(acc)
      
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)

    val_acc, val_loss, cr, model, optimizer = evaluate_accuracy(model, optimizer, val_loader, rev_label_dict, label_dict, None)
    if val_acc>best_val_acc:
      torch.save(model.state_dict(), save_path_suffix+'_best.pt')
      best_val_acc = val_acc
      print(f'Epoch {epoch+1}: Best val_acc: {best_val_acc:.4f}')
    if val_acc>=best_val_acc:
      torch.save(model.state_dict(), save_path_suffix+'_best_latest.pt')
      best_val_acc = val_acc
      print(f'Epoch {epoch+1}: Best val_acc: {best_val_acc:.4f}')
    EarlyStopper.add_metric(val_acc)
    if EarlyStopper.should_stop_early(): break

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print(f'train_size: {train_size}')

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_acc', val_acc, epoch)


# In[102]:


import warnings
from sklearn.exceptions import DataConversionWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    train(model, train_loader, val_loader, optimizer, scheduler, rev_label_dict)


# # test

# In[ ]:


#latest
def validate(model, test_loader, optimizer, rev_label_dict, label_dict):
  start = time.time()
  test_acc, test_loss, cr, model, optimizer = evaluate_accuracy(model, optimizer, test_loader, rev_label_dict, label_dict, is_training=False)
  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)

  print(f'Test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}')
  print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
  print(cr)

  return test_loss, test_acc


test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict)
writer.add_scalar('test_loss_latest', test_loss, 1)
writer.add_scalar('test_acc_latest', test_acc, 1)
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')


# In[ ]:


#best earliest
model.load_state_dict(torch.load(save_path_suffix+'_best.pt'))
test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict)
writer.add_scalar('test_loss_best_earliest', test_loss, 1)
writer.add_scalar('test_acc_best_earliest', test_acc, 1)
print(f'Latest Test Loss: {test_loss:.3f} |  Latest Test Acc: {test_acc*100:.2f}%')


# In[ ]:


#best lastest
model.load_state_dict(torch.load(save_path_suffix+'_best_latest.pt'))
test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict)
writer.add_scalar('test_loss_best_latest', test_loss, 1)
writer.add_scalar('test_acc_best_latest', test_acc, 1)
print(f'Best Test Loss: {test_loss:.3f} |  Best Test Acc: {test_acc*100:.2f}%')


# In[ ]:


#best val acc
model.load_state_dict(torch.load(save_path_suffix+'_best_latest.pt'))
test_loss, test_acc = validate(model, val_loader, optimizer, rev_label_dict, label_dict)
writer.add_scalar('val_loss_best_latest', test_loss, 1)
writer.add_scalar('val_acc_best_latest', test_acc, 1)
print(f'Val Loss: {test_loss:.3f} |  Val Acc: {test_acc*100:.2f}%')

