#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


# define macros
lang = 'zho'
BERT_MODEL = 'hfl/chinese-bert-wwm-ext' #'dbmdz/bert-base-german-cased'
# 'bert-base-multilingual-cased'
# 'bert-base-uncased'

batch_size = 4
batches_per_epoch = 110


# # Prepare data

# ## load the dataset

# In[25]:


# !wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip


# In[3]:


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
train_df = read_df_custom('../../processed/zho.rst.sctb_train_enriched.rels')
test_df = read_df_custom('../../processed/zho.rst.sctb_test_enriched.rels')
val_df = read_df_custom('../../processed/zho.rst.sctb_dev_enriched.rels')
# train_df = read_df_custom('../../processed/zho.rst.sctb_train_enriched.rels')
# test_df = read_df_custom('../../processed/zho.rst.sctb_test_enriched.rels')
# val_df = read_df_custom('../../processed/zho.rst.sctb_dev_enriched.rels')
# train_df = read_df_custom('../../processed/nld.rst.nldt_train_enriched.rels')
# test_df = read_df_custom('../../processed/nld.rst.nldt_test_enriched.rels')
# val_df = read_df_custom('../../processed/nld.rst.nldt_dev_enriched.rels')
# train_df = read_df_custom('../../processed/fas.rst.prstc_train_enriched.rels')
# test_df = read_df_custom('../../processed/fas.rst.prstc_test_enriched.rels')
# val_df = read_df_custom('../../processed/fas.rst.prstc_dev_enriched.rels')
# train_df = read_df_custom('../../processed/deu.rst.pcc_train_enriched.rels')
# test_df = read_df_custom('../../processed/deu.rst.pcc_test_enriched.rels')
# val_df = read_df_custom('../../processed/deu.rst.pcc_dev_enriched.rels')


# ## Clean the data

# In[4]:


#dropping any empty values
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
test_df.dropna(inplace=True)


# ## Prepare a dataset handler class

# In[28]:


train_df.head()


# In[29]:


train_df['sat_children'].unique()


# In[5]:


from multiprocessing.sharedctypes import Value
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sys import path
path.append('/home/VD/kaveri/anaconda3/envs/py310/lib/python3.10/site-packages/allennlp/data/data_loaders/')
from allennlp.data import allennlp_collate#, DataLoader
# from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import pandas as pd

class MNLIDataBert(Dataset):

  def __init__(self, train_df, val_df, test_df):
    self.num_labels = -1
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
        # hypothesis = 'left ' + hypothesis + ' {'
        hypothesis = '< ' + hypothesis + ' {'
        # hypothesis = '{ ' + hypothesis + ' {'
    else:
        # premise = '} ' + premise + ' right'
        premise = '} ' + premise + ' >'
        # premise = '} ' + premise + ' }'
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
    try: result = u2_dict[u]
    except: result = len(u2_dict)+1
    return result

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

  def get_feature(self, features):
    distance = self.get_distance(float(features[0]))
    u1_depdir = self.get_dep(features[1])
    u2_depdir = self.get_dep(features[2])
    u2_func = self.get_u2_func(features[3])
    u1_position = self.get_u_position(float(features[4]))
    u2_position = self.get_u_position(float(features[5]))
    sat_children = float(features[6])
    nuc_children = float(features[6])
    return [distance, u1_depdir, u2_depdir, u2_func, u1_position, u2_position, sat_children, nuc_children]

  def load_data2(self, df):
    MAX_LEN = 256 # dont need to enforce this now because snli is a sanitized dataset where sentence lenghts are reasonable. otherwise the beert model doesn't have enough parameters to handle long length sentences
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []
    idx = []
    idx_map = {}
    # self.reach = 'reach'

    premise_list = df['unit1_txt'].to_list()
    hypothesis_list = df['unit2_txt'].to_list()
    label_list = df['label'].to_list()
    dir_list = df['dir'].to_list()
    
    self.num_labels = max(self.num_labels, len(df['label'].unique()))

    count=0
    for (premise, hypothesis, label, dir) in zip(premise_list, hypothesis_list, label_list, dir_list):
      # print('old: ', premise, hypothesis)
      premise, hypothesis = self.add_directionality(premise, hypothesis, dir)
      # print('new:', premise, hypothesis, '\n')
    

  def load_data(self, df):
    MAX_LEN = 256 # dont need to enforce this now because snli is a sanitized dataset where sentence lenghts are reasonable. otherwise the beert model doesn't have enough parameters to handle long length sentences
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []
    feats = []
    idx = []
    idx_map = {}

    #"u1_depdir", "u2_depdir", "u2_func", "u1_position", "u2_position","sat_children", "nuc_children"
    
    self.num_labels = max(self.num_labels, len(df['label'].unique()))

    count=0
    # for (premise, hypothesis, label, dir, distance, u1_depdir) in zip(premise_list, hypothesis_list, label_list, dir_list, u1_depdir_list, feat_list):
    for row in df.iterrows():
      row = row[1]
      premise = row['unit1_txt']
      # premise = 'Die Stadt, in der ich wohne, ist ziemlich klein. Sie hat nur 45.000 Einwohner und ist umgeben von Landwirtschaft und Wäldern. Wir haben auch einen schönen See, welcher im Sommer eine große Attraktion ist, und viele Turisten machen hier dann Urlaub. Ich bin früher immer Eislaufen gewesen auf dem See, als es im Winter noch kälter war. Man kann dort schwimmen, segeln oder windsurfen, und man kann sogar Wakeboarding lernen. Meine Stadt liegt nicht weit von einer großen Stadt, so dass die Leute dort auch shoppen gehen können, da die Zugfahrt nur 15 Minuten dauert. Unser Stadtzentrum ist sehr alt und klein, mit kleinen Geschäften und einer entspannten Atmosphäre. Es ist toll für Familien dort, weil es sehr sicher ist und eine Fußgängerzone hat, wo keine Fahrzeuge erlaubt sind. Im Sommer kann man in einen der italienischen Eis-Cafés draußen sitzen und die Passanten beobachten. Wir haben auch viele Schwimmbäder und Freibäder. Hier kann man immer was unternehmen, außer ins Kino zu gehen. Es wurde vor ein paar Jahren geschlossen, weil jetzt jeder in das riesige Kino in der großen Stadt geht. Ich lebe gern hier, da alles was ich brauche dicht dran ist und ich hier eine tolle Zeit mit meinen Freunden haben kann.'
      # premise = 'Ich bin eine Frau .'
      hypothesis = row['unit2_txt']
      # hypothesis = 'Ich bin ein Mann .'
      label = row['label']
      dir = row['dir']
      # dir='1>2'

      distance = row['distance']
      u1_depdir = row['u1_depdir']
      u2_depdir = row['u2_depdir']
      u2_func = row['u2_func']
      u1_position = row['u1_position']
      u2_position = row['u2_position']
      sat_children = row['sat_children']
      nuc_children = row['nuc_children']
      features = [distance, u1_depdir, u2_depdir, u2_func, u1_position, u2_position, sat_children, nuc_children]

      premise, hypothesis = self.add_directionality(premise, hypothesis, dir)
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False, max_length=MAX_LEN, truncation=True)
      # print(premise)
      # print(premise_id)
      # print(self.tokenizer.encode("< "))
      # print(self.tokenizer.encode("< this"))
      # print(self.tokenizer.encode("this ."))
      # print(self.tokenizer.encode("this } "))
      # print(self.tokenizer.encode("fffffff } "))
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False, max_length=MAX_LEN, truncation=True)
      # pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      encoded = self.tokenizer.encode_plus(premise, hypothesis, add_special_tokens = True, max_length=MAX_LEN, truncation=True, padding='max_length')
      pair_token_ids = torch.tensor(encoded['input_ids'])
      # print('tokens:', self.tokenizer.tokenize(premise))
      # print('tokens:', self.tokenizer.tokenize(hypothesis))
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      # segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      # attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values
      segment_ids = torch.tensor(encoded['token_type_ids'])
      attention_mask_ids = torch.tensor(encoded['attention_mask'])
      assert len(pair_token_ids)==len(attention_mask_ids)

      token_ids.append(pair_token_ids)
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      y.append(self.label_dict[label])
      feats.append(self.get_feature(features))
      # feat.append(int(feature))

      # print('OUTPUT')
      # print(pair_token_ids.shape)
      # print(segment_ids.shape)
      # print(attention_mask_ids.shape)
      # print(feats)
      # raise ValueError()

      idx_map[count] = [premise, hypothesis]
      idx.append(count)
      count+=1
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)

    y = torch.tensor(y)
    idx = torch.tensor(idx)
    feats = torch.tensor(feats)
    print(feats)
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


# In[6]:


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


# In[7]:


mnli_dataset = MNLIDataBert(train_df, val_df, test_df)

train_loader, val_loader, test_loader = mnli_dataset.get_data_loaders(batch_size=batch_size, batches_per_epoch=batches_per_epoch) #64X250
label_dict = mnli_dataset.label_dict # required by custom func to calculate accuracy, bert model
rev_label_dict = mnli_dataset.rev_label_dict # required by custom func to calculate accuracy




print(label_dict)


# # Define the model

# ## load pretrained model

# In[11]:


from transformers import BertForSequenceClassification, AdamW
from torch import optim

# model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(label_dict)).to(device)
# optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, mode='max', patience=2, min_lr=5e-7, verbose=True)


# In[12]:


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

# In[13]:


# to evaluate model for train and test. And also use classification report for testing
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# helper function to calculate the batch accuracy
def multi_acc(y_pred, y_test, allennlp=False):
  if allennlp==False:
    # print(y_pred.shape)
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
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      feat = feat.to(device)
      
      # loss, prediction = model(pair_token_ids, 
      #                       token_type_ids=seg_ids, 
      #                       attention_mask=mask_ids, 
      #                       labels=labels).values()
      # acc = multi_acc(prediction, labels)

      ############new code#####################

      outputs = model(pair_token_ids, 
                            token_type_ids=seg_ids, 
                            attention_mask=mask_ids, 
                            feat=feat)
      # probs = F.softmax(outputs, dim=1)
      # max_idx = torch.max(outputs, 1).indices
      # one_hot = F.one_hot(max_idx, outputs.shape[1])

      criterion = nn.CrossEntropyLoss()
      loss = criterion(outputs, labels)
      acc = multi_acc(outputs, labels)
      ########################################

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


# ## define custom bert model

# In[14]:


from transformers import BertModel, AutoTokenizer
import torch.nn as nn
class CustomBERTModel(nn.Module):
    #https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
    def __init__(self, num_labels):
          super(CustomBERTModel, self).__init__()
          self.num_classes = num_labels+1 # zero indexed classes
          print('ASSIGN:', self.num_classes)
          self.bert = BertModel.from_pretrained(BERT_MODEL)
          ### New layers:
          self.linear1 = nn.Linear(776, self.num_classes)
          # self.linear2 = nn.Linear(512, 256)
          # self.linear3 = nn.Linear(256, 128)
        #   self.linear4 = nn.Linear(128, self.num_classes)
          self.act1 = nn.Softmax(dim=-1) # can i use the same activation object everywhere?
        #   self.act2 = nn.ReLU()
        #   self.act3 = nn.ReLU()
          self.drop = nn.Dropout(0.1) 

    def forward(self, pair_token_ids, token_type_ids, attention_mask, feat):
        sequence_output, pooled_output = self.bert(input_ids=pair_token_ids, 
                        token_type_ids=token_type_ids, 
                        attention_mask=attention_mask).values()

        feat = self.drop(feat)
        feat_concat = torch.concat((sequence_output[:,0,:].view(-1,768), feat),-1)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(feat_concat) ## extract the 1st token's embeddingsp
        # linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
        linear1_output = self.act1(linear1_output)
        assert linear1_output[0].sum()>0.9 and linear1_output[0].sum()<1.1
        # linear2_output = self.linear2(linear1_output)
        # linear2_output = self.act2(linear2_output)
        # linear3_output = self.linear3(linear2_output)
        # linear3_output = self.act3(linear3_output)
        # linear4_output = self.linear4(linear3_output)
        # drop_output = self.drop(linear4_output)
        return linear1_output# loss, outputs

# tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = CustomBERTModel(mnli_dataset.num_labels) # You can pass the parameters if required to have more flexible model
model.to(device) ## can be gpu
optimizer = AdamW(model.parameters(), lr=2e-6, correct_bias=False)#original 2e-5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, mode='max', patience=4, min_lr=5e-7, verbose=True)#original factor=0.6, min_lr=5e-7


# ## define training regime

# In[15]:

def save_best(best_acc, val_acc, model, best_save_path):
  if val_acc>best_acc:
    torch.save(model.state_dict(), best_save_path)
    best_acc = val_acc
  return val_acc

### MODIFIED
import time
import traceback
import torch.nn.functional as F
from typing import Optional, Iterable, Dict, Any
from EarlyStopperUtil import MetricTracker


from sklearn.metrics import classification_report

EPOCHS = 100
best_save_path = 'prediction_best_'+lang+str(batch_size)+'X'+str(batches_per_epoch)+'.pt'

def train(model, train_loader, val_loader, optimizer, scheduler, rev_label_dict):  
  EarlyStopper = MetricTracker(patience=12, metric_name='+accuracy')

  for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    best_acc = 0

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

      ############new code#####################

      outputs = model(pair_token_ids, 
                            token_type_ids=seg_ids, 
                            attention_mask=mask_ids,
                            feat=feat)
      # outputs = F.log_softmax(outputs, dim=1) # log prob
      # outputs = np.argmax(prob, axis=1) # preds
      # https://stackoverflow.com/questions/43672047/convert-probability-vector-into-target-vector-in-python
      # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
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
      accuracies.append(acc)
      
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)

    val_acc, val_loss, cr, model, optimizer = evaluate_accuracy(model, optimizer, val_loader, rev_label_dict, label_dict, None)
    best_acc = save_best(best_acc, val_acc, model, best_save_path)
    EarlyStopper.add_metric(val_acc)
    if EarlyStopper.should_stop_early(): break

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print(f'train_size: {train_size}')


# In[41]:


import warnings
from sklearn.exceptions import DataConversionWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    train(model, train_loader, val_loader, optimizer, scheduler, rev_label_dict)


# In[16]:


save_path = 'prediction_'+lang+str(batch_size)+'X'+str(batches_per_epoch)+'.pt'
# torch.save(model.state_dict(), save_path)


# # test

# In[17]:

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


# model.load_state_dict(torch.load(save_path))
test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict, save_path)
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

print('\n\n---------------Best save path---------------------')
print('--------------------------------------------------')
model.load_state_dict(torch.load(best_save_path))
test_loss, test_acc = validate(model, test_loader, optimizer, rev_label_dict, label_dict, best_save_path)
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

print(lang)