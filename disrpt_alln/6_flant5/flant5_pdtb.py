'''reads pdtb data and trains flant5 on it'''
# TODO: change code between autoregressive and autoencoder

import sys
import time
sys.path.append('../5_madx/utils')

from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration 
import datasets
import pandas as pd
from preprocessing import read_df_custom
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch import Tensor, mean, nn
import torch

MODEL_NAME = 'google/flan-t5-small'
SAVE_MODEL_DIR = './'
device = 'cuda:0'

# Load the SNLI dataset from Hugging Face Datasets
dataset = datasets.load_dataset("snli")

# Load the PDTB text classification data from a CSV file
df = read_df_custom("../../processed/eng.pdtb.pdtb_train_enriched.rels")[:1000]

# Preprocess the data
df['sentence_pair'] = df['unit1_txt']+df['unit2_txt']
sentences = list(df['sentence_pair'])
labels = list(df['label'])
num_labels = len(set(labels))
#TODO: add <entailment> and other labels from train, test and dev

# Split the data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Load the T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepend the prompt to the input sentences
prompt = "classify: "
train_sentences = [prompt + sentence for sentence in train_sentences]
val_sentences = [prompt + sentence for sentence in val_sentences]


# encode the sentences and labels
train_encodings = tokenizer(train_sentences, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
print(train_labels)
train_labels = tokenizer(train_labels, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
val_encodings = tokenizer(val_sentences, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
val_labels = tokenizer(val_labels, truncation=True, padding="max_length", max_length=128, return_tensors='pt')

# # Convert the labels to numerical values
# def get_label_dict(labels):
#     label_dict = {}
#     for (i, label) in enumerate(set(labels)):
#         label_dict[label] = i
#     return label_dict

# label_dict = get_label_dict(labels)
# train_labels = [label_dict[label] for label in train_labels]
# val_labels = [label_dict[label] for label in val_labels]

# Create the huggingface datasets
class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings['input_ids'].to(device)
        self.attention_masks = encodings['attention_mask'].to(device)
        self.labels_input_ids = labels['input_ids'].to(device)
        self.labels_attention_masks = labels['attention_mask'].to(device)
        print(labels)
        print(labels['input_ids'])
        exit(0)

        self.len = len(labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels_input_ids[idx], self.labels_attention_masks[idx]
    
    def __len__(self):
        return self.len

train_dataset = T5Dataset(train_encodings, train_labels)
val_dataset = T5Dataset(val_encodings, val_labels)

#convert huggingface dataset to pytorch dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
   
# Define the T5 model architecture
class T5Classifier(nn.Module):
    num_labels = num_labels

    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, num_labels=self.num_labels, torch_dtype=torch.float16)
        self.t5 = self.t5.to(device)

    def forward(self, input_ids, attention_masks, label_input_ids, label_attention_masks):
        outputs = self.t5(
                            input_ids=input_ids, 
                            attention_mask=attention_masks,
                            labels=label_input_ids,
                            decoder_attention_mask=label_attention_masks)
        loss = outputs.loss
        logits = outputs.logits
        #loss, prediction_scores = outputs[:2] #loss and logits
        return loss, logits
    
    def generate(self, input_ids, attention_mask, max_length=3):
        generated_ids = self.t5.generate(
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_length=max_length
                                        )
        return generated_ids

# Create the model instance
model = T5Classifier()
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Define the loss function
def cross_entropy_loss(logits, labels):
    return nn.log_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

# Define the metrics for evaluation
def accuracy(models, label_input_ids, label_attention_masks):
    generated_ids = model.generate(
            input_ids=label_input_ids,
            attention_mask=label_attention_masks,
            max_length=3
            )

    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in label_input_ids]
    return accuracy_score(target, preds)
    
# define train function
def train(model, train_dataset, val_loader, optimizer, scheduler):
  best_val_acc = 0

  for epoch in range(2):
    print('learning epoch: ', epoch+1, '...')
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    
    # logging for scheduler
    losses = []
    accuracies= []

    for batch_idx, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(train_loader):
      optimizer.zero_grad()
      loss, logits = model(input_ids, attention_masks, label_input_ids, label_attention_masks)
      acc = accuracy(logits, label_input_ids, label_attention_masks)
      criterion = nn.CrossEntropyLoss()
      loss.backward()
      optimizer.step()
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      losses.append(loss)
      accuracies.append(acc)
      
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)

    train_acc  = total_train_acc/len(train_dataset)
    train_loss = total_train_loss/len(train_dataset)

    val_acc, val_loss = evaluate_accuracy(model, val_loader)
    if val_acc>best_val_acc:
      torch.save(model.state_dict(), SAVE_MODEL_DIR+'best.pt')
      best_val_acc = val_acc
      print(f'Epoch {epoch+1}: Best val_acc: {best_val_acc:.4f}')
    if val_acc>=best_val_acc:
    #   torch.save(model.state_dict(), SAVE_MODEL_DIR+'best_latest.pt')
      best_val_acc = val_acc
      print(f'Epoch {epoch+1}: Best val_acc: {best_val_acc:.4f}')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

# Define the evaluation function
def evaluate_accuracy(model, val_loader, verbose=True):
    model.eval()
    total_val_loss = 0
    total_val_acc  = 0
    losses = []
    accuracies= []
    with torch.no_grad():
        for batch_idx, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(val_loader):
            loss, logits = model(input_ids, attention_masks, label_input_ids, label_attention_masks)
            criterion = nn.CrossEntropyLoss()
            acc = accuracy(logits, label_input_ids, label_attention_masks)
            total_val_loss += loss.item()
            total_val_acc  += acc.item()
        
            losses.append(loss)
            accuracies.append(acc)
            val_acc  = total_val_acc/len(val_loader)
            val_loss = total_val_loss/len(val_loader)
            if verbose:
                print(f'val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    return val_acc, val_loss

# Train the model
train(model, train_dataset, val_loader, optimizer, scheduler)   