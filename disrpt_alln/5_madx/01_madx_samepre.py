#!/usr/bin/env python
# coding: utf-8


'''
See how gains can be achieved via MTL using Mad-X language adn task adapters. Run across disrpt tasks.
Usage: nohup python 01_madx.py --lang2_index=7 --cuda=7 > logs_runtime/runtime_7.out 2> logs_runtime/runtime_7.err &
'''

import pdb

import wandb
import torch
import os
import numpy as np
import pandas as pd
from copy import deepcopy


from utils.logger import Logger
from utils.preprocessing import construct_dataset
from utils.iterators import get_iterators

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import TrainerCallback, EarlyStoppingCallback

from transformers.adapters.composition import Stack
from transformers import TrainingArguments, AdapterTrainer  # , BertAdapterModel

import argparse
parser = argparse.ArgumentParser()

# Raw Data
parser.add_argument("--lang1_index", type=int, default=11,
                    help="lang1 is the langauge which is used for pretraining. Index of language looked up from lang_list")
parser.add_argument("--cuda", type=str, default='0',
                    help="cuda")
args = parser.parse_args()


# TUNABLE PARAMS

# environ
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

# seeding params
SEED = 50


lang_list = [
    'deu.rst.pcc',
    'eng.pdtb.pdtb',
    'eng.rst.gum',
    'eng.rst.rstdt', #REDO
    'eng.sdrt.stac',
    'fas.rst.prstc',
    'fra.sdrt.annodis', #REDO
    'nld.rst.nldt', #DONE maybe gains wont look realistic. but if the run never existed then how did the finetuning happen?
    'por.rst.cstn', #DONE
    'rus.rst.rrt', #DONE
    'spa.rst.rststb', #DONE
    'spa.rst.sctb', #DONE
    'tur.pdtb.tdb', #DONE
    'zho.rst.sctb'] #DONE

adapter_lang_map = {
    'deu': 'de',
    'eng': 'en',
    'fas': 'ar',
    'fra': 'fr',
    'nld': 'de',
    'por': 'pt',
    'rus': 'ru',
    'spa': 'es',
    'tur': 'tr',
    'zho': 'zh'
}

# model params
BERT_MODEL = 'bert-base-multilingual-cased'
batch_size1 = 50#32
batch_size2 = 50#32
epoch1 = 2#35
epoch2 = 2#35
lr = 1e-4
early_stop = EarlyStoppingCallback(7, 2.0)
model = None
print('Running with params: BERT_MODEL=' + BERT_MODEL + ' lr=' + str(lr))

# input data params
dataset_folder = '../../processed/'
dataset1 = lang_list[args.lang1_index]
path1 = dataset_folder+dataset1+'_'
lang1 = adapter_lang_map[dataset1.split('.')[0]]


# naming params
experiment_name = 'Mad_checking'
lrfunc = 'lrfunc=Adafactor_'+'lr='+str(lr)
model_name = 'plm='+BERT_MODEL
additional_info = 'CheckFullShot=v10'
name = '_'.join([additional_info, 'pretrain', dataset1, lrfunc, model_name])


# output data params
MODEL_DIR = 'runs/full_shot/same_pretrain_lr1e-4/'+name
print('-------------------------------------------------------------------')
print('Lang1: ', dataset1)
print('Saving run to (pretrain stage): ', MODEL_DIR)
save_best_model_path_pretrain = MODEL_DIR+'_'+dataset1+"/best_acc"
save_best_model_path = save_best_model_path_pretrain
print('Will save best model to: ', save_best_model_path_pretrain)

# WANDB
wandb.init(reinit=True)
wandb.run.name = name

# LOGGING
logger = Logger(MODEL_DIR, wandb, wandb_flag=True)

# Dataset Preprocessing

# load data
train_path = path1+'train_enriched.rels'
valid_path = path1+'dev_enriched.rels'
test_path = path1+'test_enriched.rels'
train_dataset_df1, test_dataset_df1, valid_dataset_df1 = construct_dataset(
    train_path, test_path, valid_path, logger)

# preprocess data
train_dataset1, valid_dataset1, test_dataset1, labels1 = get_iterators(
    train_dataset_df1, test_dataset_df1, valid_dataset_df1, batch_size1, BERT_MODEL, logger)




# Task Adapter Configs

# load configs
from transformers import AutoConfig, AutoAdapterModel
from transformers import AdapterConfig

config = AutoConfig.from_pretrained(
    BERT_MODEL,
)
model = AutoAdapterModel.from_pretrained(
    BERT_MODEL,
    config=config,
)

# Load the language adapter
lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
model.load_adapter(lang1+"/wiki@ukp", config=lang_adapter_config)

# Add a new task adapter
model.add_adapter("disrpt")

# Add a classification head for our target task
num_labels = len(set(labels1.names))
head_name = "disrpt-"+dataset1.replace('.', '-')
print('Total prediction labels: ', num_labels)
model.add_classification_head(head_name, num_labels=num_labels)

# set trainable adapter
model.train_adapter(["disrpt"])

# Unfreeze and activate stack setup
lang = lang1
model.active_adapters = Stack(lang, "disrpt")
model.active_head = head_name
lang = dataset1


# print(model)
# bert.encoder.layer.0.attention.output.dense.weight
# bert.encoder.layer.0.attention.output.dense.bias
# bert.encoder.layer.0.attention.output.LayerNorm.weight
# bert.encoder.layer.0.attention.output.LayerNorm.bias
# bert.encoder.layer.0.intermediate.dense.weight
# bert.encoder.layer.0.intermediate.dense.bias
# bert.encoder.layer.0.output.dense.weight
# bert.encoder.layer.0.output.dense.bias
# bert.encoder.layer.0.output.LayerNorm.weight
# bert.encoder.layer.0.output.LayerNorm.bias

# train setup

# callback

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        self.best_acc = -1


    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            global model, save_best_model_path, lang
            print('USING HEAD: ', model.active_head)
            control_copy = deepcopy(control)
            output_metrics = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train@"+lang)
            print(output_metrics)
            print('trying...', 'train@'+lang+'_accuracy@'+lang)
            acc = output_metrics['train@'+lang+'_accuracy@'+lang]
            if state.global_step < state.max_steps and self.best_acc<=acc:
                print('Saving the model using CustomCallback: ', save_best_model_path)
                model.save_pretrained(save_best_model_path, from_pt=True)
                self.best_acc = acc
            return control_copy

class GradientNormCallback(TrainerCallback):
    def __init__(self, model, layer_names, lang, trainingstage):
        self.model = model
        self.layer_names = layer_names
        self.gradient_norms = {}

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        pass
        # Compute the L2 norm of gradients for each layer
        # pdb.set_trace()

        # for layer_name, layer_params in self.model.named_parameters():
        #     if any(layer_name.startswith(prefix) for prefix in self.layer_names):
        #         if layer_params.grad is not None:
        #             grad_norm = torch.norm(layer_params.grad.data, p=2).item()
        #             if layer_name not in self.gradient_norms:
        #                 self.gradient_norms[layer_name] = []
        #             self.gradient_norms[layer_name].append(grad_norm)
        #         else:
        #             if layer_name not in self.gradient_norms:
        #                 self.gradient_norms[layer_name] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        for layer_name, layer_params in self.model.named_parameters():
            if any(layer_name.startswith(prefix) for prefix in self.layer_names):
                if layer_params.grad is not None:
                    grad_norm = torch.norm(layer_params.grad.data, p=2).item()
                    if layer_name not in self.gradient_norms:
                        self.gradient_norms[layer_name] = []
                    self.gradient_norms[layer_name].append(grad_norm)
                else:
                    if layer_name not in self.gradient_norms:
                        self.gradient_norms[layer_name] = []

        epoch = state.epoch
        # Save the gradient norms to separate CSV files for each layer with the epoch number
        for layer_name, norms in self.gradient_norms.items():
            df = pd.DataFrame(norms, columns=['Gradient Norm'])
            filename = f'{layer_name}_e{epoch}_norm.csv'
            filename = os.path.join(save_best_model_path_pretrain, filename)
            df.to_csv(filename, index=False)

# CM
def compute_metrics(pred):
    global num_labels, labels1, labels2, model, best_acc
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # confusion matrix
    class_names = labels1.names if lang==dataset1 else labels2.names
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(
        probs=None, y_true=labels, preds=preds, class_names=class_names)})  # ._int2str)})
    labels_names = [class_names[x.item()] for x in labels]
    preds_names = [class_names[x.item()] for x in preds]
    # log predictions
    output_predict_folder = MODEL_DIR+'_'+lang
    if not os.path.exists(output_predict_folder):
        os.makedirs(output_predict_folder)
    output_predict_file = os.path.join(
        output_predict_folder, "predictions.txt")
    with open(output_predict_file, "w") as writer:
        writer.write(
            str({'prefix': lang, 'labels': labels_names, 'preds': preds_names}))
    wandb.save(os.path.join(output_predict_file))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    loss_fct = CrossEntropyLoss()
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(labels)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

    with open(os.path.join(output_predict_folder, "metrics.json"), "w") as writer:
        writer.write(str({'prefix': lang, 'acc': acc,
                        'precision': precision, 'recall': recall, 'f1': f1}))

    return {
        'accuracy@'+lang: acc,
        'f1@'+lang: f1,
        'precision@'+lang: precision,
        'recall@'+lang: recall,
        'loss@'+lang: loss,
    }

# train args
training_args = TrainingArguments(
    seed=SEED,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=lr,
    num_train_epochs=epoch1,
    per_device_train_batch_size=batch_size1,
    per_device_eval_batch_size=batch_size1,
    output_dir=MODEL_DIR+'_'+lang,
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
    save_total_limit=1,
    load_best_model_at_end=True,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset1,
    eval_dataset=valid_dataset1,
    compute_metrics=compute_metrics
)

layer_names = ['bert.encoder.layer.1', 'bert.encoder.layer.6', 'bert.encoder.layer.11', 'bert.pooler', 'heads.disrpt-deu-rst-pcc.1']
grad_norm_callback_obj = GradientNormCallback(model, layer_names, lang, 'pretrain')

trainer.add_callback(CustomCallback(trainer))
trainer.add_callback(grad_norm_callback_obj)
trainer.add_callback(early_stop)

# Train

train_result = trainer.train()

metrics = train_result.metrics
trainer.log_metrics("train_log", metrics)
trainer.save_metrics("train_log", metrics)

# evaluate
lang = dataset1
trainer.evaluate(metric_key_prefix='test_'+lang,
                    eval_dataset=test_dataset1)


exit(0)


for dataset2 in lang_list:
    if dataset1 == dataset2:
        continue

    # delete previous model
    del model


    # input data params
    path2 = dataset_folder+dataset2+'_'
    lang2 = adapter_lang_map[dataset2.split('.')[0]]

    # naming params
    name = '_'.join([additional_info, 'finetune', dataset1, dataset2, lrfunc, model_name])

    # output data params
    MODEL_DIR = 'runs/full_shot/same_pretrain_lr1e-4/'+name
    print('-------------------------------------------------------------------')
    print('Lang1: ', dataset1, '   Lang2: ', dataset2)
    print('Saving run to: ', MODEL_DIR)
    save_best_model_path_finetune = MODEL_DIR+'_'+dataset2+"/best_acc"
    save_best_model_path = save_best_model_path_finetune
    print('Will save best model to (finetune stage): ', save_best_model_path)

    # WANDB
    wandb.init(reinit=True)
    wandb.run.name = name

    # LOGGING
    logger = Logger(MODEL_DIR, wandb, wandb_flag=True)

    # Dataset Preprocessing

    # load data
    train_path = path2+'train_enriched.rels'
    valid_path = path2+'dev_enriched.rels'
    test_path = path2+'test_enriched.rels'
    train_dataset_df2, test_dataset_df2, valid_dataset_df2 = construct_dataset(
        train_path, test_path, valid_path, logger)

    # preprocess data
    train_dataset2, valid_dataset2, test_dataset2, labels2 = get_iterators(
        train_dataset_df2, test_dataset_df2, valid_dataset_df2, batch_size2, BERT_MODEL, logger)

    # load pretrained model
    model = AutoAdapterModel.from_pretrained(save_best_model_path_pretrain).to('cuda')

    # Load the language adapter
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    model._name_or_path = BERT_MODEL
    model.load_adapter(config=lang_adapter_config, adapter_name_or_path=lang2+"/wiki@ukp", model_name=BERT_MODEL)

    # Add a classification head for our target task
    num_labels = len(set(labels2.names))
    head_name = "disrpt-"+dataset2.replace('.', '-')
    print('Total prediction labels: ', num_labels)
    print('BEFORE: ', model.active_head)
    model.add_classification_head(head_name, num_labels=num_labels)
    print('AFTER: ', model.active_head)
    
    # set trainable adapter
    model.train_adapter(["disrpt"])
    
    # Cross-lingual transfer
    lang = lang2
    model.active_adapters = Stack(lang, "disrpt")
    lang = dataset2
    print('AFTER ADAPTER: ', model.active_head)

    # print(model)

    # train args

    training_args = TrainingArguments(
        seed=SEED,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        num_train_epochs=epoch2,
        per_device_train_batch_size=batch_size2,
        per_device_eval_batch_size=batch_size2,
        output_dir=MODEL_DIR+'_'+lang,
        overwrite_output_dir=False,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        save_total_limit=1,
        # resume_from_checkpoint=MODEL_DIR+'/last-checkpoint',
    )


    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset2,
        eval_dataset=valid_dataset2,
        compute_metrics=compute_metrics
    )

    callback_obj = CustomCallback(trainer)
    trainer.add_callback(callback_obj)

    # Train

    train_result = trainer.train()

    # load best model and activate heads and adapters
    model = AutoAdapterModel.from_pretrained(save_best_model_path).to('cuda')

    lang = lang2
    model.active_adapters = Stack(lang, "disrpt")
    model.active_head = head_name
    lang = dataset2
    print('AFTER BEST EVAL: ', model.active_head)

    metrics = train_result.metrics
    trainer.log_metrics("train_logger", metrics)
    trainer.save_metrics("train_logger", metrics)

    # evaluate

    trainer.evaluate(metric_key_prefix='test_'+lang,
                     eval_dataset=test_dataset2)
    
    del callback_obj