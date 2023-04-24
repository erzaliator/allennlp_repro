#!/usr/bin/env python
# coding: utf-8


'''
See how gains can be achieved via MTL using Mad-X language adn task adapters. Run across disrpt tasks.
Usage: nohup python 01_madx.py --lang2_index=7 --cuda=7 > logs_runtime/runtime_7.out 2> logs_runtime/runtime_7.err &
'''

import argparse
parser = argparse.ArgumentParser()

# Raw Data
parser.add_argument("--lang2_index", type=int, default=11,
                    help="lang2 is the langauge which is finally tested. Index of language looked up from lang_list")
parser.add_argument("--cuda", type=str, default='0',
                    help="cuda")
args = parser.parse_args()


# TUNABLE PARAMS

# environ
import os
import torch
import wandb
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

# seeding params
SEED = 50


lang_list = [
'deu.rst.pcc', 
'eng.pdtb.pdtb',
'eng.rst.gum', 
'eng.rst.rstdt', 
'eng.sdrt.stac', 
'fas.rst.prstc', 
'fra.sdrt.annodis', 
'nld.rst.nldt', 
'por.rst.cstn', 
'rus.rst.rrt', 
'spa.rst.rststb', 
'spa.rst.sctb', 
'tur.pdtb.tdb', 
'zho.rst.sctb']

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
dataset2 = lang_list[args.lang2_index]

for dataset1 in lang_list:
    if dataset1==dataset2: continue

    # model params
    BERT_MODEL = 'bert-base-multilingual-cased'
    batch_size1 = 32
    batch_size2 = 32
    epoch1 = 12
    epoch2 = 12
    lr=2e-5

    # input data params
    path1 = '../../processed/'+dataset1+'_'
    path2 = '../../processed/'+dataset2+'_'
    lang1 = adapter_lang_map[dataset1.split('.')[0]]
    lang2 = adapter_lang_map[dataset2.split('.')[0]]

    # naming params
    experiment_name = 'Mad'
    lrfunc = 'lrfunc=Adafactor_'+'lr='+str(lr)
    model_name = 'plm='+BERT_MODEL
    additional_info = 'FullShot=v4'
    name = '_'.join([additional_info, dataset1, dataset2, lrfunc, model_name])

    # output data params
    MODEL_DIR = 'runs/full_shot/'+name
    print('-------------------------------------------------------------------')
    print('Lang1: ', dataset1, '   Lang2: ', dataset2)
    print('Saving run to: ', MODEL_DIR)


    # WANDB
    wandb.init(reinit=True)
    wandb.run.name = name


    # LOGGING
    from utils.logger import Logger
    logger = Logger(MODEL_DIR, wandb, wandb_flag=True)
    print('Running with params: BERT_MODEL='+ BERT_MODEL+ ' lr='+ str(lr))


    ## Dataset Preprocessing


    #load data
    from utils.preprocessing import construct_dataset
    train_path = path1+'train_enriched.rels'
    valid_path = path1+'dev_enriched.rels'
    test_path = path1+'test_enriched.rels'
    train_dataset_df1, test_dataset_df1, valid_dataset_df1 = construct_dataset(train_path, test_path, valid_path, logger)


    train_path = path2+'train_enriched.rels'
    valid_path = path2+'dev_enriched.rels'
    test_path = path2+'test_enriched.rels'
    train_dataset_df2, test_dataset_df2, valid_dataset_df2 = construct_dataset(train_path, test_path, valid_path, logger)


    # preprocess data
    from utils.iterators import get_iterators
    train_dataset1, valid_dataset1, test_dataset1, labels1 = get_iterators(train_dataset_df1, test_dataset_df1, valid_dataset_df1, batch_size1, BERT_MODEL, logger)
    train_dataset2, valid_dataset2, test_dataset2, labels2 = get_iterators(train_dataset_df2, test_dataset_df2, valid_dataset_df2, batch_size2, BERT_MODEL, logger)
    labels_union = list(set(labels1.names) | set(labels2.names))



    ## Task Adapter Configs


    #load configs
    from transformers import AutoConfig, AutoAdapterModel
    from transformers import AdapterConfig


    config = AutoConfig.from_pretrained(
        BERT_MODEL,
    )
    model = AutoAdapterModel.from_pretrained(
        BERT_MODEL,
        config=config,
    )

    # Load the language adapters
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    model.load_adapter(lang1+"/wiki@ukp", config=lang_adapter_config)
    model.load_adapter(lang2+"/wiki@ukp", config=lang_adapter_config)

    # Add a new task adapter
    model.add_adapter("disrpt")

    # Add a classification head for our target task
    num_labels=len(labels_union)
    print('Total prediction labels: ', num_labels)
    model.add_classification_head("disrpt", num_labels=num_labels)


    #set trainable adapter
    model.train_adapter(["disrpt"])


    # Unfreeze and activate stack setup
    from transformers.adapters.composition import Stack
    lang = lang1
    model.active_adapters = Stack(lang, "disrpt")
    lang = dataset1

    print(model)




    ## train setup


    # callback

    from copy import deepcopy
    from transformers import TrainerCallback

    class CustomCallback(TrainerCallback):
        
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train@"+lang)
                return control_copy


    # CM
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from torch.nn import CrossEntropyLoss
    import numpy as np
    import os
        
    def compute_metrics(pred):
        global num_labels, labels1, labels2, labels_union
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # confusion matrix
        class_names = labels_union # labels1 if lang==dataset1 else labels2
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=class_names)})#._int2str)})
        labels_names = [class_names[x.item()] for x in labels]
        preds_names = [class_names[x.item()] for x in preds]
        # log predictions
        output_predict_folder = MODEL_DIR+'_'+lang
        if not os.path.exists(output_predict_folder): os.makedirs(output_predict_folder)
        output_predict_file = os.path.join(output_predict_folder, "predictions.txt")
        with open(output_predict_file, "w") as writer:
            writer.write(str({'prefix': lang, 'labels': labels_names, 'preds': preds_names}))
        wandb.save(os.path.join(output_predict_file))

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        loss_fct = CrossEntropyLoss()
        logits = torch.tensor(pred.predictions)
        labels = torch.tensor(labels)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        with open(os.path.join(output_predict_folder, "metrics.json"), "w") as writer:
            writer.write(str({'prefix': lang, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}))

        return {
            'accuracy@'+lang: acc,
            'f1@'+lang: f1,
            'precision@'+lang: precision,
            'recall@'+lang: recall,
            'loss@'+lang: loss,
        }


    # train args
    from transformers import TrainingArguments, AdapterTrainer

    training_args = TrainingArguments(
        seed = SEED,
        evaluation_strategy="epoch",
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
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset1,
        eval_dataset=valid_dataset1,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(CustomCallback(trainer)) 





    ## Train

    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train_log", metrics)
    trainer.save_metrics("train_log", metrics)


    # evaluate
    lang = dataset1
    trainer.evaluate(metric_key_prefix='test_'+lang,
                    eval_dataset=test_dataset1)


    lang = dataset2+'('+dataset1+')'
    trainer.evaluate(metric_key_prefix='test_'+lang,
                    eval_dataset=test_dataset2)









    ## Cross-lingual transfer

    lang = lang2
    model.active_adapters = Stack(lang, "disrpt")
    lang = dataset2


    # train args


    training_args = TrainingArguments(
        seed = SEED,
        evaluation_strategy="epoch",
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

    trainer.add_callback(CustomCallback(trainer)) 






    ## Train

    trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train_logger", metrics)
    trainer.save_metrics("train_logger", metrics)



    # evaluate

    trainer.evaluate(metric_key_prefix='test_'+lang,
                    eval_dataset=test_dataset2)


    lang = dataset1+'('+dataset2+')'
    trainer.evaluate(metric_key_prefix='test_'+lang,
                    eval_dataset=test_dataset1)
    