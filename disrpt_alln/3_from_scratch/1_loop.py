import os
import torch 
import wandb
import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=arg2
device = torch.device('cuda:0')


# MACROS

SEED = 42
lm_list = arg1.split()
# deepset/gbert-base deepset/gbert-large deepset/gelectra-base deepset/gelectra-large deepset/gelectra-base-generator deepset/gelectra-large-generator
# lm_list = ['bert-base-german-cased', 'deepset/gbert-base', 'deepset/gbert-large', 'deepset/gelectra-base', 'deepset/gelectra-large', 'deepset/gelectra-base-generator', 'deepset/gelectra-large-generator']

for bert_model in lm_list:
    BERT_MODEL = bert_model
    BATCH_SIZE = 8

    experiment_name = 'ModelBases'
    lr = 'lr=AdafactorDefault'
    model_name = 'plm='+BERT_MODEL
    additional_info = 'Info=v3_lrtest1'
    name = '_'.join([experiment_name, lr, model_name, additional_info])

    MODEL_DIR = name


    
    #WANDB
    wandb.init(reinit=True)
    wandb.run.name = name





    # LOGGING
    from utils.logger import Logger
    logger = Logger(MODEL_DIR, wandb, wandb_flag=True)
    print('Running with params: BERT_MODEL='+ BERT_MODEL+ ' lr='+ str(lr))


    # READ DATASET

    from utils.preprocessing import construct_dataset
    train_path  = '../../processed/deu.rst.pcc_train_enriched.rels'
    test_path = '../../processed/deu.rst.pcc_test_enriched.rels'
    valid_path = '../../processed/deu.rst.pcc_dev_enriched.rels'


    train_dataset, test_dataset, valid_dataset = construct_dataset(train_path, test_path, valid_path, logger)




    # CONSTRUCT ITERATORS

    from utils.iterators import get_iterators
    train_iter, valid_iter, test_iter = get_iterators(train_dataset, test_dataset, valid_dataset, BATCH_SIZE, BERT_MODEL, logger)




    # MODEL AND OPTIM
    from transformers import AutoModelForSequenceClassification
    from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
    from torch import optim
    # import pytorch_lightning_spells as pls


    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(list(set(train_dataset['label'])|set(test_dataset['label'])|set(valid_dataset['label']))))
    # optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # optimizer = Adafactor(model.parameters())#, relative_step=False, warmup_init=False, lr=1e-5, scale_parameter=False)
    # print(dir(optimizer))
    # lr_scheduler = (optimizer)
    



    # EVALUATION
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    #TRAINING ARGS

    from copy import deepcopy
    from transformers import TrainerCallback

    class CustomCallback(TrainerCallback):
        
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                return control_copy



    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(output_dir=MODEL_DIR, 
                                    evaluation_strategy="epoch",
                                    per_device_train_batch_size =BATCH_SIZE,
                                    per_device_eval_batch_size=BATCH_SIZE,
                                    num_train_epochs=40,
                                    save_total_limit=1,
                                    learning_rate=3e-6,
                                    weight_decay=0.01,
                                    logging_steps=1,
                                    metric_for_best_model = 'acc')






    #TRAINER

    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_iter,
        eval_dataset=valid_iter,
        compute_metrics=compute_metrics,
        # optimizers=[optimizer, lr_scheduler],
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=12)]
    )

    trainer.add_callback(CustomCallback(trainer)) 





    # TRAIN

    train_result = trainer.train() 

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


    # EVALUATE

    metrics = trainer.evaluate(test_iter)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    