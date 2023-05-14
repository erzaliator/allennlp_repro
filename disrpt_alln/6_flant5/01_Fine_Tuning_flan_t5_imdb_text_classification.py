import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

import pandas as pd
from datasets import Dataset


dataset_id = "imdb"
device = 'cuda:0'


from datasets import load_dataset

# Load dataset from the hub
dataset_train = load_dataset(dataset_id, split='train[:100]')
dataset_test = load_dataset(dataset_id, split='test[:100]')

print(f"Train dataset size: {len(dataset_train)}")
print(f"Test dataset size: {len(dataset_test)}")

print('Data preview:')
print(dataset_train)

print('Data item preview:')
print(dataset_train[1])

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id="google/flan-t5-small"

# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)


# before we can start training we need to preprocess our data. Abstractive flan t5 is good for a text2text-generation task. This means our model will take a text as input and generate a text as output. For this we want to understand how long our input and output will be to be able to efficiently batch our data. 
# - as result, we should to convert label from int to string



# dataset['train'] = dataset['train'].shuffle(seed=42).select(range(2000)) 
# dataset['test'] = dataset['test'].shuffle(seed=42).select(range(1000)) 

dataset_train = dataset_train.shuffle(seed=42)

train_df = pd.DataFrame(dataset_train)
test_df = pd.DataFrame(dataset_test)
del dataset_train
del dataset_test
train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
dataset_train = Dataset.from_pandas(train_df)
dataset_test = Dataset.from_pandas(test_df)

from datasets import concatenate_datasets

# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset_train, dataset_test]).map(lambda x: tokenizer(x["text"], truncation=True), batched=True, remove_columns=['text', 'label'])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset_train, dataset_test]).map(lambda x: tokenizer(x["label"], truncation=True), batched=True, remove_columns=['text', 'label'])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

del tokenized_inputs, tokenized_targets


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["text"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True, return_tensors="pt")

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["label"], max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    #shift 'input_ids', 'attention_mask', 'labels' to cuda device
    for key in model_inputs.keys():
        model_inputs[key] = torch.tensor(model_inputs[key]).to(device)
    return model_inputs

tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True, remove_columns=['text', 'label'])
tokenized_dataset_test = dataset_test.map(preprocess_function, batched=True, remove_columns=['text', 'label'])
print(f"Keys of tokenized dataset: {list(tokenized_dataset_train.features)}")
print(f"Keys of tokenized dataset: {list(tokenized_dataset_test.features)}")

# ## 3. Fine-tune and evaluate FLAN-T5
# 
# After we have processed our dataset, we can start training our model. Therefore we first need to load our [FLAN-T5](https://huggingface.co/models?search=flan-t5) from the Hugging Face Hub. In the example we are using a instance with a NVIDIA V100 meaning that we will fine-tune the `base` version of the model. 
# _I plan to do a follow-up post on how to fine-tune the `xxl` version of the model using Deepspeed._
# 

from transformers import AutoModelForSeq2SeqLM
import torch

# huggingface hub model id
model_id="google/flan-t5-small"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)
model = model.to(device)


# We want to evaluate our model during training. The `Trainer` supports evaluation during training by providing a `compute_metrics`.  
# The most commonly used metrics to evaluate summarization task is [rogue_score](https://en.wikipedia.org/wiki/ROUGE_(metric)) short for Recall-Oriented Understudy for Gisting Evaluation). This metric does not behave like the standard accuracy: it will compare a generated summary against a set of reference summaries
# 
# We are going to use `evaluate` library to evaluate the `rogue` score.

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

# Metric
metric = evaluate.load("f1")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# Before we can start training is to create a `DataCollator` that will take care of padding our inputs and labels. We will use the `DataCollatorForSeq2Seq` from the 🤗 Transformers library. 


from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


# The last step is to define the hyperparameters (`TrainingArguments`) we want to use for our training. We are leveraging the [Hugging Face Hub](https://huggingface.co/models) integration of the `Trainer` to automatically push our checkpoints, logs and metrics during training into a repository.


from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-imdb-text-classification"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=True,
    learning_rate=3e-4,

    num_train_epochs=2,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="epoch", 
    # logging_steps=1000,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    compute_metrics=compute_metrics,
)


# We can start our training by using the `train` method of the `Trainer`.
trainer.train()


# Nice, we have trained our model. 🎉 Lets run evaluate the best model again on the test set.
trainer.evaluate()


# ## 4. Run Inference and Classification Report
from tqdm.auto import tqdm

samples_number = len(dataset_test)
progress_bar = tqdm(range(samples_number))
predictions_list = []
labels_list = []
for i in range(samples_number):
  text = dataset_test['text'][i]
  inputs = tokenizer.encode_plus(text, padding='max_length', max_length=512, return_tensors='pt').to(device)
  outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
  prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
  predictions_list.append(prediction)
  labels_list.append(dataset_test['label'][i])

  progress_bar.update(1)

str_labels_list = []
for i in range(len(labels_list)): str_labels_list.append(str(labels_list[i]))


from sklearn.metrics import classification_report

report = classification_report(str_labels_list, predictions_list, zero_division=0)
print(report)