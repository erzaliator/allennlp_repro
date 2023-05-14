import torch
from datasets import load_dataset#, load_metric
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load 1% of SNLI dataset
train_dataset = load_dataset("snli", split="train[:1000]")

#Load test and validation sets
validation_dataset = load_dataset("snli", split="validation[:100]")
test_dataset = load_dataset("snli", split="test[:100]")

MODEL_NAME = 'google/flan-t5-small'

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding='max_length', max_length=512)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load the T5ForConditionalGeneration model
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

# Define the metric for evaluation
metric = evaluate.load("accuracy")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    compute_metrics=lambda pred: metric.compute(predictions=pred.predictions.argmax(axis=1), references=pred.label_ids),
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
evaluation = trainer.evaluate(tokenized_test_dataset)
print(evaluation)
