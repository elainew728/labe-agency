import pandas as pd
import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1_micro = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="micro")
    f1_weighted = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    return {'accuracy': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', default="bert-base-cased", help='Name of model.')
parser.add_argument('-c', '--checkpoint_dir', default=None, help='Path to checkpoint.')
parser.add_argument('-s', '--seed', default=0, help='Random Seed.')
parser.add_argument('-d', '--dataset_path', default="./lac_dataset_construction/lac_dataset", help='Path to dataset.')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)

data_files = {"train": "train.csv", "test": "test.csv", "valid": "validation.csv"}
dataset = load_dataset(args.dataset_path)

tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

if args.checkpoint_dir is not None:
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
training_args = TrainingArguments(output_dir="./checkpoints/{}_{}".format(args.model_name, str(args.seed)), 
                                  evaluation_strategy="epoch", 
                                  per_device_train_batch_size=6,
                                  num_train_epochs=10,
                                  learning_rate=5e-6)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
res = trainer.predict(test_dataset=test_dataset)
print('Result: ', res)