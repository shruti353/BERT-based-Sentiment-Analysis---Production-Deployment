import mlflow
import torch
from datasets import load_dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from tokenizer import tokenizer
from config import *

dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

mlflow.set_experiment("bert-sentiment")

with mlflow.start_run():
    trainer.train()
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, "model")
