# train_depression_severity.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset, ClassLabel, load_metric

# ========== Config ==========
MODEL_NAME = "bert-base-uncased"   # swap for language-specific model if needed
CSV_PATH = "dataset.csv"           # path to the aggregated CSV (change as needed)
TEXT_COL = "text"
LABEL_COL = "label"
SEED = 42
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
OUTPUT_DIR = "depression_model"
MAX_LENGTH = 256
# ============================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) Load CSV
df = pd.read_csv(CSV_PATH)
# Basic cleaning: drop na and duplicates
df = df[[TEXT_COL, LABEL_COL]].dropna().drop_duplicates().reset_index(drop=True)

# 2) Map labels
label2id = {"Minimal":0, "Mild":1, "Moderate":2, "Severe":3}
id2label = {v:k for k,v in label2id.items()}
df["label_id"] = df[LABEL_COL].map(label2id)
assert df["label_id"].isnull().sum() == 0, "Found labels not in mapping!"

# 3) Stratified split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label_id"])
val_df, test_df  = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label_id"])

print("Sizes:", len(train_df), len(val_df), len(test_df))

# 4) Convert to HuggingFace datasets
def to_hf(ds):
    return Dataset.from_pandas(ds[[TEXT_COL, "label_id"]].rename(columns={TEXT_COL:"text","label_id":"label"}))

train_ds = to_hf(train_df)
val_ds = to_hf(val_df)
test_ds = to_hf(test_df)

# 5) Tokenizer + preprocessing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

# set format for PyTorch
columns_to_return = ["input_ids","attention_mask","label"]
train_ds.set_format(type="torch", columns=columns_to_return)
val_ds.set_format(type="torch", columns=columns_to_return)
test_ds.set_format(type="torch", columns=columns_to_return)

# 6) Compute class weights (optional but helpful)
labels = train_df["label_id"].values
class_counts = np.bincount(labels, minlength=len(label2id))
print("Class counts (train):", class_counts)
total = labels.shape[0]
class_weights = {i: total/(len(label2id)*count) for i, count in enumerate(class_counts)}
print("Class weights:", class_weights)
# convert to tensor for use in loss
weights_tensor = torch.tensor([class_weights[i] for i in range(len(label2id))]).to(torch.float)

# 7) Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Replace the default loss to use class weights by passing them to forward via Trainer compute_loss override.
# We'll implement a custom Trainer to apply class weights.
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("label")
        outputs = model(**{k: v for k, v in inputs.items() if k != "label"})
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 8) Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    report = classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))], zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1, "report": report}

# 9) Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=200,
    learning_rate=LR,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available()
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 10) Train
trainer.train()

# 11) Evaluate on test set
metrics = trainer.evaluate(test_dataset=test_ds)
print("Test metrics:", metrics)

# For a more readable classification report:
preds_output = trainer.predict(test_ds)
print(preds_output.metrics)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))

# 12) Save final model & tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved model to", OUTPUT_DIR)

# 13) Inference example
def infer_texts(texts):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    return [id2label[int(p)] for p in preds]

print(infer_texts(["I have been feeling hopeless and can't sleep.", "I am doing okay, enjoying my classes."]))
