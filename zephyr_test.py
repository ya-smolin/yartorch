# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate
import torch
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TrainingArguments
import numpy as np
import evaluate

print("DATASET_LOADING...")
huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)
N = 100
dataset_orig = load_dataset("yelp_review_full")
dataset = dataset_orig.select(range(N))
print("loaded dataset. Example:\n", dataset["train"][55])
tokenizer = pipe.tokenizer
tokenize_function = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets


print("MODEL INIT...")
model_path = "/home/sysgen/Pycharm/wotspace/text-generation-webui/models/HuggingFaceH4_zephyr-7b-beta"
pipe = pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map="auto")

model = pipe.model
config = AutoConfig.from_pretrained(model_path, num_labels=5)
# You need to ensure that the base model extracted from the pipeline is compatible with sequence classification
sequence_classification_model = AutoModelForSequenceClassification.from_config(config)
sequence_classification_model.base_model = model.base_model
print("MODEL INITED!")

tasets["test"]
print("DATASET_LOADED!")

print("TRAINING INIT...")
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    fp16=True,
    gradient_accumulation_steps=1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
print("TRAINING INITED!")

print("TRAINING...")
trainer.train()
