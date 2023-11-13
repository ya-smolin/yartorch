# useful examples https://huggingface.co/TheBloke/zephyr-7B-beta-GPTQ
import json
import time

import pandas as pd
import torch
from auto_gptq.quantization import gptq
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig, Trainer, TrainingArguments, \
    AutoModelForSeq2SeqLM, GPTQConfig

# model_path = "/home/sysgen/Pycharm/wotspace/text-generation-webui/models/HuggingFaceH4_zephyr-7b-beta"

model_path = "TheBloke/zephyr-7B-beta-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"
print("TOKENIZER:", tokenizer.eos_token)



with open("/home/sysgen/Downloads/result_cleaned_daily.json", "r") as f:
    dataset_d = json.load(f)
instruction="Below is a conversation between a user and you. Instruction: Write a response appropriate to the conversation."
dataset_raw = []
for _, dialog_date in dataset_d.items():
    chat = [
        {"role": "system", "content": f"{instruction}"},
    ]
    for d in dialog_date:
        chat.append({"role": "user", "content": d['A']})
        chat.append({"role": "assistant", "content": d['Y']})
    dialog_str = tokenizer.apply_chat_template(chat, tokenize=False)
    dataset_raw.append(dialog_str)
df = pd.DataFrame({"dialogues" : dataset_raw})

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)
train_dataset = dataset

# # Split the dataset into train and validation sets
# # Assuming you want an 80-20 split
# train_test_split = dataset.train_test_split(test_size=0.2)
#
# # Extract the train and validation datasets
# train_dataset = train_test_split['train']
# validation_dataset = train_test_split['test']
#
# # Now you have separate datasets for training and validation
# print("Train Dataset:", train_dataset)
# print("Validation Dataset:", validation_dataset)

from peft import prepare_model_for_kbit_training, PeftModel

quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             quantization_config=quantization_config_loading,
                                             trust_remote_code=False,
                                             use_flash_attention_2=True,
                                             revision="main")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)



output_dir = f'./zephyr_telegram-{str(int(time.time()))}'

from peft import LoraConfig, get_peft_model


config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj", "o_proj", "q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#model = get_peft_model(model, config)
model = PeftModel.from_pretrained(model, "/home/sysgen/Pycharm/wotspace/yartorch/zephyr_telegram-1699896435/checkpoint-840",  is_trainable=True, use_flash_attention_2=True)

print("CUDA:", torch.cuda.is_available())

query="""
Та мені купа всього треба
Сумно
"""

if IS_INFERENCE := True:
    prompt = f"<|system|>\n{instruction}</s>\n<|user|>\n{query}</s>\n<|assistant|>"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    print(tokenizer.decode(output[0]))

model.print_trainable_parameters()
max_size = 4096
train_dataset = train_dataset.map(lambda samples: tokenizer(samples["dialogues"],  max_length=max_size, truncation=True), batched=True)
#validation_dataset = validation_dataset.map(lambda samples: tokenizer(samples["dialogues"],  max_length=max_size, truncation=True,  padding="max_length"), batched=True)

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

#eval_dataset = validation_dataset,
#evaluation_strategy="steps",
#eval_steps=10,

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,

    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_steps=3000,
        learning_rate=9e-5, #
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="adamw_hf",
        save_steps=30,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


