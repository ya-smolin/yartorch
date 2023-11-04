#useful examples https://huggingface.co/TheBloke/zephyr-7B-beta-GPTQ
import time
import evaluate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig, Trainer, TrainingArguments

#model_path = "/home/sysgen/Pycharm/wotspace/text-generation-webui/models/HuggingFaceH4_zephyr-7b-beta"
model_path = "TheBloke/zephyr-7B-beta-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

print("\n\n*** Generate:")

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))


# In[]:
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


print(print_number_of_trainable_model_parameters(model))

# In[]:
index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']


prompt = f"You summarize the following conversation.\n\n{dialogue}\n\nSummary: "
prompt_template=f'''<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
'''

inputs = tokenizer(prompt_template, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"].cuda(),
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')


def tokenize_function(example):
    prompt_template = lambda dialogue: f'''<|system|>
    </s>
    <|user|>
    f"You summarize the following conversation.\n\n{dialogue}\n\nSummary: "</s>
    <|assistant|>
    '''
    prompt = [prompt_template(dialogue) for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], max_length=128, padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.

dataset_short = dataset  # .filter(lambda example, index: index % 10 == 0, with_indices=True)
tokenized_datasets = dataset_short.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary', ])
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

# In[]:
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
)

peft_model = get_peft_model(model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    # logging_steps=1,
    # max_steps=1
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

#TODO: make it run
peft_trainer.train()
