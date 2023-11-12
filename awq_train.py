from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "TheBloke/zephyr-7B-beta-AWQ"
#model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")  # , use_flash_attention_2=True

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/zephyr-7B-beta-AWQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
# Load model
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True)

prompt = "Tell me about AI"
prompt_template=f'''<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
'''

print("*** Running model.generate:")

token_input = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    token_input,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=512
)

# Get the tokens from the output, decode them, print them
token_output = generation_output[0]
text_output = tokenizer.decode(token_output)
print("LLM output: ", text_output)

"""
# Inference should be possible with transformers pipeline as well in future
# But currently this is not yet supported by AutoAWQ (correct as of September 25th 2023)
from transformers import pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
"""