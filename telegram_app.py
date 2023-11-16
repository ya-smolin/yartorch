
#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
from collections import defaultdict

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

model_path = "TheBloke/zephyr-7B-beta-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"
quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             quantization_config=quantization_config_loading,
                                             trust_remote_code=False,
                                             use_flash_attention_2=True,
                                             revision="main")

instruction="Below is a conversation between a user and you. Instruction: Write a response appropriate to the conversation."

user2dialog = defaultdict(lambda : [{"role": "system", "content": f"{instruction}"},])

print("TOKENIZER:", tokenizer.eos_token)

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привітики) {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.from_user.id in user2dialog:
        del user2dialog[update.message.from_user.id]
    await update.message.reply_text("Привітики)")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    text = "" if update.message.text is None else update.message.text
    user2dialog[update.message.from_user.id].append({"role": "user", "content": text})
    chat = user2dialog[update.message.from_user.id]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    text_reply = tokenizer.decode(output[0])

    # Finding the <s> token that precedes this last </s> token
    last_start_token = text_reply.rfind("<|assistant|>")
    last_end_token = text_reply.rfind("</s>")
    # Extracting the substring between these two positions
    text_reply_new = text_reply[last_start_token + len("<|assistant|>\n"):last_end_token]
    user2dialog[update.message.from_user.id].append({"role": "assistant", "content": text_reply_new})
    await update.message.reply_text(text_reply_new)


def main() -> None:
    # Create the Application and pass it your bot's token.
    with open("token.txt", "r") as f:
        token = f.read().strip()
    application = Application.builder().token(token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()