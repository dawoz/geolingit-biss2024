# %% [markdown]
# <a href="https://colab.research.google.com/github/crux82/BISS-2024/blob/main/BISS-2024_LAB-2.1_ExtremITA_data_encoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Camoscio VS Minerva LLMs comparison when finetuned on the GeoLingIt dataset
# 
# The code is split into 4 steps, reflecting the following process:
# - Step 1 - Encoding the data
# - Step 2 - Training the models
# - Step 3 - Inference: generating answers

# %% [markdown]
# ## Step 1 - Encoding the data
# 
# In this Notebook we will see the encoding part of the data, given that we have some datasets each of which in its own format, in order to transform it into a sequence to sequence format. We will save the data on a file for next steps.

# %% [markdown]
# ## Input
# The "input" of the Notebook is a file in the PubTator format, as given from the challenge.
# 
# ## Output
# The "output" (i.e. the result) of this Notebook is a simple txt file delimited by tabs, with four columns:
# - id
# - task name, from which the natural language task description is generated
# - input text
# - expected output

# %%
import random
from os.path import isdir
from os import mkdir, makedirs
import spacy
import re
import csv
import math
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import csv
import json
from huggingface_hub import login

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import torch
from datasets import load_dataset
import pandas as pd

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Path to the folder 'GeoLingIt' or 'GeoLingIt-mod'

# %%
relPath = '.'
TASK = "geolingit"
DATASET = "GeoLingIt" # folder where the dataset is located
MODEL = "LLaMA" # model type 'LLaMA' or 'ANITA' or 'MINERVA'

random.seed(23)

# %% [markdown]
# Login to access to the MINERVA model (read the access token from the secret.txt file)

# %%
if MODEL == 'MINERVA':
    with open(f"secret.txt", mode="r", encoding="utf-8") as scrF:
        secret_token = scrF.readline()
        login(secret_token)

# %% [markdown]
# # Dataset generation

# %%
nlp = spacy.load("it_core_news_sm", disable=["lemmatizer", "tagger"])



def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{DATASET}"):
        makedirs(f"out/{DATASET}")

    for split in ['dev', 'train']:
        data = dict()

        with open(f"{relPath}/{DATASET}/subtask_a/{split}_a.tsv", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                text = clean_input_text(row['text'])
                label = row['region']
                data[row['id']] = {
                    'text': text,
                    'label': label,
                }

        with open(f"{relPath}/{DATASET}/subtask_b/{split}_b.tsv", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                latitude = math.floor(eval(row['latitude'])*100)/100.
                longitude = math.floor(eval(row['longitude'])*100)/100.
                data[row['id']]['latitude'] = latitude
                data[row['id']]['longitude'] = longitude

        with open(f"out/{DATASET}/{split}.txt", "w", encoding="utf-8") as f_o:
            for id, features in data.items():
                f_o.write(f"{id}\t{TASK}\t{features['text']}\t[regione] {features['label']} [geo] {features['latitude']} {features['longitude']}\n")



# %% [markdown]
# It will generate a file for the task. In order to fine-tune the model you should merge them into one single file and split them into `train.txt` and `dev.txt`.
# These files are made of 4 columns (with a tab character as a delimiter) without any header:
# - id
# - task name, from which the natural language task description is generated
# - input text
# - expected output

# %%
with open(f"{relPath}/{DATASET}/subtask_a/train_a.tsv", "r", encoding='utf-8') as f:
  lines = f.readlines()
  for i, line in enumerate(lines):
    print(line)
    if i > 3:
      break

# %%
encode()

# %% [markdown]
# Let's open the newly created file and see what's inside

# %%
with open(f"out/{DATASET}/train.txt", "r") as f:
  lines = f.readlines()
  print(lines[0])
  print(lines[8])

# %% [markdown]
# Generating text in the ad-hoc form

# %%
# TODO We will probably add here the geolingit variation with the regional capital coordinates
def target_answer_to_text(target_text: str, task: str):
    if task == "geolingit":
        return target_text
    else:
        return "task sconosciuto"

def target_text_to_answer(target_text: str, task: str):
    if task == "geolingit":
        return target_text
    else:
        return "task sconosciuto"

def task_to_prompt(task: str):
    if task == "geolingit":
        return "Scrivi la regione di appartenenza di chi ha scritto questo testo, seguito dalla latitudine, seguita dalla longitudine."
    else:
        return "task sconosciuto"


 ################ GENERATE METHODS ################
def generate_prompt_pred(instruction, input_):
    return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{instruction}
### Input:
{input_}
### Risposta:"""

def generate_prompt_str(instruction, input_):
    return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{instruction}
### Input:
{input_}
### Risposta:"""

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Risposta:
{data_point["output"]}"""
    else:
        return f"""Di seguito è riportata un'istruzione che descrive un task. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{data_point["instruction"]}
### Risposta:
{data_point["output"]}"""

# %% [markdown]
# ------------------------------
# ## Hyper-parameteres
# 
# Here we set all the hyper-parameters we need:
# - specify the device to exploit the GPU (`cuda`)
# - use the tokenizer for llama 7b
# - take the base model of Camoscio from [Huggingface](https://huggingface.co/sag-uniroma2/extremITA-Camoscio-7b)
# - the paths for our training and development set
# - we cut off the length of sentences to maximum `512` words and `1200` charachters
# - we set then the LoRA hyper-params:
#   - the rank `R` of the two matrices A and B
#   - the normalization factor `Alpha`
#   - the `dropout rate`
#   - the target modules, i.e. where to insert the LoRA modules: `q`, `k` and `v` for the attention and `o` for the final output layer.
# - `number of epochs` to train the model
# - `batch_size` is the global size of our batch of examples, but since we have a small GPU with only 15GB of memory, we need to scale this down, so we introduce 2 new concepts:
#   - `micro_batch_size` is the real size of the batch we will use, which will be smaller (usually 2,4,8)
#   - `gradient_accumulation_steps` how many steps (of batches) we want to accumulate the `loss` for before we update the parameters of the model. In this way we simulate a bigger `batch_size` by accumulating the `loss` for more than one iteration, and then we update the model.
# - the `learning_rate`, which controls the intensity of the update
# - the `warmup_ratio`, as we are using a scheduler for the learning rate, i.e. it will not be fixed during the whole training, but will vary based on the time.

# %%
DEVICE = "cuda"
if MODEL == "LLaMA":
    TOKENIZER_MODEL = "yahma/llama-7b-hf"
    BASE_MODEL = "sag-uniroma2/extremITA-Camoscio-7b"
elif MODEL == "ANITA": 
    TOKENIZER_MODEL = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    BASE_MODEL = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
elif MODEL == 'MINERVA':
    TOKENIZER_MODEL = "sapienzanlp/Minerva-3B-base-v1.0"
    BASE_MODEL = "sapienzanlp/Minerva-3B-base-v1.0"

input_train_path = f"out/{DATASET}/train.txt"
input_dev_path = f"out/{DATASET}/dev.txt"
OUTPUT_DIR = f"LLaMinerva/{DATASET}/{MODEL}"

CUTOFF_LEN = 512
CUT_INPUT_CHAR_LENGTH = 1200

task = "*"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

EPOCHS = 10 # better 10 epochs
BATCH_SIZE = 32 #it would be better 128 but it may require too much GPU memory (original 32 for LLaMA and ANITA, but 64 for MINERVA)
MICRO_BATCH_SIZE = 8 #it would be better 32 but it may require too much GPU memory (original 8 for LLaMA and ANITA, but 32 for MINERVA)
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.1

tmp_train_file_name = "tmp_train.json"
tmp_dev_file_name = "tmp_dev.json"

# %% [markdown]
# 

# %% [markdown]
# ## Functions
# We will define now some functions in order to facilitate the readability of the Notebook.

# %%
#============================================
#               FUNCTIONS
#============================================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

#LOAD INPUT TSV files in the extremITA format
def load(input_file_path):
    dataset_df = pd.read_csv(input_file_path, header=None, usecols=[0,1, 2, 3], names=['0', '1', '2', '3'], \
                             sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8').astype(str)
    dataset_df = dataset_df.rename(
        columns={"0": "id", "1": "prefix", "2": "input_text", "3": "target_text"}
    )
    dataset_df = dataset_df[["id", "input_text", "target_text", "prefix"]]
    return dataset_df


# Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
def tokenize(prompt, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

# Notice: result["labels"] is rewritten so that only the output is considered
def generate_and_tokenize_prompt(data_point, add_eos_token=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, CUTOFF_LEN)

    user_prompt = generate_prompt_str(
        data_point["instruction"], data_point["input"]
    )
    tokenized_user_prompt = tokenize(
        user_prompt, CUTOFF_LEN, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt



def load_and_prepare_data(input_file_path: str, tasks):

    df = load(input_file_path)

    if isinstance(tasks, str):
        if(tasks != "*"):
            df = df[df["prefix"]==tasks]
    elif isinstance(tasks, list):
        tmp = None
        for task in tasks:
            if tmp == None:
                tmp = df[df["prefix"]==task]
            else:
                tmp += df[df["prefix"]==task]
        df = tmp

    print(df.target_text.value_counts())

    dataset_data = [
        {
            "instruction": task_to_prompt(row_dict["prefix"]),
            "input": row_dict["input_text"],
            "output": target_text_to_answer(row_dict["target_text"], row_dict["prefix"])
        }
        for row_dict in df.to_dict(orient="records")
    ]

    return dataset_data

def trim_long_input(json_input, cutoff_len=10000000):
    for json_data in json_input:
        json_data["input"] = json_data["input"][:cutoff_len]
    return json_input


# %% [markdown]
# ## Fine-tuning the LLM
# We first load the data that was already prepared then we cut the maximum length accordingly to the previous params.

# %%
#-------------------
#    LOAD DATA
#-------------------
train_data = load_and_prepare_data(input_train_path, task)
dev_data = load_and_prepare_data(input_dev_path, task)


with open(tmp_train_file_name, "w") as f:
   json.dump(train_data, f)
with open(tmp_dev_file_name, "w") as f:
   json.dump(dev_data, f)

json_train = load_dataset("json", data_files=tmp_train_file_name)
json_dev = load_dataset("json", data_files=tmp_dev_file_name)

# TRIM LONG INPUT
json_train["train"] = trim_long_input(json_train["train"], CUT_INPUT_CHAR_LENGTH)
json_dev["train"] = trim_long_input(json_dev["train"], CUT_INPUT_CHAR_LENGTH)

# %% [markdown]
# Now we need to load the model and its associated tokenizer. Choose here the number of bits for loading the models. Remember that lower bits mean lower precision, and thus a drop in performance.

# %%
#-------------------
#    LOAD MODEL
#-------------------
# base model here, choose between 4, 8 bits or full precision

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# we need to explicitly assign the ids for the pad token, begin and end of sentence here
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

# we will add the pad on the left side, as to simulate an older chat history for which we don't care anymore.
# More importantly we want the model to generate on the "right" side of the sentence, as to complete a request
tokenizer.padding_side = "left"


# PREPARE DATA
train_data = ( json_train["train"].shuffle().map(generate_and_tokenize_prompt) )
val_data = ( json_dev["train"].shuffle().map(generate_and_tokenize_prompt) )

# To reduce GPU memory usage
if MODEL == "ANITA":
   model.gradient_checkpointing_enable()

# PREPARE MODEL and add the LoRA modules
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_strategy = "steps",
    logging_steps=1,
    optim="adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    output_dir=OUTPUT_DIR,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    label_names=["labels"]
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# istantiate a Trainer object using the hyper-params we defined earlier
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

# we need to explicitly assign the ids for the pad token, begin and end of sentence here
model.config.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

# speeds up training
if torch.__version__ >= "2":
    model = torch.compile(model)

# %%
#-------------------
#    TRAIN & SAVE
#-------------------

trainer.train()

model.save_pretrained(OUTPUT_DIR)

# %%
#import gc
#import torch
#del model
#gc.collect()
#torch.cuda.empty_cache()


