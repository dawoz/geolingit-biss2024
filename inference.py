# %% [markdown]
# <a href="https://colab.research.google.com/github/crux82/BISS-2024/blob/main/BISS-2024_LAB-2.3_ExtremITA_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## MINERVA VS Camoscio VS LLaMAntino LLMs comparison when finetuned on the GeoLingIt dataset

# %% [markdown]
# The code is split into 3 steps:
# - Step 1 - Encoding the data
# - Step 2 - Training the model
# - Step 3 - Inference: generating answers

# %% [markdown]
# # Index:
# 1. Introduction, Workflow and Objectives
# 2. Preliminary steps
# 3. Loading the model
# 4. Generating answers
# 5. Saving the data in the 4-column format

# %%
# install eventually required packages

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import torch
import pandas as pd

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from os import makedirs
from os.path import isdir
import csv
import math
import pprint
from tqdm import tqdm
from geopy.distance import geodesic
from sklearn.metrics import f1_score
import numpy as np
from huggingface_hub import login

# %%
relPath = '.'
TASK = 'geolingit'
DATASET = 'GeoLingIt'  
MODEL = 'MINERVA' # model type 'LLaMA' or 'ANITA' or 'MINERVA'

# %% [markdown]
# Login to access to the MINERVA model (read the access token from the secret.txt file)

# %%
if MODEL == 'MINERVA':
    with open(f"secret.txt", mode="r", encoding="utf-8") as scrF:
        secret_token = scrF.readline()
        login(secret_token)

# %% [markdown]
# ## Encode the test set

# %%
def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        makedirs(f"out/{TASK}")

    data = dict()

    with open(f"{relPath}/{DATASET}/test_a_GOLD.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            text = clean_input_text(row['text'])
            label = row['region']
            data[row['id']] = {
                'text': text,
                'label': label,
            }

    with open(f"{relPath}/{DATASET}/test_b_GOLD.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            latitude = math.floor(eval(row['latitude'])*100)/100.
            longitude = math.floor(eval(row['longitude'])*100)/100.
            data[row['id']]['latitude'] = latitude
            data[row['id']]['longitude'] = longitude

    with open(f"out/{DATASET}/test.txt", "w", encoding="utf-8") as f_o:
        for id, features in data.items():
            f_o.write(f"{id}\t{TASK}\t{features['text']}\t[regione] {features['label']} [geo] {features['latitude']} {features['longitude']}\n")
            
encode()

# %% [markdown]
# ### Utils code for generating text in the ad hoc form for each task

# %%
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

# %% [markdown]
# # Download the model
# 
# This section the already fine-tuned model is downloaded from HuggingFace  
# 
# It is important to note that the model is loaded on 4-bits precision due to memory limitations

# %%
if MODEL == "LLaMA":
  tokenizer =  AutoTokenizer.from_pretrained("yahma/llama-7b-hf")
elif MODEL == "ANITA":
  tokenizer =  AutoTokenizer.from_pretrained("swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA")
elif MODEL == 'MINERVA':
  tokenizer =  AutoTokenizer.from_pretrained("sapienzanlp/Minerva-3B-base-v1.0")
  
modelPath = f"Dosclic98/{MODEL}-{DATASET}"

tokenizer.padding_side = "left"
tokenizer.pad_token_id = (0)

quantization_config = BitsAndBytesConfig(load_in_4_bit = True)

model = AutoModelForCausalLM.from_pretrained(
  modelPath,
  quantization_config=quantization_config,
  torch_dtype=torch.float16,
  device_map="auto",
)

# %%
model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = tokenizer.bos_token_id = 1
model.config.eos_token_id = tokenizer.eos_token_id = 2

if MODEL == "ANITA":
    model.generation_config.pad_token_id = tokenizer.pad_token_id = 0
    model.generation_config.bos_token_id = tokenizer.bos_token_id = 1
    model.generation_config.eos_token_id = tokenizer.eos_token_id = 2

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

# %%
inputs = []
with open(f"{relPath}/out/{DATASET}/test.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        lc = re.split(r"\t|\[regione\]|\[geo\]", line)
        inputs.append(["1", TASK, lc[2], "[regione]" + lc[4] + " [geo]" + lc[5]])
    pprint.pp(inputs)

# %%
def elaborate_generated_output(text):
    region, coordinates = [e.strip() for e in text.removeprefix("[regione]").strip().split('[geo]')]
    latitude, longitude = [float(e) for e in coordinates.split()]
    return region, (latitude, longitude)

# %%
# generate prompts based on task and text
pred_text = []
true_text = []

for input in tqdm(inputs):
    id = input[0]
    task = input[1]
    text = input[2]
    expected_output = input[3]

    instruction = task_to_prompt(task)
    prompt = generate_prompt_pred(instruction, text) # pay attention that the input is not too long (over the max length of your model)

    # tokenization
    tokenized_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # inference
    model.eval()
    with torch.no_grad():
        gen_outputs = model.generate(
            **tokenized_inputs,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256, # how many token (wordpieces) to add to the input prompt.
            do_sample=False # we do not need any sampling or beam seach. We just need the "best" solution, so the greedy search is fine.
        )

        # decoding and printing
        for i in range(len(gen_outputs[0])):
            output = tokenizer.decode(gen_outputs[0][i], skip_special_tokens=True)
            if "### Risposta:" in output:
                response = output.split("### Risposta:")[1].rstrip().lstrip()
                if MODEL == "ANITA":
                    # Remove the # at the end
                    response = response.rstrip("#")
            else:
                response = "UNK"

            #print(text)
            #print(f"\t {expected_output} \t {response}")
            #print(50*"*")
            
            pred_text.append(elaborate_generated_output(response))
            true_text.append(elaborate_generated_output(expected_output))

# %% [markdown]
# ## Compute the metrics: F1-score (macro) for classification and Avg Km (error) between coordinates

# %%
pred_region, pred_coord = tuple(zip(*pred_text))
true_region, true_coord = tuple(zip(*true_text))

avg_km = np.array([geodesic(p, t).km for p, t in zip(pred_coord, true_coord)]).mean()
score = f1_score(true_region, pred_region, average='macro')

print(f"Average distance in km: {avg_km}")
print(f"F1 score: {score}")

# %% [markdown]
# ![image.png](assets/extremita.PNG)


# now save predictions and true labels in a file to use it later for the evaluation with confussion matrix

# %%

with open(f"out/{DATASET}/predictions_{MODEL}.tsv", "w", encoding="utf-8") as f:
    f.write("id\tpred_region\tpred_latitude\tpred_longitude\ttrue_region\ttrue_latitude\ttrue_longitude\n")
    for i, (p, t) in enumerate(zip(pred_text, true_text)):
        f.write(f"{i}\t{p[0]}\t{p[1][0]}\t{p[1][1]}\t{t[0]}\t{t[1][0]}\t{t[1][1]}\n")
