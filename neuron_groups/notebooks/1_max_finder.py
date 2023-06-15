#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[2]:


import plotly.io as pio
try:
    import google.colab
    print("Running as a Colab notebook")
    pio.renderers.default = "colab"
    get_ipython().run_line_magic('pip', 'install transformer-lens fancy-einsum')
    get_ipython().run_line_magic('pip', 'install -U kaleido # kaleido only works if you restart the runtime. Required to write figures to disk (final cell)')
except:
    print("Running as a Jupyter notebook")
    get_ipython().run_line_magic('pip', 'install transformer-lens fancy-einsum')
    pio.renderers.default = "vscode"
    from IPython import get_ipython
    ipython = get_ipython()


# In[59]:


import torch
from fancy_einsum import einsum
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils, ActivationCache
from torchtyping import TensorType as TT
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
from typing import List, Union, Optional
from functools import partial
import pandas as pd
from pathlib import Path
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_dataset
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://stackoverflow.com/q/62691279
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[4]:


get_ipython().system('pip install circuitsvis')
import circuitsvis as cv


# In[5]:


pio.renderers.default='vscode'

def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


# In[6]:


model = HookedTransformer.from_pretrained(
    "gpt2-large",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device=device,
)


# ## Find Activating Examples

# In[7]:


from datasets import load_dataset
dataset = load_dataset("NeelNanda/pile-10k", split="train")


# In[8]:


print(len(dataset.to_dict()['text'][0]))
dataset_text_list = dataset.to_dict()['text']
print(model.to_tokens(dataset.to_dict()['text'][0:5]).shape)


# In[64]:


neurons_json = json.load(open("neuron_finder_results.json"))
neurons = []
for layer, results in neurons_json.items():
    indices = results.keys()
    for index in indices:
        neurons.append((int(layer), int(index)))

print(neurons)


# In[53]:


def get_neurons_acts(model, texts, neurons):
    tokens = model.to_tokens(texts)
    
def cache_to_tuples(cache):
    new_cache = {}
    for key in cache.keys():
        x = torch.max(cache[key], dim=1)
        y = list(x)
        y = [y[0].tolist(), y[1].tolist()]
        y = list(zip(*y))
        new_cache[key] = y # y is a list of tuples, i.e. [(max_value, max_index), ...]
    return new_cache


# In[60]:


batch_size = 4
batched_texts = [dataset_text_list[i: i+batch_size] for i in range(0, len(dataset_text_list), batch_size)]
print(len(batched_texts))

neuron_max_acts = {neuron: [] for neuron in neurons}

for texts in tqdm(batched_texts):
    model.reset_hooks()

    cache = {}

    def return_caching_hook(neuron):
        layer, neuron_index = neuron
        def caching_hook(act, hook):
            cache[(layer, neuron_index)] = act[:, :, neuron_index] # act shape is (batch_size, seq_len, neuron_index)
        return caching_hook
    
    hooks = list(((f"blocks.{layer}.mlp.hook_post", return_caching_hook((layer, index))) for layer, index in neurons))
    print(hooks)

    model.run_with_hooks(
        model.to_tokens(texts),
        fwd_hooks=hooks,
    )
    cache = cache_to_tuples(cache)

    for key in cache.keys():
        neuron_max_acts[key].extend(cache[key])
    break


# In[65]:


print(neuron_max_acts)
with open("neuron_max_acts.json", "w") as f:
    json.dump(neuron_max_acts, f)

