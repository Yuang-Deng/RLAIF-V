import torch
import json
import pandas as pd
from llava.model.builder import load_pretrained_model
import datasets as hf_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch.utils.data as torch_data
from muffin.eval.muffin_inference_logp import inference_logp
import torch.nn.functional as F
import os

hf_data = hf_datasets.load_dataset("splited_dataset/1w1")['train'].cast_column("image", hf_datasets.Image(decode=False))
hf_data = hf_data.shuffle(seed=42)
hf_data1 = hf_data.select(range(2635))
hf_data1.to_parquet(os.path.join("splited_dataset/1w1_2635", f'RLAIF-V-Dataset-withlogp_{0}-{2635}.parquet'))

# hf_data1 = hf_data.select(range(40000, 50000))
# hf_data1.to_parquet(os.path.join("splited_dataset/1w1", f'RLAIF-V-Dataset-withlogp_{40000}-{50000}.parquet'))

# hf_data1 = hf_data.select(range(50000, 60000))
# hf_data1.to_parquet(os.path.join("splited_dataset/1w2", f'RLAIF-V-Dataset-withlogp_{50000}-{60000}.parquet'))

# hf_data1 = hf_data.select(range(60000, 70000))
# hf_data1.to_parquet(os.path.join("splited_dataset/1w3", f'RLAIF-V-Dataset-withlogp_{60000}-{70000}.parquet'))

# hf_data1 = hf_data.select(range(70000, 80000))
# hf_data1.to_parquet(os.path.join("splited_dataset/1w4", f'RLAIF-V-Dataset-withlogp_{70000}-{80000}.parquet'))
