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
import numpy as np

output_path = "splited_dataset/1w1_dpo_img_flip_theta"
diff_resp_data = "splited_dataset/1w1_dpo_img_flip"
data_source = "splited_dataset/1w1"

base_model = "liuhaotian/llava-v1.5-7b"
dpo_model = ".ckpt/llava15_7b_DPO-llava15_rlaifv_lora_3k1_from_4w/checkpoints"

base_data_path = "splited_dataset/1w1_3kdpo_img_flip_base"
dpo_data_path = "splited_dataset/1w1_3kdpo_img_flip"

torch.distributed.init_process_group(
    backend='nccl',
    world_size=int(os.getenv('WORLD_SIZE', '1')),
    rank=int(os.getenv('RANK', '0')),
    init_method='tcp://localhost:29700'
    )
torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

# resp_datas = torch.load("splited_dataset/1w1_3kdpo_img_flip/RLAIF-V-Dataset-withlogp_1000.pt")
resp_datas = torch.load(f"{diff_resp_data}/RLAIF-V-Dataset-withlogp_2000.pt")
resp_datas += torch.load(f"{diff_resp_data}/RLAIF-V-Dataset-withlogp_4000.pt")
resp_datas += torch.load(f"{diff_resp_data}/RLAIF-V-Dataset-withlogp_6000.pt")
resp_datas += torch.load(f"{diff_resp_data}/RLAIF-V-Dataset-withlogp_8000.pt")
resp_datas += torch.load(f"{diff_resp_data}/RLAIF-V-Dataset-withlogp_10000.pt")

hf_data = hf_datasets.load_dataset(data_source)['train'].cast_column("image", hf_datasets.Image(decode=False))
conv_data = []
for hd, rd in zip(hf_data, resp_datas):
    assert hd["question"] == rd["prompt"]
    hd["chosen"] = rd["text1"]
    hd["rejected"] = rd["text2"]
    hd.pop("logps")
    conv_data.append(hd)

# # # for hd in hf_data:
# # #     hd.pop("logps")
# # #     conv_data.append(hd)
    
hf_data = hf_datasets.Dataset.from_list(conv_data)

tokenizer, model, image_processor, context_len = load_pretrained_model(base_model, None, 'llava-v1.5-7b', device_map={"": 'cuda'})
vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.bfloat16 if True else torch.float16, device=model.device)
image_processor = lambda x: vision_tower.image_processor(x)['pixel_values'][0]
if not os.path.exists(base_data_path):
    os.makedirs(base_data_path)
inference_logp(model, tokenizer, hf_data, base_data_path,
                            0, image_processor, False, is_llava15=True)
torch.distributed.barrier()

tokenizer, model, image_processor, context_len = load_pretrained_model(dpo_model, 'liuhaotian/llava-v1.5-7b', 'llava-v1.5-7b_lora', device_map={"": 'cuda'})
vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.bfloat16 if True else torch.float16, device=model.device)
image_processor = lambda x: vision_tower.image_processor(x)['pixel_values'][0]
if not os.path.exists(dpo_data_path):
    os.makedirs(dpo_data_path)
inference_logp(model, tokenizer, hf_data, dpo_data_path,
               0, image_processor, False, is_llava15=True)
torch.distributed.barrier()

# hf_data = hf_datasets.load_dataset("splited_dataset/1w1")['train'].cast_column("image", hf_datasets.Image(decode=False))
hf_data_base = hf_datasets.load_dataset(base_data_path)['train'].cast_column("image", hf_datasets.Image(decode=False))
hf_data_dpo = hf_datasets.load_dataset(dpo_data_path)['train'].cast_column("image", hf_datasets.Image(decode=False))
res_datas = []

for data_base, data_dpo in zip(hf_data_base.select(range(1000)), hf_data_dpo.select(range(1000))):
    data_dict_base = {}
    data_dict_dpo = {}
    assert data_base['question'] == data_dpo['question']
    assert data_base['chosen'] == data_dpo['chosen']
    assert data_base['rejected'] == data_dpo['rejected']
    
    logps_base=json.loads(data_base['logps'])
    logps_dpo=json.loads(data_dpo['logps'])

    if type(logps_base['logps']) == type([]):
        (data_dict_base['ref_win_logp'], data_dict_base['ref_win_avg_logp'], data_dict_base['ref_win_per_token_logp'],
        data_dict_base['ref_rej_logp'], data_dict_base['ref_rej_avg_logp'], data_dict_base['ref_rej_per_token_logp']) = logps_base['logps']
        (data_dict_dpo['ref_win_logp'], data_dict_dpo['ref_win_avg_logp'], data_dict_dpo['ref_win_per_token_logp'],
        data_dict_dpo['ref_rej_logp'], data_dict_dpo['ref_rej_avg_logp'], data_dict_dpo['ref_rej_per_token_logp']) = logps_dpo['logps']
    
    data_dict_dpo['ref_win_logp'] = torch.tensor(data_dict_dpo['ref_win_logp'])
    data_dict_dpo['ref_rej_logp'] = torch.tensor(data_dict_dpo['ref_rej_logp'])
    data_dict_base['ref_win_logp'] = torch.tensor(data_dict_base['ref_win_logp'])
    data_dict_base['ref_rej_logp'] = torch.tensor(data_dict_base['ref_rej_logp'])
    
    pi_logratios = data_dict_dpo['ref_win_logp'] - data_dict_dpo['ref_rej_logp']
    ref_logratios = data_dict_base['ref_win_logp'] - data_dict_base['ref_rej_logp']

    chosen_rewards = 0.1 * (data_dict_dpo['ref_win_logp'] -
                             data_dict_base['ref_win_logp'])
    rejected_rewards = 0.1 * \
        (data_dict_dpo['ref_rej_logp'] - data_dict_base['ref_rej_logp'])
        
    P_theta_with_ref = F.sigmoid(chosen_rewards - rejected_rewards)
    
    if P_theta_with_ref < 0.5:
        data_base['chosen'], data_base['rejected'] = data_base['rejected'], data_base['chosen']
        logps_base['logps'][0], logps_base['logps'][1], logps_base['logps'][2], logps_base['logps'][3], logps_base['logps'][4], logps_base['logps'][5] = logps_base['logps'][3], logps_base['logps'][4], logps_base['logps'][5], logps_base['logps'][0], logps_base['logps'][1], logps_base['logps'][2]
        data_base["logps"] = json.dumps(logps_base)
        
    data_base["P_theta_with_ref"] = P_theta_with_ref.item()
    # res_datas.append(data_base)
    if P_theta_with_ref < 0.4 or P_theta_with_ref > 0.7:
        res_datas.append(data_base)
    
import matplotlib.pyplot as plt

# Extract P_theta_with_ref values
P_theta_with_ref_values = [data["P_theta_with_ref"] for data in res_datas]

# Calculate the distribution
bins = np.arange(0, 1.1, 0.1)
hist, bin_edges = np.histogram(P_theta_with_ref_values, bins=bins)

# Print the distribution
for i in range(len(hist)):
    print(f"Range {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}: {hist[i]}")

# # Plot the distribution
# plt.hist(P_theta_with_ref_values, bins=bins, edgecolor='black')
# plt.xlabel('P_theta_with_ref')
# plt.ylabel('Frequency')
# plt.title('Distribution of P_theta_with_ref')
# plt.savefig('P_theta_with_ref_distribution.png')

# print("writing， len:", len(res_datas))
if not os.path.exists(output_path):
    os.makedirs(output_path)
pd.DataFrame(res_datas).to_parquet(os.path.join(output_path, "RLAIF-V-Dataset-withlogp_40000-50000.parquet"))
print("done")