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

torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

resp_datas = torch.load("splited_dataset/1w1/RLAIF-V-Dataset-withlogp_2000.pt")
resp_datas += torch.load("splited_dataset/1w1/RLAIF-V-Dataset-withlogp_4000.pt")
resp_datas += torch.load("splited_dataset/1w1/RLAIF-V-Dataset-withlogp_6000.pt")

hf_data = hf_datasets.load_dataset("splited_dataset/1w1")['train'].cast_column("image", hf_datasets.Image(decode=False))
conv_data = []
for hd, rd in zip(hf_data, resp_datas):
    assert hd["prompt"] == rd["prompt"]
    hd["chosen"] = rd["text1"]
    hd["rejected"] = rd["text2"]
    conv_data.append(hd)
    
hf_data = hf_datasets.Dataset.from_list(conv_data)

model_name = 'llava-v1.5-7b'
tokenizer, model, image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-7b", None, model_name, device_map={"": 'cuda'})
inference_logp(model, tokenizer, hf_data, "splited_dataset/1w1_base",
                            0, image_processor, False, is_llava15=True)

model_path = ".ckpt/llava15_7b_DPO-llava15_rlaifv_lora/checkpoints"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 'liuhaotian/llava-v1.5-7b', 'llava-v1.5-7b_lora', device_map={"": 'cuda'})
inference_logp(model, tokenizer, hf_data, "splited_dataset/1w1_dpo",
                            0, image_processor, False, is_llava15=True)

hf_data_base = hf_datasets.load_dataset("splited_dataset/1w1_base")['train'].cast_column("image", hf_datasets.Image(decode=False))
hf_data_dpo = hf_datasets.load_dataset("splited_dataset/1w1_dpo")['train'].cast_column("image", hf_datasets.Image(decode=False))
res_datas = []

for data_base, data_dpo in zip(hf_data_base, hf_data_dpo):
    data_dict_base = []
    data_dict_dpo = []
    assert data_base['prompt'] == data_dpo['question']
    assert data_base['text1'] == data_dpo['chosen']
    assert data_base['text2'] == data_dpo['rejected']
    
    logps_base=json.loads(data_base['logps'])
    logps_dpo=json.loads(data_dpo['logps'])

    if type(logps_base) == type([]):
        (data_dict_base['ref_win_logp'], data_dict_base['ref_win_avg_logp'], data_dict_base['ref_win_per_token_logp'],
        data_dict_base['ref_rej_logp'], data_dict_base['ref_rej_avg_logp'], data_dict_base['ref_rej_per_token_logp']) = logps_base
        (data_dict_dpo['ref_win_logp'], data_dict_dpo['ref_win_avg_logp'], data_dict_dpo['ref_win_per_token_logp'],
        data_dict_dpo['ref_rej_logp'], data_dict_dpo['ref_rej_avg_logp'], data_dict_dpo['ref_rej_per_token_logp']) = logps_dpo
    
    logits = torch.cat(out_dict1["logits"], dim=0).unsqueeze(0)
    labels = out_dict1["sequences"]
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)
    log_prob_chosen = per_token_logps.sum(-1)
    
    logits = torch.cat(out_dict2["logits"], dim=0).unsqueeze(0)
    labels = out_dict2["sequences"]
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)
    log_prob_reject = per_token_logps.sum(-1)
    
    pi_logratios = data_dict_dpo['ref_win_logp'] - data_dict_dpo['ref_rej_logp']
    ref_logratios = data_dict_base['ref_win_logp'] - data_dict_base['ref_rej_logp']

    chosen_rewards = 0.1 * (data_dict_dpo['ref_win_logp'] -
                             data_dict_base['ref_win_logp']).detach()
    rejected_rewards = 0.1 * \
        (data_dict_dpo['ref_rej_logp'] - data_dict_base['ref_rej_logp']).detach()
        
    P_theta_with_ref = F.sigmoid(chosen_rewards - rejected_rewards)
    
    res_datas.append({"qid": data["question_id"], "prompt": data['prompt'], "text1": data['text1'], "text2": data['text2'], "label": label_data[0], "备注": label_data["Unnamed: 1"], "P_theta": P_theta.item(), "P_theta_with_ref": P_theta_with_ref.item()})
    
df = pd.DataFrame(res_datas)
df.to_excel('result/RLAIF-V-7B-test/diff-resp_labeled_withref.xlsx', index=False)



