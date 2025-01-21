from llava.model.builder import load_pretrained_model

model_path = ".ckpt/llava15_7b_DPO-llava15_rlaifv_lora_3k1_from_4w/checkpoints"
model_name = 'llava-v1.5-7b_lora'
model_base = 'liuhaotian/llava-v1.5-7b'
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device_map={"": 'cuda'})

model.save_pretrained(".ckpt/llava15_7b_DPO-llava15_rlaifv_lora_3k1_from_4w_merged")
tokenizer.save_pretrained(".ckpt/llava15_7b_DPO-llava15_rlaifv_lora_3k1_from_4w_merged")