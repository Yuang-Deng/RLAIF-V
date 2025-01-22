## resp gen batch
批量推理，用于生成不同回复
model path: lora权重 或 全量权重，由model name控制
model name: 带_lora 则使用lora权重
model base: 如果lora，使用作为base model

## data split
数据拆分

## model merge
lora权重与基础模型合并，用于后续训练

## score cal
用于计算p_theta

## muffin/train/train_llava15_lora_MPO.py
用于MPO训练， 训练命令如下
```bash
/home/mnt/share_server/miniconda3/envs/mmlm/bin/deepspeed /home/mnt/share_server/workspace/dengyuang/RLAIF-V/muffin/train/train_llava15_lora_MPO.py --deepspeed script/zero2.json --model_name_or_path liuhaotian/llava-v1.5-7b --delta 0.3 --data_dir ./splited_dataset/4w1 --image_folder not_used --vision_tower openai/clip-vit-large-patch14-336 --mm_use_im_start_end False --mm_use_im_patch_token False --fully_tune False --image_aspect_ratio pad --bf16 True --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --output_dir /home/mnt/share_server/workspace/dengyuang/RLAIF-V/.ckpt/llava15_7b_DPO-llava15_rlaifv_lora_4w_mpo_0.3/checkpoints --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy steps --save_steps 5000 --save_total_limit 50 --data_source_names "" --data_source_weights 1 --max_steps 10000 --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.05 --lr_scheduler_type cosine --logging_steps 2 --logging_dir /home/mnt/share_server/workspace/dengyuang/RLAIF-V/.ckpt/llava15_7b_DPO-llava15_rlaifv_lora_4w_mpo_0.3/RLAIF-V/log --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True --task DPO --report_to wandb --run_name llava15_rlaifv_lora --dataloader_num_workers 16 --dpo_use_average False --dpo_token_weighted False --dpo_token_weight 1.0 --dpo_beta 0.1 --lora_enable True
```