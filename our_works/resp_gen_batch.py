import argparse
import torch
import os
from tqdm import tqdm
import io
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import datasets as hf_datasets
import pandas as pd
import random
from PIL import ImageFilter


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    hf_data = hf_datasets.load_dataset('splited_dataset/1w1_dpo')['train'].cast_column("image", hf_datasets.Image(decode=False))

    disable_torch_init()
    model_path = ".ckpt/llava15_7b_DPO-llava15_rlaifv_lora/checkpoints"
    model_name = 'llava-v1.5-7b_lora'
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 'liuhaotian/llava-v1.5-7b', model_name, device_map={"": 'cuda'})
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    question_idx=0
    output_dicts = []
    batch_len = 8
    for i in tqdm(range(0, len(hf_data), batch_len)):
        batch_data = hf_data[i:i+batch_len]
        batch_images = []
        batch_prompts = []
        for qs, image in zip(batch_data["question"], batch_data["image"]):
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_file = image['bytes']
            image = Image.open(io.BytesIO(image_file)).convert('RGB')
            
            # Randomly apply super-resolution or low-resolution
            if random.choice([True, False]):
                # Apply super-resolution using Lanczos interpolation
                high_res_size = (image.width * 2, image.height * 2)
                image = image.resize(high_res_size, Image.LANCZOS)
            else:
                # Apply low-resolution by resizing down and then up
                image = image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS)
                image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            
            batch_images.append(image)
            batch_prompts.append([input_ids, cur_prompt])
            
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                [input_ids.squeeze(0) for input_ids, _ in batch_prompts],
                batch_first=True,
                padding_side='left',
                padding_value=tokenizer.pad_token_id)

        image_tensors = process_images(batch_images, image_processor, model.config)

        with torch.inference_mode():
            args.temperature = 1
            top_k = 20
            args.num_beams = 1

            # output_ids_1 = model.generate(
            #     batch_input_ids,
            #     images=image_tensors.half().cuda(),
            #     image_sizes=[image.size for image in batch_images],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     top_k=top_k,
            #     num_beams=args.num_beams,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     output_logits=True,
            #     max_new_tokens=1024,
            #     use_cache=True)
            
            # output_ids_2 = model.generate(
            #     batch_input_ids,
            #     images=image_tensors.half().cuda(),
            #     image_sizes=[image.size for image in batch_images],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     top_k=top_k,
            #     num_beams=args.num_beams,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     output_logits=True,
            #     max_new_tokens=1024,
            #     use_cache=True)

            output_ids_3 = model.generate(
                batch_input_ids,
                images=image_tensors.half().cuda(),
                image_sizes=[image.size for image in batch_images],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=top_k,
                num_beams=args.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                max_new_tokens=1024,
                use_cache=True)

        # outputs1 = tokenizer.batch_decode(output_ids_1.sequences, skip_special_tokens=True)
        # outputs2 = tokenizer.batch_decode(output_ids_2.sequences, skip_special_tokens=True)
        outputs3 = tokenizer.batch_decode(output_ids_3.sequences, skip_special_tokens=True)
        
        for idx, (line, output1, output3) in enumerate(zip(batch_data["image"], batch_data["chosen"], outputs3)):
            question_idx += 1
            output_dicts.append({
                "question_id": question_idx,
                "image": line,
                "prompt": batch_prompts[idx][1],
                "text1": output1.strip(),
                "text2": output3.strip(),
                "model_id": model_name,
            })
            if len(output_dicts) % 1000 == 0:
                output_dir = os.path.join("splited_dataset/1w1_dpo_img_srlr")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(output_dicts, os.path.join(output_dir, f'RLAIF-V-Dataset-withlogp_{question_idx}.pt'))
                output_dicts = []
                exit(0)
    # if len(output_dicts) > 0:
    #     torch.save(output_dicts, os.path.join("splited_dataset/1w1_mpo_nodpo", f'RLAIF-V-Dataset-withlogp_{question_idx}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=3)
    args = parser.parse_args()

    print(args.conv_mode)

    eval_model(args)
