import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import base64
import io
import sys
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import datasets as hf_datasets


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    hf_data = hf_datasets.load_dataset('openbmb/RLAIF-V-Dataset', cache_dir='./RLAIF-V-Dataset')['train'].cast_column("image", hf_datasets.Image(decode=False))
    
    # Randomly select 50 samples from hf_data
    questions = hf_data.shuffle(seed=42).select(range(50))
    # questions = torch.load("result/RLAIF-V-7B-test/mmhal-bench_answer.pt")
    # image = Image.open(io.BytesIO(hf_data[0]["image"]["bytes"])).convert("RGB")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_path_base = "liuhaotian/llava-v1.5-7b"
    model_path = "openbmb/RLAIF-V-7B"
    model_name = 'llava-v1.5-7b'
    tokenizer_base, model_base, image_processor_base, context_len_base = load_pretrained_model(model_path_base, args.model_base, model_name, device_map={"": 'cuda'})
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map={"": 'cuda'})

    
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    question_idx=0
    output_dicts = []
    for line in tqdm(questions):
        qs = line["question"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids_base = tokenizer_image_token(prompt, tokenizer_base, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if 'image' in line.keys():
            image_file = line["image"]['bytes']
            # image_bytes = base64.b64decode(image_file)
            image = Image.open(io.BytesIO(image_file)).convert('RGB')
        elif 'image_path' in line.keys():
            image_path = line['image_path']
            image = Image.open(image_path).convert('RGB')
        else:
            raise NotImplementedError

        image_tensor_base = process_images([image], image_processor_base, model_base.config)[0]
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            # output_ids_base = model_base.generate(
            #     input_ids_base,
            #     images=image_tensor_base.unsqueeze(0).half().cuda(),
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     output_logits=True,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=1024,
            #     use_cache=True)
            
            args.temperature = 1
            # args.top_p = 0.7
            top_k = 20
            args.num_beams = 1
            # output_ids_1 = model_base.generate(
            #     input_ids_base,
            #     images=image_tensor_base.unsqueeze(0).half().cuda(),
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     top_k=top_k,
            #     num_beams=args.num_beams,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     output_logits=True,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=1024,
            #     use_cache=True)
            
            # output_ids_2 = model_base.generate(
            #     input_ids_base,
            #     images=image_tensor_base.unsqueeze(0).half().cuda(),
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     top_k=top_k,
            #     num_beams=args.num_beams,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     output_logits=True,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=1024,
            #     use_cache=True)
            
            output_ids_1 = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=top_k,
                num_beams=args.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
            
            output_ids_2 = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=top_k,
                num_beams=args.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs1 = tokenizer.batch_decode(output_ids_1.sequences, skip_special_tokens=True)[0].strip()
        outputs2 = tokenizer.batch_decode(output_ids_2.sequences, skip_special_tokens=True)[0].strip()

        print(outputs1, "\n", outputs2, "\n\n")
        ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": question_idx,
        #                         #    "image_id": line['image_id'] if 'image_id' in line else line['image_url'],
        #                            "image": line["image"],
        #                            "prompt": cur_prompt,
        #                            "text1": outputs1,
        #                            "text2": outputs2,
        #                            "model_id": model_name
        #                            }) + "\n")
        image = Image.open(io.BytesIO(line["image"]['bytes'])).convert('RGB')
        image.save(f"result/RLAIF-V-7B-test/images/{question_idx}.png")
        ans_file.flush()
        question_idx += 1
        output_dicts.append({
            "question_id": question_idx,
            "image": line["image"],
            "prompt": cur_prompt,
            "text1": outputs1,
            "text2": outputs2,
            "model_id": model_name,
            "out_dict1": {"logits": output_ids_1.logits, "sequences": output_ids_1.sequences, "scores": output_ids_1.scores},
            "out_dict2": {"logits": output_ids_2.logits, "sequences": output_ids_2.sequences, "scores": output_ids_2.scores},})
    ans_file.close()
    torch.save(output_dicts, answers_file.replace(".jsonl", ".pt"))

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
