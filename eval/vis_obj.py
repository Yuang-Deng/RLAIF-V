import jsonlines
import json
import os
import base64
import io
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import base64
import io

# import cv2

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in f1.readlines():
            data.append(json.loads(item))
    return data
  
  

questions = [json.loads(q) for q in open(os.path.expanduser("./eval/data/obj_halbench_300_with_image.jsonl"), "r")]
answers = json.load(open('result/RLAIF-V/hall_obj_halbench_answer_-1.json', 'r', encoding='utf-8'))
gen_file = [json.loads(q) for q in open(os.path.expanduser("./result/RLAIF-V/obj_halbench_answer.jsonl"), "r")]

answers_base = json.load(open('result/llava/hall_obj_halbench_answer_-1.json', 'r', encoding='utf-8'))
gen_file_base = [json.loads(q) for q in open(os.path.expanduser("./result/llava/obj_halbench_answer.jsonl"), "r")]

# for q, a, g in zip(questions, answers["sentences"], gen_file):
#     assert q["image_id"] == a["image_id"]
#     image_file = q["image"]
#     image_bytes = base64.b64decode(image_file)
#     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     prompt = g["prompt"]
#     gen_cap = g["text"]
#     gt_words = a["mscoco_gt_words"]
#     mscoco_generated_words = a["mscoco_generated_words"]
#     mscoco_hallucinated_words = a["mscoco_hallucinated_words"]

# 初始化 Flask 应用
app = Flask(__name__, static_folder="/root/dya/RLAIF-V/static", template_folder="/root/dya/RLAIF-V/templates")

# 全局索引
current_index = 0

@app.route("/")
def display():
    global current_index
    q = questions[current_index]
    a_lora = answers["sentences"][current_index]
    g_lora = gen_file[current_index]
    a_base = answers_base["sentences"][current_index]
    g_base = gen_file_base[current_index]

    # 解码图片
    image_bytes = base64.b64decode(q["image"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 将图片保存为临时文件
    image_path = f"static/image_{current_index}.png"
    image.save(image_path)

    # 获取文本字段
    prompt = g_lora["prompt"]
    gen_cap = g_lora["text"]
    gt_words = a_lora["mscoco_gt_words"]
    mscoco_generated_words = a_lora["mscoco_generated_words"]
    mscoco_hallucinated_words = a_lora["mscoco_hallucinated_words"]
    
    prompt_base = g_base["prompt"]
    gen_cap_base = g_base["text"]
    gt_words_base = a_base["mscoco_gt_words"]
    mscoco_generated_words_base = a_base["mscoco_generated_words"]
    mscoco_hallucinated_words_base = a_base["mscoco_hallucinated_words"]

    return render_template(
        "index.html",
        image_path=image_path,
        prompt=prompt,
        gen_cap=gen_cap,
        gt_words=gt_words,
        mscoco_generated_words=mscoco_generated_words,
        mscoco_hallucinated_words=mscoco_hallucinated_words,
        gen_cap_base=gen_cap_base,
        gt_words_base=gt_words_base,
        mscoco_generated_words_base=mscoco_generated_words_base,
        mscoco_hallucinated_words_base=mscoco_hallucinated_words_base,
        index=current_index
    )

@app.route("/next")
def next_item():
    global current_index
    current_index = (current_index + 1) % len(questions)
    return redirect(url_for("display"))

@app.route("/pre")
def pre_item():
    global current_index
    current_index = (current_index - 1) % len(questions)
    return redirect(url_for("display"))

if __name__ == "__main__":
    app.run(debug=True)

    
    
            
