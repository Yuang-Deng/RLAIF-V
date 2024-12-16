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
  
  

questions = [json.loads(q) for q in open(os.path.expanduser("./eval/data/mmhal-bench_with_image.jsonl"), "r")]

answers = json.load(open('result/RLAIF-V-7B/mmhal-bench_answer.jsonl.mmhal_test_eval.json.merge_gpt4_score.json', 'r', encoding='utf-8'))
gen_file = [json.loads(q) for q in open(os.path.expanduser("./result/RLAIF-V-7B/mmhal-bench_answer.jsonl"), "r")]

answers_base = json.load(open('result/llava/mmhal-bench_answer.jsonl.mmhal_test_eval.json.merge_gpt4_score.json', 'r', encoding='utf-8'))
gen_file_base = [json.loads(q) for q in open(os.path.expanduser("./result/llava/mmhal-bench_answer.jsonl"), "r")]

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
    a_lora = answers[current_index]
    a_base = answers_base[current_index]

    # 解码图片
    image_bytes = base64.b64decode(q["image"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 将图片保存为临时文件
    image_path = f"static/image_mmhl_{current_index}.png"
    image.save(image_path)

    # 获取文本字段
    print()
    question_type = a_lora["question_type"]
    question_topic = a_lora["question_topic"]
    image_content = a_lora["image_content"]
    question = a_lora["question"]
    gt_answer = a_lora["gt_answer"]
    
    model_answer_DPO = a_lora["model_answer"]
    gpt4_review_DPO = a_lora["gpt4_review"]
    model_answer_base = a_base["model_answer"]
    gpt4_review_base = a_base["gpt4_review"]

    return render_template(
        "index_mmhl.html",
        image_path=image_path,
        question_type=question_type,
        question_topic=question_topic,
        image_content=image_content,
        
        question=question,
        gt_answer=gt_answer,
        model_answer_DPO=model_answer_DPO,
        gpt4_review_DPO=gpt4_review_DPO,
        model_answer_base=model_answer_base,
        gpt4_review_base=gpt4_review_base,
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

    
    
            
