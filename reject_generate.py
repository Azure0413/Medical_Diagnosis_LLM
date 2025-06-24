import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from unsloth import FastLanguageModel
import os

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 定義檔案路徑
input_file = "output.json"
output_file = "output_reject_add.json"
backup_file = "moutput_reject_add_backup.json"

# 載入原始JSON檔案
try:
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"成功載入JSON檔案，共有 {len(data)} 筆資料")
except Exception as e:
    print(f"載入JSON檔案時發生錯誤: {e}")
    exit(1)

# 載入 base model 與 fine-tuned 權重
try:
    base_model = "deepseek-ai/deepseek-math-7b-base"
    fine_tuned_path = "./deepseek-math-7b-orpo-lora"  # 微調模型的路徑

    print(f"開始載入 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"開始載入 fine-tuned model: {fine_tuned_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=fine_tuned_path,
        dtype=None,
        load_in_4bit=True
    )
    print("模型與 tokenizer 載入成功")
except Exception as e:
    print(f"載入模型時發生錯誤: {e}")
    exit(1)

# 生成答案的函數
def generate_answer(model, question):
    prompt = question
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **input_ids,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 移除原始問題，只保留生成的答案部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text

# 處理每個問題並添加模型回答
print("開始處理問題並生成回答...")
for i, item in enumerate(tqdm(data)):
    try:
        question = item["prompt"]
        model_response = generate_answer(model, question)

        # 添加 reject 欄位存放模型回答
        item["reject"] = model_response

        # 每處理10個問題保存一次結果，避免資料遺失
        if (i + 1) % 10 == 0:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"已處理 {i+1}/{len(data)} 個問題，已備份到 {backup_file}")
    except Exception as e:
        print(f"處理問題 {i+1} 時發生錯誤: {e}")
        item["reject"] = f"處理錯誤: {str(e)}"

# 最終保存完整的更新後的JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"所有結果已儲存到 {output_file}，共處理 {len(data)} 筆資料")
print("處理完成！")
