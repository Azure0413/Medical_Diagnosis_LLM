import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, pipeline
from opencc import OpenCC

# Step 1: 設定模型與 tokenizer 位置
model_path = "./deepseek-math-7b-orpo-lora-continued"

# Step 2: 判斷是否支援 bfloat16
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Step 3: 載入模型與 tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=dtype,
    load_in_4bit=True
)

# Step 4: 指定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 5: 準備 prompt 模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an expert with medical knowledge in speaking Chinese and English. You are an AI assistant. You will be given a task. You must generate a detailed and correct answer. And you should also know the knowledge in Chinese.
Please answer the following medical question. Please using Traditional Chinese to answer.

1. Identify key clinical findings
2. Consider differential diagnoses  
3. Assess severity and urgency
4. Recommend appropriate management

### Question:
{}

### Response:
<think>
{}
</think>
Answer:"""

# Step 6: 準備推論問題
question = "醫生您好，我是病人背景 55 歲男性，終末期腎病（ESRD），血液透析 3 年，每週 3 次 主訴 最近兩週透析中頻繁頭暈、噁心，血壓最低 80/50 mmHg。請問我得了什麼病呢？應該怎麼處理？"
cot = ""

# 套用 prompt 模板
prompt = alpaca_prompt.format(question, cot)

# Step 7: Tokenize 輸入
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Step 8: 推論
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== 模型輸出 ===")
print(response)

answer_start = response.find("Answer:")
translated_text = response[answer_start + len("Answer:"):].strip() if answer_start != -1 else response

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
zh_simplified = translator(translated_text, max_length=512)[0]["translation_text"]

cc = OpenCC('s2tw')  # s2tw = 簡體轉繁體（臺灣）
zh_traditional = cc.convert(zh_simplified)

# Step 13: 輸出結果
print("\n=== 模型英文輸出 ===")
print(translated_text)
print("\n=== 中文翻譯（繁體）輸出 ===")
print(zh_traditional)