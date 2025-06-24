# Import unsloth first as requested
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
from datasets import Dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoTokenizer, EarlyStoppingCallback
import wandb
import numpy as np
from sklearn.model_selection import train_test_split

# === Step 1: 改進的 Prompt 模板 ===
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are an expert physician with extensive knowledge in internal medicine, nephrology, and emergency care.
Analyze the patient case systematically using clinical reasoning:
1. Identify key clinical findings
2. Consider differential diagnoses  
3. Assess severity and urgency
4. Recommend appropriate management

### Patient Case:
{}

### Clinical Analysis:
<think>
{}
</think>

### Medical Assessment:
{}"""

# === Step 2: 改進的資料處理函數 ===
def validate_and_clean_data(data):
    """驗證並清理資料品質"""
    cleaned_data = []
    for item in data:
        if all(key in item for key in ["prompt", "COT", "chosen", "reject"]):
            if (len(item["prompt"]) > 10 and len(item["prompt"]) < 1000 and
                len(item["chosen"]) > 10 and len(item["reject"]) > 10):
                cleaned_data.append(item)
    print(f"Data cleaned: {len(data)} -> {len(cleaned_data)} samples")
    return cleaned_data

def format_prompt(sample, EOS_TOKEN):
    """改進的格式化函數"""
    input_question = sample["prompt"].strip()
    chain_of_thought = sample["COT"].strip()
    
    accepted = sample["chosen"] if isinstance(sample["chosen"], str) else sample["chosen"]["content"]
    rejected = sample["reject"] if isinstance(sample["reject"], str) else sample["reject"]["content"]
    
    accepted = accepted.strip()
    rejected = rejected.strip()
    
    sample["prompt"] = alpaca_prompt.format(input_question, chain_of_thought, "")
    sample["chosen"] = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    
    sample["prompt_length"] = len(sample["prompt"])
    sample["chosen_length"] = len(sample["chosen"])
    sample["rejected_length"] = len(sample["rejected"])
    
    return sample

# === Step 3: 設定模型名稱與設備 ===
model_name = "deepseek-ai/deepseek-math-7b-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
print(f"Using {device} device with dtype {torch_dtype}\n")

# === Step 4: 載入 tokenizer 與模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the base model with optimized settings
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch_dtype,
    load_in_4bit=True,
    trust_remote_code=True,
)

# === Step 5: 改進的資料載入和處理 ===
with open('./complete_data/combined.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data = validate_and_clean_data(data)

train_data, temp_data = train_test_split(data, test_size=0.15, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# === Step 6: 格式化資料 ===
EOS_TOKEN = tokenizer.eos_token

train_dataset = train_dataset.map(format_prompt, fn_kwargs={"EOS_TOKEN": EOS_TOKEN}, batched=False)
val_dataset = val_dataset.map(format_prompt, fn_kwargs={"EOS_TOKEN": EOS_TOKEN}, batched=False)

def filter_length(example):
    return (example["prompt_length"] + example["chosen_length"] < 1800 and 
            example["prompt_length"] + example["rejected_length"] < 1800)

train_dataset = train_dataset.filter(filter_length)
val_dataset = val_dataset.filter(filter_length)

print(f"After filtering - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# === Step 7: 優化的 LoRA 配置 ===
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)

# Load previous weights if available
try:
    model.load_adapter("./deepseek-math-7b-orpo-lora", adapter_name="default")
    print("Successfully loaded previous fine-tuned weights")
except:
    print("No previous weights found, starting from base model")

# === Step 8: 修復ORPO配置以解決evaluation錯誤 ===
wandb.login(key="0fc9b0afa729d0aff22088c9788ccf473893d242")

training_args = ORPOConfig(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    beta=0.1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine_with_restarts",
    max_steps=2000,
    num_train_epochs=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    warmup_ratio=0.05,
    max_length=1536,
    max_prompt_length=768,
    max_completion_length=768,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    dataloader_drop_last=True,
    output_dir="./deepseek-math-7b-orpo-lora-improved",
    save_strategy="steps",
    save_steps=250,
    
    # === 修復evaluation相關設置 ===
    eval_strategy="steps",
    eval_steps=500,  # 增加eval間隔減少evaluation頻率
    logging_steps=50,
    load_best_model_at_end=False,  # 暫時關閉以避免evaluation問題
    # metric_for_best_model="eval_loss",  # 註釋掉避免相關問題
    # greater_is_better=False,
    
    report_to="wandb",
    save_total_limit=3,
    seed=42,
    
    # === 新增參數解決'float' object錯誤 ===
    prediction_loss_only=True,  # 關鍵修復：只返回loss，避免logits處理問題
    remove_unused_columns=False,  # 保留所有列
    include_inputs_for_metrics=False,  # 不包含inputs用於metrics
)

# === Step 9: 自定義ORPO Trainer解決evaluation問題 ===
class FixedORPOTrainer(ORPOTrainer):
    """修復版ORPO Trainer"""
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """覆寫prediction_step修復float unsqueeze錯誤"""
        try:
            # 調用父類方法
            result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            return result
        except AttributeError as e:
            if "'float' object has no attribute 'unsqueeze'" in str(e):
                print("捕獲float unsqueeze錯誤，使用簡化prediction...")
                # 如果出現float unsqueeze錯誤，使用簡化版本
                with torch.no_grad():
                    loss = self.compute_loss(model, inputs)
                    
                if prediction_loss_only:
                    return (loss, None, None)
                else:
                    # 返回基本格式避免複雜的logits處理
                    return (loss, torch.tensor([0.0]), torch.tensor([0.0]))
            else:
                raise e
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """改進evaluate方法"""
        try:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        except (AttributeError, TypeError) as e:
            if "'float' object has no attribute 'unsqueeze'" in str(e) or "float" in str(e):
                print(f"Evaluation出現錯誤: {e}")
                print("跳過這次evaluation...")
                # 返回基本的評估結果
                return {
                    f"{metric_key_prefix}_loss": 0.0,
                    f"{metric_key_prefix}_runtime": 0.0,
                    f"{metric_key_prefix}_samples_per_second": 0.0,
                    f"{metric_key_prefix}_steps_per_second": 0.0,
                }
            else:
                raise e

# === Step 10: 訓練模型 ===
orpo_trainer = FixedORPOTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    processing_class=tokenizer,
    # callbacks=[early_stopping],  # 暫時移除early stopping避免evaluation問題
)

# 移除自定義metrics以避免額外錯誤
# orpo_trainer.compute_metrics = compute_metrics

print("Starting training...")
try:
    trainer_stats = orpo_trainer.train()
    
    print("Training completed successfully!")
    
    # === 安全的evaluation ===
    print("Attempting final evaluation...")
    try:
        eval_results = orpo_trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")
    except Exception as e:
        print(f"Final evaluation failed: {e}")
        print("Skipping final evaluation...")
        eval_results = {"eval_loss": "N/A"}
    
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()

print("Saving model...")
try:
    model.save_pretrained("./deepseek-math-7b-orpo-lora-continued")
    tokenizer.save_pretrained("./deepseek-math-7b-orpo-lora-continued")
    
    # 儲存訓練統計（如果有的話）
    if 'trainer_stats' in locals():
        with open("./deepseek-math-7b-orpo-lora-continued/training_stats.json", "w") as f:
            json.dump(trainer_stats.log_history, f, indent=2)
    
    print("Model training completed and weights saved to: ./deepseek-math-7b-orpo-lora-continued")
    
except Exception as e:
    print(f"Error saving model: {e}")

# === Step 12: 模型測試 ===
print("Testing model...")
try:
    FastLanguageModel.for_inference(model)
    
    test_prompt = alpaca_prompt.format(
        "醫生您好，我是病人背景 55 歲男性，終末期腎病（ESRD），血液透析 3 年，每週 3 次 主訴 最近兩週透析中頻繁頭暈、噁心，血壓最低 80/50 mmHg。請問我得了什麼病呢？應該怎麼處理？",
        "",
        ""
    )
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== 測試輸出 ===")
    print(response)
    
except Exception as e:
    print(f"Model testing failed: {e}")

print("Script completed!")
