# Import unsloth first as requested
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
from datasets import Dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoTokenizer
import wandb

# === Step 1: 定義 Prompt 模板 ===
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an expert with medical knowledge in speaking Chinese and English. You are an AI assistant. You will be given a task. You must generate a detailed and correct answer. And you should also know the knowledge in Chinese.
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
Answer: {}"""

# === Step 2: 定義格式化函數 ===
def format_prompt(sample, EOS_TOKEN):
    input_question = sample["prompt"]
    chain_of_thought = sample["COT"]
    
    # Directly use the content instead of using apply_chat_template
    accepted = sample["chosen"] if isinstance(sample["chosen"], str) else sample["chosen"]["content"]
    rejected = sample["reject"] if isinstance(sample["reject"], str) else sample["reject"]["content"]
    
    sample["prompt"] = alpaca_prompt.format(input_question, chain_of_thought, "")
    sample["chosen"] = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    return sample

# === Step 3: 設定模型名稱與設備 ===
model_name = "deepseek-ai/deepseek-math-7b-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
print(f"Using {device} device\n")

# === Step 4: 載入 tokenizer 與模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# === Step 5: 載入本地 JSON 檔案並轉換為 HuggingFace Dataset ===
with open('./complete_data/combined.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# === Step 6: 格式化資料 ===
EOS_TOKEN = tokenizer.eos_token
dataset = dataset.map(format_prompt, fn_kwargs={"EOS_TOKEN": EOS_TOKEN}, batched=False)
dataset = dataset.train_test_split(test_size=0.01)

# === Step 7: 準備 LoRA 微調模型 ===
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load the previously fine-tuned weights
model.load_adapter("./deepseek-math-7b-orpo-lora", adapter_name="default")
print("Successfully loaded previous fine-tuned weights from: ./deepseek-math-7b-orpo-lora")

# === Step 8: 訓練參數 ===
wandb.login(key="0fc9b0afa729d0aff22088c9788ccf473893d242")

training_args = ORPOConfig(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    beta=0.1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    max_steps=1500,
    num_train_epochs=3,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    output_dir="./deepseek-math-7b-orpo-lora-big",
    save_strategy="epoch",
    report_to="wandb"
)

# === Step 9: 訓練模型 ===
orpo_trainer = ORPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    processing_class=tokenizer
)

# Continue training from previous weights
orpo_trainer.train()

# === Step 10: 儲存模型 ===
model.save_pretrained("./deepseek-math-7b-orpo-lora-continued")
tokenizer.save_pretrained("./deepseek-math-7b-orpo-lora-continued")
print("Model training completed and weights saved to: ./deepseek-math-7b-orpo-lora-continued")
