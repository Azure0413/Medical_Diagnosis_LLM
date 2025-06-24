# Import unsloth first as requested
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
from transformers import AutoTokenizer
from opencc import OpenCC
import time

# === Step 1: 模型配置 ===
model_path = "./deepseek-math-7b-orpo-lora-continued"  # 與training code中保存路徑一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

print(f"Using {device} device with dtype {torch_dtype}")
print(f"Loading model from: {model_path}")

# === Step 2: Prompt 模板（與training一致） ===
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. Please using Traditional Chinese to answer.

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

# === Step 3: 載入Fine-tuned模型 ===
def load_model():
    """載入已訓練的模型"""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch_dtype,
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # 設定為inference模式
        FastLanguageModel.for_inference(model)
        model.to(device)
        
        print("✅ 模型載入成功！")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None, None

# === Step 4: 推理函數 ===
def generate_medical_response(model, tokenizer, patient_case, chain_of_thought="", max_new_tokens=400):
    """生成醫學回答"""
    
    # 創建完整prompt
    prompt = alpaca_prompt.format(patient_case, chain_of_thought, "")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成參數（與training時測試參數保持一致）
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True
    }
    
    # 生成回答
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, **generation_kwargs)
        end_time = time.time()
    
    # 解碼回答
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取新生成的部分
    generated_text = response[len(prompt):].strip()
    
    generation_time = end_time - start_time
    
    return {
        "prompt": prompt,
        "response": generated_text,
        "full_response": response,
        "generation_time": generation_time,
        "token_count": len(outputs[0]) - len(inputs["input_ids"][0])
    }

# === Step 5: 兩步推理函數（與原始代碼相似） ===
def two_step_inference(model, tokenizer, patient_case):
    """執行兩步推理過程"""
    
    print("🔍 開始第一步推理...")
    
    # 第一步：初步分析
    first_question = patient_case + " 請根據上述症狀進行分析和思考。"
    first_result = generate_medical_response(model, tokenizer, first_question, "", 300)
    
    print("✅ 第一步推理完成")
    print(f"⏱️  生成時間: {first_result['generation_time']:.2f}秒")
    
    # 提取第一步的答案部分
    first_response = first_result["response"]
    
    # 尋找Answer部分
    answer_start = first_response.find("### Medical Assessment:")
    if answer_start != -1:
        first_analysis = first_response[answer_start + len("### Medical Assessment:"):].strip()
    else:
        # 如果沒有找到特定標記，使用整個回應
        first_analysis = first_response
    
    print("\n📋 第一步分析結果:")
    print("=" * 60)
    print(first_analysis[:300] + "..." if len(first_analysis) > 300 else first_analysis)
    print("=" * 60)
    
    print("\n🔍 開始第二步推理...")
    
    # 第二步：基於第一步的分析進行更深入的推理
    second_result = generate_medical_response(model, tokenizer, patient_case, first_analysis, 400)
    
    print("✅ 第二步推理完成")
    print(f"⏱️  生成時間: {second_result['generation_time']:.2f}秒")
    
    return {
        "first_step": first_result,
        "second_step": second_result,
        "first_analysis": first_analysis
    }

# === Step 6: 主要執行函數 ===
def main():
    """主要執行函數"""
    
    print("🚀 開始載入醫學AI模型...")
    print("=" * 70)
    
    # 載入模型
    model, tokenizer = load_model()
    
    if model is None:
        print("❌ 無法載入模型，程式結束")
        return
    
    # === 測試案例（與training code一致） ===
    test_cases = [
        # {
        #     "name": "透析低血壓案例",
        #     "case": "醫生您好，我是病人背景 55 歲男性，終末期腎病（ESRD），血液透析 3 年，每週 3 次 主訴 最近兩週透析中頻繁頭暈、噁心，血壓最低 80/50 mmHg。請問我得了什麼病呢？應該怎麼處理？"
        # },
        {
            "name": "心血管案例",
            "case": "65歲女性患者，糖尿病史15年，最近出現胸悶、呼吸困難，特別是活動時加重。心電圖顯示ST段壓低。請問可能的診斷和處理建議？"
        },
        # {
        #     "name": "急腹症案例", 
        #     "case": "40歲男性，右上腹疼痛2天，伴發燒38.5°C，噁心嘔吐。體檢發現右上腹壓痛，Murphy's sign陽性。請問診斷和治療建議？"
        # }
    ]

    # test_cases = [
    #     {
    #         "name": "主動脈剝離",
    #         "case": "醫生您好，一位 58 歲男性，已知 高血壓 15 年、不規則服藥。早上於工作時 突發劇烈撕裂樣胸痛，自前胸向背部游移，合併冒冷汗與頭暈。 到院時 BP 190/100 mmHg（右臂）、160/88 mmHg（左臂），HR 110 bpm，RR 22/min，SpO₂ 96%（室內空氣）。在等待影像確認之際，下列哪一項靜脈給藥最能有效降低主動脈壁剪切力，應作為初始藥物治療？"
    #     },
    #     {
    #         "name": "急性胰臟炎",
    #         "case": "一位 48 歲男性，有 10 年酗酒史（每日約 250 mL 高粱酒），昨晚聚餐後突發 持續性上腹部劇痛，向背部放射，伴噁心、嘔吐 3 次。 到院時 BP 100/64 mmHg、HR 112 bpm、RR 20/min、T 37.9 °C，皮膚稍冷汗。 檢查：上腹壓痛，無反彈痛；無黃疸。"
    #     },
    #     {
    #         "name": "糖尿病酮酸中毒", 
    #         "case": "一位 23 歲女性，已知 第一型糖尿病 5 年，最近因課業繁忙 兩天未注射胰島素。 症狀：多尿、口渴、噁心嘔吐 2 次，呼吸急促。 到院時： BP 90/58 mmHg、HR 120 bpm、RR 28 /min (Kussmaul 呼吸)、T 37.3 °C 皮膚乾燥、黏膜乾裂"
    #     },
    #     {
    #         "name": "透析低血壓案例",
    #         "case": "一位 72 歲女性，既往有 高血壓、心房顫動（未規律服用抗凝劑）。 病史：今晨 08:10 與家人談話時突發 右側肢體無力合併語言障礙；08:15 家人即撥打 119，08:55 抵達急診。 到院評估（09:00）： NIHSS 14 分（右側上下肢 4/5、全球失語、凝視偏向） BP 175/95 mmHg，HR 90 bpm，RR 18/min，T 36.8 °C 指尖血糖 120 mg/dL"
    #     },
    #     {
    #         "name": "急性缺血性腦中風",
    #         "case": "65歲女性患者，糖尿病史15年，最近出現胸悶、呼吸困難，特別是活動時加重。心電圖顯示ST段壓低。請問可能的診斷和處理建議？"
    #     },
    #     {
    #         "name": "出血性食道靜脈瘤", 
    #         "case": "一位 54 歲男性，已知 B 型肝炎相關肝硬化 (Child–Pugh B)，近半年未定期門診追蹤。 今晨 06:30 突發 大量鮮紅色血吐 (≈300 mL)，合併黑糞。 119 送醫，07:05 抵達急診。 到院生命徵象（07:10）： BP 88/50 mmHg，HR 118 bpm，RR 22/min，T 36.6 °C 神智清楚，但明顯口渴、皮膚濕冷 初步處置：立刻給予 O₂ 3 L/min、18G 雙側靜脈路、0.9% NaCl 1 L"
    #     },
    #     {
    #         "name": "張力性氣胸",
    #         "case": "醫生您好，急診收治一位 28 歲男性，無明顯病史，騎機車與汽車側撞後被送至院。 到院時（T0）： BP 80/50 mmHg、HR 132 bpm、RR 34 / min、SpO₂ 86 %（室內空氣） 意識：煩躁不安，斷續喊「喘不過氣」 皮膚：濕冷、蒼白 查體（T + 2 min）： 頭頸：頸靜脈怒張 胸部：右胸無明顯開放性傷口；右側呼吸音幾乎消失，叩診鼓音；氣管輕度左偏 四肢：毛細血管再充盈延遲（>3 秒） 病歷訊息： 救護員報告現場未見大量出血；途中已輸 500 mL 生理食鹽水但血壓無明顯回升。此時最應立即執行下列哪一項處置？"
    #     },
    #     {
    #         "name": "敗血性休克",
    #         "case": "一位 67 歲男性，慢性阻塞性肺病（COPD）病史，近三日咳嗽、黃痰伴發燒。今日上午 08:20 因意識混亂被家屬送至急診。 到院（08:40）生命徵象 BP 78/40 mmHg（Mean ≈ 53 mmHg） HR 122 bpm RR 28 /min T 39.2 °C SpO₂ 92 %（面罩 6 L/min），已完成初步輸液（總量 30 mL/kg）後仍血壓偏低，以下何者為第一線血管升壓劑以維持目標平均動脈壓（MAP ≥ 65 mmHg）？"
    #     },
    #     {
    #         "name": "透析低血壓案例",
    #         "case": "醫生您好，我是病人背景 55 歲男性，終末期腎病（ESRD），血液透析 3 年，每週 3 次 主訴 最近兩週透析中頻繁頭暈、噁心，血壓最低 80/50 mmHg。請問我得了什麼病呢？應該怎麼處理？"
    #     },
    #     {
    #         "name": "心血管案例",
    #         "case": "65歲女性患者，糖尿病史15年，最近出現胸悶、呼吸困難，特別是活動時加重。心電圖顯示ST段壓低。請問可能的診斷和處理建議？"
    #     },
    # ]
    
    # === 執行推理測試 ===
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'🏥 ' + '=' * 20} 測試案例 {i}: {test_case['name']} {'=' * 20}")
        print(f"📝 病例描述: {test_case['case']}")
        print("\n" + "=" * 70)
        
        try:
            # 執行兩步推理
            results = two_step_inference(model, tokenizer, test_case['case'])
            
            # 顯示最終結果
            final_response = results["second_step"]["response"]
            
            print(f"\n🎯 最終醫學建議:")
            print("=" * 70)
            print(final_response)
            print("=" * 70)
            
            print(f"\n📊 統計資訊:")
            print(f"   - 第一步生成token數: {results['first_step']['token_count']}")
            print(f"   - 第二步生成token數: {results['second_step']['token_count']}")
            print(f"   - 總處理時間: {results['first_step']['generation_time'] + results['second_step']['generation_time']:.2f}秒")
            
            # 如果需要翻譯（檢查是否有英文內容）
            if any(ord(char) < 128 and char.isalpha() for char in final_response):
                print(f"\n🌏 偵測到英文內容，是否需要翻譯？（這裡可以添加翻譯功能）")
            
            print("\n" + "🔄 " + "=" * 68 + "\n")
            
        except Exception as e:
            print(f"❌ 處理案例 {i} 時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    print("🎉 所有測試完成！")

# === Step 7: 互動式推理函數 ===
def interactive_inference():
    """互動式推理模式"""
    
    print("🚀 載入模型...")
    model, tokenizer = load_model()
    
    if model is None:
        return
    
    print("\n💬 進入互動模式（輸入 'quit' 結束）")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👨‍⚕️ 請描述病例: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再見！")
                break
            
            if not user_input:
                print("⚠️  請輸入有效的病例描述")
                continue
            
            print(f"\n🔍 正在分析病例...")
            
            # 執行推理
            results = two_step_inference(model, tokenizer, user_input)
            
            print(f"\n🎯 醫學建議:")
            print("-" * 50)
            print(results["second_step"]["response"])
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 程式中斷，再見！")
            break
        except Exception as e:
            print(f"❌ 處理時出錯: {e}")

# === Step 8: 命令列選項 ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_inference()
    else:
        main()
