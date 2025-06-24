from flask import Flask, render_template, request, jsonify, stream_template
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
from transformers import AutoTokenizer
import time
import threading

app = Flask(__name__)

# 全局變量存儲模型
model = None
tokenizer = None
model_loaded = False

# === 模型配置 ===
model_path = "./deepseek-math-7b-orpo-lora-continued"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

# === Prompt 模板 ===
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

def load_model():
    """載入模型"""
    global model, tokenizer, model_loaded
    try:
        print("🚀 開始載入模型...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch_dtype,
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        FastLanguageModel.for_inference(model)
        model.to(device)
        model_loaded = True
        print("✅ 模型載入成功！")
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        model_loaded = False

def generate_medical_response(patient_case, chain_of_thought="", max_new_tokens=400):
    """生成醫學回答"""
    if not model_loaded:
        return {"error": "模型尚未載入完成"}
    
    prompt = alpaca_prompt.format(patient_case, chain_of_thought, "")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
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
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, **generation_kwargs)
        end_time = time.time()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()
    
    return {
        "response": generated_text,
        "generation_time": end_time - start_time,
        "token_count": len(outputs[0]) - len(inputs["input_ids"][0])
    }

def two_step_inference(patient_case):
    """執行兩步推理"""
    if not model_loaded:
        return {"error": "模型尚未載入完成"}
    
    # 第一步推理
    first_question = patient_case + " 請根據上述症狀進行分析和思考。"
    first_result = generate_medical_response(first_question, "", 300)
    
    if "error" in first_result:
        return first_result
    
    first_response = first_result["response"]
    answer_start = first_response.find("### Medical Assessment:")
    if answer_start != -1:
        first_analysis = first_response[answer_start + len("### Medical Assessment:"):].strip()
    else:
        first_analysis = first_response
    
    # 第二步推理
    second_result = generate_medical_response(patient_case, first_analysis, 400)
    
    if "error" in second_result:
        return second_result
    
    return {
        "first_step": first_result,
        "second_step": second_result,
        "first_analysis": first_analysis,
        "final_response": second_result["response"],
        "total_time": first_result["generation_time"] + second_result["generation_time"]
    }

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patient_case = data.get('patient_case', '').strip()
        
        if not patient_case:
            return jsonify({"error": "請輸入病例描述"})
        
        if not model_loaded:
            return jsonify({"error": "模型尚未載入完成，請稍後再試"})
        
        # 執行推理
        result = two_step_inference(patient_case)
        
        if "error" in result:
            return jsonify(result)
        
        return jsonify({
            "success": True,
            "final_response": result["final_response"],
            "generation_time": f"{result['total_time']:.2f}",
            "first_step_tokens": result["first_step"]["token_count"],
            "second_step_tokens": result["second_step"]["token_count"]
        })
        
    except Exception as e:
        return jsonify({"error": f"處理請求時出錯: {str(e)}"})

@app.route('/status')
def status():
    return jsonify({"model_loaded": model_loaded})

# if __name__ == '__main__':
#     # 在背景線程中載入模型
#     threading.Thread(target=load_model, daemon=True).start()
    
#     # 啟動 Flask 應用，綁定到所有網路介面
#     app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    # 在背景線程中載入模型
    threading.Thread(target=load_model, daemon=True).start()
    
    # 使用已映射的端口 8080 或 8888
    app.run(host='0.0.0.0', port=8080, debug=False)  # 使用 8080
    # 或者
    # app.run(host='0.0.0.0', port=8888, debug=False)  # 使用 8888

