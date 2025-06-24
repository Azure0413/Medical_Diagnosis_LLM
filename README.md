# Medical_Diagnosis_LLM  

### Introduction  
隨著人工智慧技術的快速發展，特別是大型語言模型（Large Language Models, LLMs）在自然語言處理領域的突破，將其應用於醫學診斷輔助成為一個極具潛力的方向。本研究旨在透過對LLM進行針對腎臟相關疾病的微調（fine-tuning），使模型能根據病患的描述與症狀，提供初步的診斷判斷及治療參考。由於腎臟疾病的複雜性與專業性，我們與腎臟科醫生密切合作，確保模型的診斷結果具備臨床可行性與準確性。此方法不僅提升了模型在特定醫學領域的專業度，也為未來結合AI與臨床決策支持系統奠定基礎，期望能在醫療資源有限的環境下，提供更有效率且精準的診斷輔助工具。  

![image](https://github.com/Azure0413/Medical_Diagnosis_LLM/blob/main/src/model.png)  

### Training data  
[Click here to download the complete_data folder](https://drive.google.com/drive/folders/1xt8L3lgNJeakXneSL71Nk-NECKxhnXzP)  

### Model Weights  
[Click here to download three deepseek-math folder](https://drive.google.com/drive/folders/1xt8L3lgNJeakXneSL71Nk-NECKxhnXzP)  

### Implementation （Training） 
1. Use `python OPRO.py` or `ORPO_better.py`  

### Implementation （Inference） 
1. Please download the Model Weights for inference.  
2. Use `python app.py` to open the website.  
if you only want to inference to check the model output in the terminal, Use `python inference.py`.

###  Resource（My Slide）    
[My Slide Click here](https://docs.google.com/presentation/d/1LieUY-IJSg18fvF3PCINxAdSzGjXkOjI0S_ZuxBNNuU/edit?usp=sharing)  
