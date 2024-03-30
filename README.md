# AI CUP 2023 Fall NLP
- Competitions: 「真相只有一個: 事實文字檢索與查核競賽」
- 「T-Brain AI實戰吧」官網: https://tbrain.trendmicro.com.tw/Competitions/Details/28\
- Final Report: https://docs.google.com/document/d/1ng73ByqHX7_qwKRT675fDIL1vcdK2IR_Gn2ycHGpddE/mobilebasic


## 運行環境

### 使用 Anaconda 建立環境
```bash
conda env create -f enviroment.yml
conda activate AICUP_2023
```

## 資料
- `data/public_train.jsonl`: 官方提供的訓練資料 (v1)
- `data/public_train_v2.jsonl`: 官方提供的訓練資料 (v2)
- `data/public_train_all.jsonl`: 官方提供的訓練資料 (v1 + v2)
- `data/public_test.jsonl`：官方提供的測試資料 (public)
- `data/private_test_data`：官方提供的測試資料 (private)
- `data/public_private_test_all`：官方提供的測試資料 (public + privte)

### 已經完成訓練的模型權重 (checkpoint) 之下載連結：
- Link: https://drive.google.com/drive/folders/1qRS0XKUKVorXoQFNUWEdMb4USqQqPebP?usp=sharing
- Sentence Retrival: model.350.pt  
- Claim verification: val_acc=0.5099_model.1900.pt


## 程式架構

### PART 1. Document retrieval： 
- We use constituency parsing to separate part of speeches or so called constituent to extract noun phrases. In the later stages, we will use the noun phrases as the query to search for relevant documents.
- Caulate Precision and Recall
    - Precision: Among all the documents retrieved, how many are actually relevant.
    - Recall: Among all the relevant documents that should have been found, how many were successfully retrieved.
- Main function for document retrieval
    - Step 1. Get noun phrases from hanlp consituency parsing tree 
        - Setup HanLP predictor
        - creating parsing tree
        - Get pages via wiki online api
    - Step 2. Calculate our results (f1, precision, recall)
    - Step 3. Repeat the same process on test set
        - creating parsing tree
        - Get pages via wiki online api

### PART 2. Sentence retrieval：
- Step 1. Setup training environment
- Step 2. Combine claims and evidences
- Step 3. Start training
- Step 4. Check on our test data

### PART 3. Claim verification：
- Step 1. Setup training environment
- Step 2. Concat claim and evidences
- Step 3. Training
- Step 4. Make submission: Prediction and Write files
