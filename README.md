# AI CUP 2023 Fall NLP

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
- `public_test.jsonl`：官方提供的測試資料 (public)
- `private_test_data`：官方提供的測試資料 (private)
- `public_private_test_all`：官方提供的測試資料 (public + privte)

## 程式架構

### PART 1. Document retrieval： 
- We use constituency parsing to separate part of speeches or so called constituent to extract noun phrases. In the later stages, we will use the noun phrases as the query to search for relevant documents.
- Caulate Precision and Recall
    - Precision: 在所有系統檢索到的文檔中，系統找到多少相關的文檔
    - Recall：在所有相關文檔中，系統找到多少相關的文檔
- Main function for document retrieval
    - Step 1. Get noun phrases from hanlp consituency parsing tree 
        - Setup HanLP predictor (1 min)
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
- Step 4. Check on our test data (5 min)

### PART 3. Claim verification：
- Step 1. Setup training environment
- Step 2. Concat claim and evidences
    - join topk evidence
- Step 3. Training
- Step 4. Make your submission: Prediction and Write files
