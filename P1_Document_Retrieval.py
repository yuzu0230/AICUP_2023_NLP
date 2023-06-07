### PART 1. Document retrieval ###

# ===== import all library =====
# built-in libs
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# 3rd party libs
import hanlp
import opencc
import pandas as pd
import wikipedia
from hanlp.components.pipeline import Pipeline
from pandarallel import pandarallel

# our own libs
from utils import load_json

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
wikipedia.set_lang("zh")
# ==============================

# parameters
page_num = 7 

# ========== Preload the data. ==========
def load_json(file_path: Union[Path, str]) -> pd.DataFrame:
    """jsonl_to_df read jsonl file and return a pandas DataFrame.
    Args: file_path (Union[Path, str]): The jsonl file path.
    Returns: pd.DataFrame: The jsonl file content.
    """
    with open(file_path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]

# TRAIN_DATA = load_json("data/public_train.jsonl")
# TEST_DATA = load_json("data/public_test.jsonl")
# TEST_DATA = load_json("data/private_test_data.jsonl")

TRAIN_DATA = load_json("data/public_train_all.jsonl")
TEST_DATA = load_json("data/public_private_test_all.jsonl")
# =======================================

# Data class for type hinting
@dataclass
class Claim:
    data: str

@dataclass
class AnnotationID:
    id: int

@dataclass
class EvidenceID:
    id: int

@dataclass
class PageTitle:
    title: str

@dataclass
class SentenceID:
    id: int

@dataclass
class Evidence:
    data: List[List[Tuple[AnnotationID, EvidenceID, PageTitle, SentenceID]]]

# ========== Helper function ==========
# For the sake of consistency, we convert traditional to simplified Chinese first before converting it back to traditional Chinese.  
# This is due to some errors occuring when converting traditional to traditional Chinese.

CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

# 簡繁轉換：先把資料全部轉成簡體，之後return的時候再轉回繁體
def do_st_corrections(text: str) -> str: 
    simplified = CONVERTER_T2S.convert(text)
    return CONVERTER_S2T.convert(simplified)

# We use constituency parsing to separate part of speeches or so called constituent to extract noun phrases.  
# In the later stages, we will use the noun phrases as the query to search for relevant documents.  

# 從句子中提取出名詞短語 ("NP"，Noun Phrase) 
# using constituency parsing
def get_nps_hanlp(predictor: Pipeline, d: Dict[str, Union[int, Claim, Evidence]]) -> List[str]:
    claim = d["claim"]
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves())) # 簡繁轉換
        for subtree in tree.subtrees(lambda t: t.label() == "NP")  # 只取詞性為"NP"的詞
    ]
    return nps

# ===== Caulate Precision and Recall =====
# Precision（精確度）: 在所有被檢索到的文檔中，有多少實際上是相關的。
# Recall（召回率）: 在所有應該被找到的相關文檔中，有多少被成功找到。

def calculate_precision(data: List[Dict[str, Union[int, Claim, Evidence]]], predictions: pd.Series,) -> None:
    precision = 0
    count = 0
    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue
        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)
        count += 1
    # Macro precision
    print(f"Precision: {precision / count}")

def calculate_recall(data: List[Dict[str, Union[int, Claim, Evidence]]], predictions: pd.Series,) -> None:
    recall = 0
    count = 0
    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue
        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1
    print(f"Recall: {recall / count}")

#==========================================

"""
The default amount of documents retrieved is at most five documents.  
This `num_pred_doc` can be adjusted based on your objective.  
Save data in jsonl format.
"""
def save_doc(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
    mode: str = "train",
    num_pred_doc: int = page_num, # 5,
) -> None:
    with open(f"data/{mode}_doc{num_pred_doc}.jsonl","w",encoding="utf8",) as f:
        for i, d in enumerate(data):
            d["predicted_pages"] = list(predictions.iloc[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

""" 
Step 1. Get noun phrases from hanlp consituency parsing tree
Setup [HanLP](https://github.com/hankcs/HanLP) predictor
"""

# 把句子切斷，分別安排詞性
predictor = (hanlp.pipeline().append(
    hanlp.load("FINE_ELECTRA_SMALL_ZH"),
    output_key="tok",
).append(
    hanlp.load("CTB9_CON_ELECTRA_SMALL"),
    output_key="con",
    input_key="tok",
))

# creating parsing tree
hanlp_file = f"data/hanlp_con_results.pkl"
if Path(hanlp_file).exists():
  print(f'{hanlp_file} exist')
  with open(hanlp_file, "rb") as f:
      hanlp_results = pickle.load(f)
else:
  print(f'{hanlp_file} does not exist')
  hanlp_results = [get_nps_hanlp(predictor, d) for d in TRAIN_DATA]
  with open(hanlp_file, "wb") as f:
    pickle.dump(hanlp_results, f)

print('Lenth of hanlp_result: ', len(hanlp_results))
# print('hanlp_results', hanlp_results)

# ===== 取得出現頻率最高的前 N 個名詞短語 =====
# from collections import Counter

# train_df = pd.DataFrame(TRAIN_DATA)

# train_df.loc[:, "hanlp_results"] = hanlp_results
# nps = train_df["hanlp_results"]
# N = 200  # 可調整
# np_counter = Counter([np for nps in train_df["hanlp_results"] for np in nps])

# top_nps = [np for np, _ in np_counter.most_common(N)]
# print(top_nps[:25])  # 200
#============================================


# Main function for document retrieval

# ========== common list ==========
common_list = ['人', '中國', '臺灣', '美國', '年', '他', '世界', '之一', '其', '日本', '香港', '國家', '民族', 
        '政府', '希臘', '地區', '其中', '英國', '歐洲', '總統', '概念', '人類', '神話', '電影', '世紀', 
        '國', '馬克思', '時期', '語言', '作品', '自己', '族羣', '城市', '動物', '皇帝', '亞洲', '法國', 
        '學校', '目前', '大學', '者', '一', '生物', '昆蟲', '獎', '人羣', '德國', '衛星', '同一族羣', 
        '墨西哥', '印度', '羅馬', '人員', '面積', '人口', '政權', '時代', '女性', '它', '戰爭', '全球', 
        '中學', '韓國', '功能', '海', '目', '屬', '現在', '部分', '傳', '期間', '名', '木', '影響', '現今', 
        '年代', '名稱', '國別', '階段', '兒子', '稱號', '全國', '發展', '語', '俄羅斯', '物質', '關係', '王', 
        '地', '詞', '南部', '字', '核', '小時', '人物', '規模', '區', '漢', '水', '臺', '地方', '衛', 
        '時間', '文化', '表面', '內容', '目的', '當時', '日', '她', '他們', '子', '名字', '市', '早期' ]

common_set = set(common_list)  # 轉換成 set 以提高檢查速度
# =================================

def get_pred_pages(series_data: pd.Series) -> Set[Dict[int, str]]:
    results = []
    tmp_muji = []
    # wiki_page: its index showned in claim
    mapping = {}
    claim = series_data["claim"]
    nps = series_data["hanlp_results"]
    first_wiki_term = []

    for i, np in enumerate(nps):
        '''
        只對一部分的名詞短語進行搜索，將常見且沒有意義的詞過濾。
        可縮小搜索範疇，進而提高程式的效能和結果的精確度。
        '''
        if np not in common_set:
          # Simplified Traditional Chinese Correction
          wiki_search_results = [
              do_st_corrections(w) for w in wikipedia.search(np)
          ]
          # Remove the wiki page's description in brackets (去後綴)
          wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]

          wiki_df = pd.DataFrame({
              "wiki_set": wiki_set,  
              "wiki_results": wiki_search_results  
          })
          # Elements in wiki_set --> index
          # Extracting only the first element is one way to avoid extracting too many of the similar wiki pages
          grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
          candidates = grouped_df["wiki_results"].tolist()
          # print(candidates)

          # muji refers to wiki_set
          muji = grouped_df.index.tolist()

          for prefix, term in zip(muji, candidates):
              if prefix not in tmp_muji:
                  matched = False

                  # Take at least one term from the first noun phrase
                  if i == 0:
                      first_wiki_term.append(term)

                  # Walrus operator :=
                  # https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
                  # Through these filters, we are trying to figure out if the term is within the claim
                  if (((new_term := term) in claim) or
                      ((new_term := term.replace("·", "")) in claim) or
                      ((new_term := term.split(" ")[0]) in claim) or
                      ((new_term := term.replace("-", " ")) in claim)):
                      matched = True
                  
                  # 額外處理：針對外國人名
                  elif "·" in term:
                      splitted = term.split("·")
                      for split in splitted:
                          if (new_term := split) in claim:
                              matched = True
                              break
                  
                  # wiki的page跟claim有關
                  if matched:
                      # post-processing
                      term = term.replace(" ", "_")
                      term = term.replace("-", " ")
                      results.append(term)
                      mapping[term] = claim.find(new_term)
                      tmp_muji.append(new_term)

    # page_num: 可調整page數量(取大一點)
    if len(results) >= page_num:
        assert -1 not in mapping.values()
        results = sorted(mapping, key=mapping.get)[:page_num] 
    elif len(results) < 1:
        results = first_wiki_term

    return set(results)


# Get pages via wiki online api
doc_path = f"data/train_doc{page_num}.jsonl"  # 5
if Path(doc_path).exists():
  print(f'{doc_path} exist')
  with open(doc_path, "r", encoding="utf8") as f:
    predicted_results = pd.Series([
        set(json.loads(line)["predicted_pages"])
        for line in f
    ])
else:
  print(f'{doc_path} does not exist')
  train_df = pd.DataFrame(TRAIN_DATA)
  train_df.loc[:, "hanlp_results"] = hanlp_results
  print('processing...')
  predicted_results = train_df.parallel_apply(get_pred_pages, axis=1)
  save_doc(TRAIN_DATA, predicted_results, mode="train")
  print('Finished!')

### Step 2. Calculate our results ###
calculate_precision(TRAIN_DATA, predicted_results) # Precision: 0.262
calculate_recall(TRAIN_DATA, predicted_results)   # Recall: 0.852

### Step 3. Repeat the same process on test set Create parsing tree ###
hanlp_test_file = f"data/hanlp_con_results_test.pkl"
if Path(hanlp_test_file).exists():
  print(f'{hanlp_test_file} exist')
  with open(hanlp_test_file, "rb") as f:
      hanlp_results_test = pickle.load(f)
else:
  print(f'{hanlp_test_file} does not exist')
  hanlp_results_test = [get_nps_hanlp(predictor, d) for d in TEST_DATA]
  with open(hanlp_test_file, "wb") as f:
      pickle.dump(hanlp_results_test, f)

### Get pages via wiki online api ###
test_doc_path = f"data/test_doc{page_num}.jsonl" # 5
if Path(test_doc_path).exists():
    print(f'{test_doc_path} exist')
    with open(test_doc_path, "r", encoding="utf8") as f:
        test_results = pd.Series(
            [set(json.loads(line)["predicted_pages"]) for line in f])
else:
    print(f'{test_doc_path} does not exist')
    
    test_df = pd.DataFrame(TEST_DATA)
    test_df.loc[:, "hanlp_results"] = hanlp_results_test
    test_results = test_df.parallel_apply(get_pred_pages, axis=1)
    save_doc(TEST_DATA, test_results, mode="test")

# print('hanlp_results_test', hanlp_results_test)
