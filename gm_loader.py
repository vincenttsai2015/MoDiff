"""讀 GraphMaker 的產出，給需要「序列」的分析工具共用。

GraphMaker 生成的是單一靜態圖，沒有時間軸。存檔時每張複製 GM_SEQ_LEN 次
當成「完全不變的序列」，`gm_dedup.py` 又把重複的部分縮掉，所以磁碟上有兩種
形式：序列的 list（原始）與圖的 list（縮過）。這支兩種都讀得動。

展開時用同一個物件重複參照而不是 copy——每張本來就相同，唯讀的分析共用參照
不會有問題，也不佔額外記憶體。

目錄名是 macro_<資料集>_<mode>_<層>_<variant>_seed<N>，
與 baseline 的 macro_<資料集>_<層>_<mode>_<seed> 順序不同。
"""
import os
import pickle

SEQ_LEN = 32
VARIANT = "sync"


def path(root, dataset, mode, layer, seed, variant=VARIANT):
    return os.path.join(
        root, f"macro_{dataset}_{mode}_{layer}_{variant}_seed{seed}",
        "GraphMaker", "sampled_ts.pkl")


def load_seqs(p, seq_len=SEQ_LEN):
    """回傳序列的 list，每條 seq_len 張。"""
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return [x if isinstance(x, list) else [x] * seq_len for x in obj]


def find(root, dataset, mode, layer, seed, variant=VARIANT, seq_len=SEQ_LEN):
    """找得到就回傳序列，否則 None。"""
    p = path(root, dataset, mode, layer, seed, variant)
    return load_seqs(p, seq_len) if os.path.exists(p) else None
