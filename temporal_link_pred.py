"""Temporal link prediction：用生成圖訓練，用真實圖測試。

各模型生成的序列當訓練集、原始序列當測試集，比較哪一組生成資料
訓練出來的模型在真實資料上表現較好。生成品質越高，學到的時序規律
越接近真實，測試分數就越高。

任務定義
    給定前 W 張 snapshot，預測第 W+1 張裡某個節點對之間有沒有邊。
    正樣本有兩種定義，兩種都算：
      accuracy / AUC          該張實際存在的所有邊
      accuracy_new / AUC_new  前 W 張都沒有、這一張才出現的邊
    前者被邊的持續性主導——真實序列有 65~88% 的邊在前一張就存在，
    而特徵裡的 n_seen / recency 直接編碼了那件事，所以所有模型都會
    貼著上限（舊表裡連產出是同一張圖複製 32 次的 GraphMaker 都拿 0.9357）。
    後者才是要預測的東西。負樣本都是從不存在的節點對等量抽樣。

特徵（純結構，不依賴節點屬性，五個模型都適用）
    共同鄰居數、Adamic-Adar、Jaccard、偏好連接、兩端的 degree、
    以及這條邊在前 W 張裡出現過幾次、上次出現距今幾張。
    最後兩項是時序資訊，靜態的 link prediction 沒有。

模型是 logistic regression。刻意選簡單的——要量的是「訓練資料的品質」，
模型太強會把資料差異吃掉。

執行：
    python refs/tools/temporal_link_pred.py --results <test_and_generated_graphs>
    python refs/tools/temporal_link_pred.py --results ... --only wiki_vote
"""
import argparse
import glob
import json
import math
import os
import pickle
import re

import numpy as np

import gm_loader

MODELS = ["DAMNET", "AGE", "DYMOND"]
DATASETS = ("wiki_vote", "twitter", "superuser")
MODES = ["burst_hysteresis", "hysteresis", "burst", "raw"]
MODE_ALT = "|".join(MODES)
WINDOW = 4          # 用前幾張預測下一張
MAX_SEQ = 40        # 每組最多取幾條序列，控制執行時間
NEG_RATIO = 1.0     # 負樣本對正樣本的比例


def neighbors(graphs):
    """每張 snapshot 的鄰居集合，索引對得回 graphs。"""
    return [{n: set(g.neighbors(n)) for n in g.nodes()} for g in graphs]


def features(hist_nbrs, hist_edges, u, v):
    """前 W 張的結構與時序特徵。"""
    # 結構特徵取最後一張
    nb = hist_nbrs[-1]
    su, sv = nb.get(u, set()), nb.get(v, set())
    common = su & sv
    union = su | sv
    du, dv = len(su), len(sv)

    aa = sum(1.0 / math.log(len(nb.get(w, set())) + 1e-9)
             for w in common if len(nb.get(w, set())) > 1)
    jac = len(common) / len(union) if union else 0.0

    # 時序特徵：這條邊在前 W 張出現過幾次、上次出現距今幾張
    key = (u, v) if u < v else (v, u)
    seen = [i for i, es in enumerate(hist_edges) if key in es]
    n_seen = len(seen)
    recency = (len(hist_edges) - seen[-1]) if seen else len(hist_edges) + 1

    return [len(common), aa, jac, du * dv, du + dv, abs(du - dv),
            n_seen, recency]


def build_xy(seqs, rng, max_seq, max_pairs=4000, new_only=False):
    """把序列轉成 (X, y)。每條序列的每個時間點各取一批正負樣本。"""
    X, y = [], []
    for ts in seqs[:max_seq]:
        T = len(ts)
        if T <= WINDOW:
            continue
        nbrs = neighbors(ts)
        edges = [{(a, b) if a < b else (b, a) for a, b in g.edges()}
                 for g in ts]
        nodes = sorted(ts[0].nodes())
        n = len(nodes)

        for t in range(WINDOW, T):
            if new_only:
                # 前 W 張都沒有、這一張才出現的邊。舊定義把已經存在的邊
                # 也算成正樣本，那部分靠 recency 就能答對，會把分數灌到
                # 所有模型都貼著上限。
                seen = set().union(*edges[t - WINDOW:t]) if WINDOW else set()
                pos = list(edges[t] - seen)
            else:
                pos = list(edges[t])
            if not pos:
                continue
            if len(pos) > max_pairs:
                pos = [pos[i] for i in
                       rng.choice(len(pos), max_pairs, replace=False)]
            hn, he = nbrs[t - WINDOW:t], edges[t - WINDOW:t]

            for u, v in pos:
                X.append(features(hn, he, u, v))
                y.append(1)

            # 負樣本：隨機抽不存在的節點對
            n_neg = int(len(pos) * NEG_RATIO)
            tries = 0
            got = 0
            while got < n_neg and tries < n_neg * 20:
                tries += 1
                a, b = nodes[rng.integers(n)], nodes[rng.integers(n)]
                if a == b:
                    continue
                k = (a, b) if a < b else (b, a)
                if k in edges[t]:
                    continue
                X.append(features(hn, he, k[0], k[1]))
                y.append(0)
                got += 1
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def fit_predict(Xtr, ytr, Xte, yte):
    """logistic regression，回傳 (accuracy, AUC)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan"), float("nan")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)

    Xte_s = sc.transform(Xte)
    acc = float((clf.predict(Xte_s) == yte).mean())
    try:
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte_s)[:, 1]))
    except ValueError:
        auc = float("nan")
    return acc, auc


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--modiff", default="",
                    help="MoDiff 的 modiff_generated 目錄。它的目錄名是 "
                         "MoDiff_macro_<資料集>_<mode>_<層>_<seed>，"
                         "與另外三個的 macro_<資料集>_<層>_<mode>_<seed> 不同")
    ap.add_argument("--only", default="")
    ap.add_argument("--models", default="", help="只算這幾個模型（逗號分隔），例如 GraphMaker。分批算完再把 CSV 併起來，不必為了補一個模型把其他的重算一遍")
    ap.add_argument("--seed-tag", default="0")
    ap.add_argument("--modiff-seed", default="",
                    help="MoDiff 產出的 seed 編號，預設與 --seed-tag 相同。"
                         "要拿不同 seed 的 MoDiff 對同一份真實序列比較時才需要")
    ap.add_argument("--graphmaker", default="",
                    help="GraphMaker 的 gm_generated 目錄")
    ap.add_argument("--gm-variant", default=gm_loader.VARIANT)
    ap.add_argument("--gm-seed", default="",
                    help="GraphMaker 產出的 seed，預設與 --seed-tag 相同")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    kre = re.compile(r"^macro_(.+?)_(" + MODE_ALT + r")_(\d+)$")
    rows = []

    print(f"{'組合':40s} {'訓練來源':10s} {'訓練樣本':>9s} "
          f"{'accuracy':>9s} {'AUC':>8s}")

    for d in sorted(glob.glob(os.path.join(args.results,
                                           f"macro_*_{args.seed_tag}"))):
        key = os.path.basename(d)
        if args.only and args.only not in key:
            continue
        m = kre.match(key)
        if not m:
            continue

        src_p = os.path.join(d, "DAMNET", "test_graphs.pkl")
        if not os.path.exists(src_p):
            continue

        # 測試集固定是真實序列
        rng = np.random.default_rng(args.rng_seed)
        _src = load(src_p)
        Xte, yte = build_xy(_src, rng, args.max_seq)
        Xte_n, yte_n = build_xy(_src, rng, args.max_seq, new_only=True)
        if not len(Xte):
            continue

        # MoDiff 的產出在另一個目錄，命名順序也不同
        md_dir = ""
        if args.modiff:
            mk = re.match(r"^macro_(" + "|".join(DATASETS) + r")_(.+?)_("
                          + MODE_ALT + r")_(\d+)$", key)
            if mk:
                cand = os.path.join(
                    args.modiff,
                    f"MoDiff_macro_{mk.group(1)}_{mk.group(3)}_{mk.group(2)}"
                    f"_{args.modiff_seed or mk.group(4)}", "MoDiff")
                if os.path.exists(os.path.join(cand, "sampled_ts.pkl")):
                    md_dir = cand

        # GraphMaker 的目錄名順序又不同，而且只涵蓋各資料集的一個層
        gm_seqs = None
        if args.graphmaker:
            gk = re.match(r"^macro_(" + "|".join(DATASETS) + r")_(.+?)_("
                          + MODE_ALT + r")_(\d+)$", key)
            if gk:
                gm_seqs = gm_loader.find(
                    args.graphmaker, gk.group(1), gk.group(3), gk.group(2),
                    args.gm_seed or gk.group(4), args.gm_variant)

        names = (MODELS + (["MoDiff"] if md_dir else [])
                 + (["GraphMaker"] if gm_seqs else []) + ["真實（對照）"])
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        if want:
            # 真實（對照）是上限，任何情況都要留著才有得比。
            names = [x for x in names if x == "真實（對照）" or x in want]
        for mm in names:
            if mm == "真實（對照）":
                # 上限：用真實資料訓練，同一份資料上測試
                Xtr, ytr = Xte, yte
                Xtr_n, ytr_n = Xte_n, yte_n
            elif mm == "GraphMaker":
                rng = np.random.default_rng(args.rng_seed)
                Xtr, ytr = build_xy(gm_seqs, rng, args.max_seq)
                Xtr_n, ytr_n = build_xy(gm_seqs, rng, args.max_seq, new_only=True)
                if not len(Xtr):
                    continue
            else:
                p = (os.path.join(md_dir, "sampled_ts.pkl") if mm == "MoDiff"
                     else os.path.join(d, mm, "sampled_ts.pkl"))
                if not os.path.exists(p):
                    continue
                rng = np.random.default_rng(args.rng_seed)
                _g = load(p)
                Xtr, ytr = build_xy(_g, rng, args.max_seq)
                Xtr_n, ytr_n = build_xy(_g, rng, args.max_seq, new_only=True)
                if not len(Xtr):
                    continue

            acc, auc = fit_predict(Xtr, ytr, Xte, yte)
            acc_n, auc_n = fit_predict(Xtr_n, ytr_n, Xte_n, yte_n)
            print(f"{key if mm == MODELS[0] else '':40s} {mm:10s} "
                  f"{len(Xtr):>9d} {acc:>9.4f} {auc:>8.4f}"
                  f"{acc_n:>9.4f} {auc_n:>8.4f}", flush=True)
            rows.append(dict(組合=m.group(1), 模式=m.group(2), 訓練來源=mm,
                             訓練樣本=len(Xtr), accuracy=acc, AUC=auc,
                             accuracy_new=acc_n, AUC_new=auc_n))
        print()

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"寫出 {args.csv}（{len(rows)} 列）")

    # 各來源的中位數
    if rows:
        import statistics as st
        print()
        print(f"{'訓練來源':12s} {'n':>4s} {'accuracy 中位':>13s} {'AUC 中位':>10s}")
        for mm in MODELS + ["MoDiff", "GraphMaker", "真實（對照）"]:
            a = [r["accuracy"] for r in rows
                 if r["訓練來源"] == mm and r["accuracy"] == r["accuracy"]]
            u = [r["AUC"] for r in rows
                 if r["訓練來源"] == mm and r["AUC"] == r["AUC"]]
            if a:
                print(f"{mm:12s} {len(a):>4d} {st.median(a):>13.4f} "
                      f"{st.median(u):>10.4f}")


if __name__ == "__main__":
    main()
