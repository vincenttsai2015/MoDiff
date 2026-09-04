"""用同一套指標評估所有模型，補齊 DAMNETS 家族缺的四個 KS。

DAMNETS 家族的 temporal_baseline_evaluator.py 只算五項（degree / clustering /
spectral MMD 加 node_behavior / random_walk KS），MoDiff 的 evaluation.stats
算十項。這支對所有模型統一用後者，五個模型的數字才在同一套定義上。

GraphMaker 的序列是同一張靜態圖複製 32 次，重複的張數對分佈估計沒有貢獻，
所以只取每條的第一張。它只涵蓋各資料集的一個層，其餘組合沒有它的列。

**GraphMaker 與其他模型不在同一個尺度上**：它學的是整個資料集攤平成的單一大圖，
每張約 6000 節點，而真實序列與另外四個模型是 100–200 節點的窗口。跨模型比較
它的 MMD 與 KS 時要記得這件事。

執行：
    sbatch run_eval_nano4.sh 0

MMD 是 O(n²) 且每次比對要解 EMD，一個 seed 要數小時，所以走 sbatch。
直接跑的話：
    python eval_all_metrics.py --results <test_and_generated_graphs> \
        --modiff <modiff_generated> --graphmaker <gm_generated> \
        --skip-seed 1 --seed 0 --csv out.csv
"""
import argparse
import glob
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# evaluation.stats 在這個 repo 裡，HERE 就是 repo 根目錄。
sys.path.insert(0, HERE)

MODELS = ["DAMNET", "AGE", "DYMOND"]
MODES = ["raw", "burst", "hysteresis", "burst_hysteresis"]
MODE_ALT = "|".join(sorted(MODES, key=len, reverse=True))
DATASETS = ("wiki_vote", "twitter", "superuser")
METHODS = ["degree", "cluster", "spectral", "node_behavior_ks",
           "random_walk_ks", "pagerank_ks", "node_degree_behavior_ks",
           "degree_centrality_behavior_ks",
           "betweenness_centrality_behavior_ks",
           "closeness_centrality_behavior_ks"]
MAX_G = 300      # 每組最多取幾張圖，MMD 是 O(n²)


def ensure_eden():
    """evaluation.mmd 在模組層級 import eden.graph，那只有 nspdk 用得到。

    eden-kernel 的相依常常裝不起來，而我們的十個指標都不需要它。
    MMD 用 ProcessPoolExecutor 平行計算，子行程會重新 import，
    所以替身要落成真的檔案而不只是塞進 sys.modules。
    """
    try:
        import eden.graph  # noqa: F401
        return
    except ImportError:
        pass

    stub = os.path.join(HERE, "_eden_stub")
    os.makedirs(os.path.join(stub, "eden"), exist_ok=True)
    with open(os.path.join(stub, "eden", "__init__.py"), "w") as f:
        f.write("")
    body = [
        "# eden-kernel 的替身，只給 nspdk 用，本專案的指標都不需要",
        "def vectorize(*a, **k):",
        "    raise NotImplementedError('eden-kernel 未安裝')",
        "",
    ]
    with open(os.path.join(stub, "eden", "graph.py"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(body))
    sys.path.insert(0, stub)
    os.environ["PYTHONPATH"] = stub + os.pathsep + os.environ.get(
        "PYTHONPATH", "")


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def gm_graphs(obj):
    """GraphMaker 的產出攤成圖的 list。

    存檔有兩種形式：原始的每條複製 32 張，以及 gm_dedup.py 縮過的每條一張。
    複製的張數對分佈估計沒有貢獻，兩種都只取每條的第一張。
    """
    return [x[0] if isinstance(x, list) else x for x in obj]


def flatten(seqs, cap, skip=0):
    """序列的 list 攤平成圖的 list。

    `skip` 跳過每條開頭幾張。序列用 t0=15 切出來時第 0 張是真圖——
    DAMNETS/AGE 拿它當自迴歸的起點，不是模型產出的，算進分佈會灌水。
    參考序列也要跳過同樣的張數，兩邊才比得起來。
    """
    out = []
    for ts in seqs:
        out.extend(ts[skip:])
        if len(out) >= cap:
            break
    return out[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--modiff", default="")
    ap.add_argument("--graphmaker", default="",
                    help="GraphMaker 的 gm_generated 目錄。它的目錄名是 "
                         "macro_<資料集>_<mode>_<層>_<variant>_seed<N>")
    ap.add_argument("--gm-variant", default="sync",
                    help="要讀 GraphMaker 的哪個 variant")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--only", default="", help="只跑名稱含此字串的組合")
    ap.add_argument("--models", default="",
                    help="只算這幾個模型（逗號分隔），例如 GraphMaker。"
                         "分批算完再把 CSV 併起來，不必為了補一個模型"
                         "把其他四個重算一遍")
    ap.add_argument("--combo", default="",
                    help="只跑這一個組合，精確比對（不含 seed 後綴）。"
                         "burst 是 burst_hysteresis 的前綴，切成平行任務時"
                         "要用這個而不是 --only")
    ap.add_argument("--skip-seed", type=int, default=0,
                    help="每條序列跳過開頭幾張。t0=15 切出來的序列第 0 張是"
                         "自迴歸的真圖起點，不是模型產出，設 1 排除掉")
    ap.add_argument("--max-graphs", type=int, default=MAX_G)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    want = {x.strip() for x in args.models.split(",") if x.strip()}

    ensure_eden()
    from evaluation.stats import eval_graph_list
    from evaluation.mmd import gaussian_emd
    kernels = {"degree": gaussian_emd, "cluster": gaussian_emd,
               "spectral": gaussian_emd}

    kre = re.compile(r"^macro_(.+?)_(" + MODE_ALT + r")_(\d+)$")
    import csv as _csv
    f = open(args.csv, "w", newline="", encoding="utf-8-sig")
    w = _csv.writer(f)
    w.writerow(["組合", "模式", "模型"] + METHODS)

    for d in sorted(glob.glob(os.path.join(args.results,
                                           f"macro_*_{args.seed}"))):
        key = os.path.basename(d)
        if args.only and args.only not in key:
            continue
        if args.combo and key != f"{args.combo}_{args.seed}":
            continue
        m = kre.match(key)
        if not m:
            continue
        ds_layer, mode, _ = m.groups()
        ds = next((x for x in DATASETS if ds_layer.startswith(x)), None)
        layer = ds_layer[len(ds) + 1:] if ds else ""

        src_p = os.path.join(d, "DAMNET", "test_graphs.pkl")
        if not os.path.exists(src_p):
            continue
        ref = flatten(load(src_p), args.max_graphs, args.skip_seed)

        cand = []
        for mm in MODELS:
            if want and mm not in want:
                continue
            p = os.path.join(d, mm, "sampled_ts.pkl")
            if os.path.exists(p):
                cand.append((mm, flatten(load(p), args.max_graphs,
                                         args.skip_seed)))
        if args.modiff and (not want or "MoDiff" in want):
            mp = os.path.join(args.modiff,
                              f"MoDiff_macro_{ds}_{mode}_{layer}_{args.seed}",
                              "MoDiff", "sampled_ts.pkl")
            if os.path.exists(mp):
                cand.append(("MoDiff", flatten(load(mp), args.max_graphs,
                                              args.skip_seed)))
        if args.graphmaker and (not want or "GraphMaker" in want):
            gp = os.path.join(
                args.graphmaker,
                f"macro_{ds}_{mode}_{layer}_{args.gm_variant}"
                f"_seed{args.seed}", "GraphMaker", "sampled_ts.pkl")
            if os.path.exists(gp):
                cand.append(("GraphMaker",
                             gm_graphs(load(gp))[:args.max_graphs]))

        for name, pred in cand:
            print(f"--- {key} / {name} ({len(ref)} vs {len(pred)}) ---",
                  flush=True)
            try:
                r = eval_graph_list(ref, pred, methods=METHODS,
                                    kernels=kernels)
            except Exception as ex:
                print(f"  失敗 {type(ex).__name__}: {ex}", flush=True)
                continue
            w.writerow([ds_layer, mode, name] +
                       [f"{r.get(x, float('nan')):.6f}" for x in METHODS])
            f.flush()

    f.close()
    print(f"寫出 {args.csv}")


if __name__ == "__main__":
    main()
