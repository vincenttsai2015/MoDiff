"""從生成的拓撲補上節點屬性。

五個 baseline 的輸出只有鄰接矩陣，論文 Sec. 4.4 那組指標（R、Var(ΔR)、C(t)、H）
建立在節點屬性上，沒有屬性就算不出來。這支在拓撲之後接一個
`p(X_t | structural state)`，五個模型共用同一份，差異才會落在拓撲品質上。

估計方式是經驗條件分佈——非參數、沒有額外訓練的模型，也就沒有「替某個 baseline
做了特別強的 attribute model」的疑慮。

structural state 取「當下 degree」與「累積 degree」兩個分箱：

  - 當下 degree 對應 latent state 的瞬時強度
  - 累積 degree 才追得到 hysteresis。屬性的雙門檻 z_t 是路徑相依的
    （z_t = 1 if s>=0.7, 0 if s<=0.3, else z_{t-1}），只看當下那張圖的條件分佈
    在結構上產不出記憶，實測 H 只還原得到兩成

**不把 t 放進特徵**。事件峰值固定落在 t in [8, 14]，條件在 t 上等於讓解碼器從
訓練集學到「大家在 t=10 附近活躍」，然後不管拓撲長什麼樣都複製一次 burst。
實測用全空的圖也能還原 burst 的 H 到五成五，那樣指標就沒有鑑別力了。

參數只在訓練切分上估。`npz_to_baseline.py` 保留 `window_index` 排序，
baseline 取最後 10% 當 test，所以這裡取前 80%。

執行：
    python refs/tools/attribute_decoder.py --results <解開的 test_and_generated_graphs>
    python refs/tools/attribute_decoder.py --results ... --only wiki_vote
"""
import argparse
import glob
import json
import os
import pathlib
import pickle
import re
import sys

import numpy as np

import gm_loader

HERE = pathlib.Path(os.path.realpath(__file__)).parent
sys.path.insert(0, str(HERE))
import descriptors as D  # noqa: E402

MODELS = ["DAMNET", "AGE", "DYMOND"]
DATASETS = ("wiki_vote", "twitter", "superuser")
MODES = ["raw", "burst", "hysteresis", "burst_hysteresis"]

DEG_BINS = [1, 2, 3, 5, 9]
CUM_BINS = [1, 3, 6, 12, 25]
TRAIN_FRAC = 0.8

_WIN_RE = re.compile(r"_w(\d+)(?:_b(\d+))?\.npz$")


def window_index(path):
    m = _WIN_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (-1, -1)


def bins(deg):
    """(當下 degree 分箱, 累積 degree 分箱)。"""
    return (np.digitize(deg, DEG_BINS),
            np.digitize(np.cumsum(deg, axis=0), CUM_BINS))


# ── 解碼器 ──────────────────────────────────────────────────────────────
def fit(samples):
    """samples: [(degree (T,N), active (T,N))] -> {(deg分箱, 累積分箱): 機率}"""
    hit, tot = {}, {}
    for deg, act in samples:
        b, c = bins(deg)
        for k, a in zip(zip(b.ravel(), c.ravel()), act.ravel()):
            k = (int(k[0]), int(k[1]))
            tot[k] = tot.get(k, 0) + 1
            hit[k] = hit.get(k, 0) + int(a)
    return {k: hit[k] / tot[k] for k in tot}, float(
        sum(hit.values()) / max(1, sum(tot.values())))


def decode(deg, table, fallback, rng):
    """回傳 (T, N) 的 0/1 屬性。沒見過的分箱組合退回邊際機率。"""
    T, N = deg.shape
    b, c = bins(deg)
    p = np.empty((T, N))
    for t in range(T):
        for i in range(N):
            p[t, i] = table.get((int(b[t, i]), int(c[t, i])), fallback)
    return (rng.random((T, N)) < p).astype(np.int8)


# ── 描述子 ──────────────────────────────────────────────────────────────
def descriptors(act):
    """act: (T, N) -> (Var(ΔR), H)。兩通道 one-hot，mask 全 1。"""
    T, N = act.shape
    X = np.zeros((T, 1, N, 2))
    X[:, 0, :, 0] = 1 - act
    X[:, 0, :, 1] = act
    R = D.compute_R(X, np.ones((T, 1, N), dtype=bool))
    return float(D.burst_intensity(R).mean()), float(D.hysteresis_intensity(R).mean())


# ── 讀取 ────────────────────────────────────────────────────────────────
def degree_from_nx(graphs, n_nodes):
    deg = np.zeros((len(graphs), n_nodes), dtype=np.int32)
    for t, g in enumerate(graphs):
        for v, d in g.degree():
            if 0 <= int(v) < n_nodes:
                deg[t, int(v)] = d
    return deg


def load_train(npz_root, dataset, mode, layer, limit):
    """訓練切分的 (degree, ground-truth 屬性)。"""
    paths = sorted(glob.glob(os.path.join(npz_root, "data_processed", dataset,
                                          mode, "*.npz")), key=window_index)
    paths = [p for p in paths
             if json.load(open(p[:-4] + ".json", encoding="utf-8")).get("accepted")]
    paths = paths[:int(len(paths) * TRAIN_FRAC)]
    if limit and len(paths) > limit:
        paths = paths[::max(1, len(paths) // limit)][:limit]

    out = []
    for p in paths:
        z = np.load(p)
        act = z["active"][:, layer, :].astype(np.int8)
        T, N = act.shape
        e = z["edges"]
        sel = e[e[:, 1] == layer]
        deg = np.zeros((T, N), dtype=np.int32)
        for t, _, u, v in sel:
            deg[int(t), int(u)] += 1
            deg[int(t), int(v)] += 1
        out.append((deg, act))
    return out


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


LAYER_OF = {}


def layer_index(npz_root, dataset, layer_name):
    key = (dataset, layer_name)
    if key not in LAYER_OF:
        p = sorted(glob.glob(os.path.join(npz_root, "data_processed", dataset,
                                          "raw", "*.json")))[0]
        names = json.load(open(p, encoding="utf-8"))["layer_names"]
        LAYER_OF[key] = names.index(layer_name) if layer_name in names else None
    return LAYER_OF[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="test_and_generated_graphs 的路徑")
    ap.add_argument("--npz-root",
                    default=os.environ.get("NPZ_ROOT",
                                           str(HERE.parent / "MulDyDiff")),
                    help="序列 npz 的根目錄，用來估解碼器參數")
    ap.add_argument("--only", default="", help="只跑名稱含此字串的組合")
    ap.add_argument("--models", default="", help="只算這幾個模型（逗號分隔），例如 GraphMaker。分批算完再把 CSV 併起來，不必為了補一個模型把其他的重算一遍")
    ap.add_argument("--seed-tag", default="0")
    ap.add_argument("--modiff", default="",
                    help="MoDiff 的 modiff_generated 目錄。它的目錄名是 "
                         "MoDiff_macro_<資料集>_<mode>_<層>_<seed>")
    ap.add_argument("--graphmaker", default="",
                    help="GraphMaker 的 gm_generated 目錄")
    ap.add_argument("--gm-variant", default=gm_loader.VARIANT)
    ap.add_argument("--fit-limit", type=int, default=300,
                    help="估參數最多取幾條訓練序列，0 為全部")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.rng_seed)
    key_re = re.compile(r"^macro_(.+?)_(" + "|".join(MODES) + r")_(\d+)$")

    print(f"{'組合':40s} {'來源':10s} {'Var(ΔR)':>9s} {'對真值':>8s} "
          f"{'H':>8s} {'對真值':>8s}")
    rows = []

    for d in sorted(glob.glob(os.path.join(args.results, f"macro_*_{args.seed_tag}"))):
        key = os.path.basename(d)
        if args.only and args.only not in key:
            continue
        m = key_re.match(key)
        if not m:
            continue
        ds_layer, mode, _ = m.groups()

        # <dataset>_<layer>：dataset 是 npz 目錄名，layer 是剩下的部分
        ds = next((x for x in ("wiki_vote", "twitter", "superuser")
                   if ds_layer.startswith(x)), None)
        if ds is None:
            continue
        layer_name = ds_layer[len(ds) + 1:]
        l = layer_index(args.npz_root, ds, layer_name)
        if l is None:
            print(f"{key:40s} 找不到層 {layer_name}")
            continue

        src_p = os.path.join(d, "DAMNET", "test_graphs.pkl")
        if not os.path.exists(src_p):
            continue

        table, fallback = fit(load_train(args.npz_root, ds, mode, l,
                                         args.fit_limit))

        # 真值：test 切分是排序後的最後一段，條數與 test_graphs.pkl 相同
        src = load_pkl(src_p)
        paths = sorted(glob.glob(os.path.join(args.npz_root, "data_processed",
                                              ds, mode, "*.npz")), key=window_index)
        paths = [p for p in paths
                 if json.load(open(p[:-4] + ".json", encoding="utf-8")).get("accepted")]
        true_v, true_h = [], []
        for p in paths[-len(src):]:
            z = np.load(p)
            v, h = descriptors(z["active"][:, l, :].astype(np.int8))
            true_v.append(v); true_h.append(h)
        tv, th = float(np.mean(true_v)), float(np.mean(true_h))
        print(f"{key:40s} {'真值':10s} {tv:>9.5f} {'':>8s} {th:>8.4f}")
        rows.append(dict(key=key, source="真值", var=tv, h=th))

        # oracle：同一個解碼器餵原始拓撲，當成上限
        cand = [("oracle", src)]
        for mm in MODELS:
            p = os.path.join(d, mm, "sampled_ts.pkl")
            if os.path.exists(p):
                cand.append((mm, load_pkl(p)))
        if args.modiff:
            mp = os.path.join(
                args.modiff,
                f"MoDiff_macro_{ds}_{mode}_{layer_name}_{args.seed_tag}",
                "MoDiff", "sampled_ts.pkl")
            if os.path.exists(mp):
                cand.append(("MoDiff", load_pkl(mp)))
        if args.graphmaker:
            gs = gm_loader.find(args.graphmaker, ds, mode, layer_name,
                                args.seed_tag, args.gm_variant)
            if gs:
                cand.append(("GraphMaker", gs))

        want = {x.strip() for x in args.models.split(",") if x.strip()}
        if want:
            # oracle 是真值對照，任何情況都要留著才有得比。
            cand = [c for c in cand if c[0] == "oracle" or c[0] in want]

        for name, seqs in cand:
            vs, hs = [], []
            for i, ts in enumerate(seqs):
                n = max((max(g.nodes) + 1 if g.number_of_nodes() else 0)
                        for g in ts)
                if n < 2:
                    continue
                dec = decode(degree_from_nx(ts, n), table, fallback, rng)
                v, h = descriptors(dec)
                vs.append(v); hs.append(h)
            if not vs:
                continue
            v_, h_ = float(np.mean(vs)), float(np.mean(hs))
            print(f"{'':40s} {name:10s} {v_:>9.5f} "
                  f"{v_ / tv if tv else 0:>7.2f}x {h_:>8.4f} "
                  f"{h_ / th if th else 0:>7.2f}x")
            rows.append(dict(key=key, source=name, var=v_, h=h_,
                             var_ratio=v_ / tv if tv else None,
                             h_ratio=h_ / th if th else None))
        print()

    if args.csv and rows:
        import csv
        cols = ["key", "source", "var", "var_ratio", "h", "h_ratio"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print(f"寫出 {args.csv}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
