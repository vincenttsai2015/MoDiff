"""四個拓撲指標，三個 seed。

    python macro_topo_eval.py <seed> <results 根目錄>         [modiff_generated] [gm_generated] [每組最多幾條]

輸出 CSV：組合, 模式, 來源, A_t, a_t, D_t, T_recovery, 未回復比例

D(t) 與 T_recovery 需要同窗口的 raw 當對照，用 npz 檔名的 (窗口, 批次) 配對——
各 mode 的接受序列數不同，test 切分裡的索引對不起來。
"""
import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

import gm_loader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macro_topology import (edge_change_activity, active_ratio_from_edges,  # noqa
                            deviation_from_control, edges_from_nx,
                            baseline_tolerance, recovery_time)

# npz 原始序列不在 repo 裡，用 NPZ_ROOT 指到它解開的位置。
NPZ = os.environ.get("NPZ_ROOT", os.path.join(BASE, "MulDyDiff"))
MODELS = ["DAMNET", "AGE", "DYMOND"]
MODES = ["raw", "burst", "hysteresis", "burst_hysteresis"]
MODE_ALT = "|".join(sorted(MODES, key=len, reverse=True))
DS = ("wiki_vote", "twitter", "superuser")
WINDOW = 6
MAX_SEQ = 40

_WIN = re.compile(r"_w(\d+)(?:_b(\d+))?\.npz$")


def window_index(p):
    m = _WIN.search(os.path.basename(p))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (-1, -1)


def npz_paths(dataset, mode):
    paths = sorted(glob.glob(os.path.join(NPZ, "data_processed", dataset,
                                          mode, "*.npz")), key=window_index)
    return [p for p in paths
            if json.load(open(p[:-4] + ".json", encoding="utf-8")).get("accepted")]


def test_meta(dataset, mode, n_test):
    """test 切分每條序列的 (窗口, 批次) 與事件時間點。"""
    paths = npz_paths(dataset, mode)[-n_test:]
    out = []
    for p in paths:
        j = json.load(open(p[:-4] + ".json", encoding="utf-8"))
        out.append((window_index(p), j["macro_ground_truth"]["event_peak"]))
    return out


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def summarize(seqs, meta, pairs, max_seq):
    """回傳 (A_t, a_t, D_t, T_recovery, 未回復比例)。"""
    a, act, dev, rec = [], [], [], []
    for i, ts in enumerate(seqs[:max_seq]):
        seq = edges_from_nx(ts)
        n = max((g.number_of_nodes() for g in ts), default=0)
        if n < 2:
            continue
        a.append(edge_change_activity(seq).mean())
        act.append(active_ratio_from_edges(seq, n).mean())

    for i, ctl_ts in pairs:
        if i >= max_seq:
            continue
        d = deviation_from_control(edges_from_nx(seqs[i]),
                                   edges_from_nx(ctl_ts), window=WINDOW)
        dev.append(np.median(d[-8:]))
        t_peak = meta[i][1] if i < len(meta) else None
        if t_peak is not None:
            delta = baseline_tolerance(d, t_peak)
            rec.append(recovery_time(d, t_peak, delta))

    def med(v):
        return float(np.median(v)) if v else float("nan")

    fin = [x for x in rec if np.isfinite(x)]
    return (med(a), med(act), med(dev),
            med(fin), (len(rec) - len(fin)) / len(rec) if rec else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("root", help="test_and_generated_graphs")
    ap.add_argument("md_root", nargs="?", default="", help="modiff_generated")
    ap.add_argument("gm_root", nargs="?", default="", help="gm_generated")
    ap.add_argument("max_seq", nargs="?", type=int, default=MAX_SEQ)
    ap.add_argument("--models", default="",
                    help="只算這幾個模型（逗號分隔），例如 GraphMaker。"
                         "分批算完再把 CSV 併起來，不必為了補一個模型"
                         "把其他的重算一遍")
    args = ap.parse_args()
    seed, root = args.seed, args.root
    md_root, gm_root, max_seq = args.md_root, args.gm_root, args.max_seq
    want = {x.strip() for x in args.models.split(",") if x.strip()}

    kre = re.compile(r"^macro_(.+?)_(" + MODE_ALT + r")_(\d+)$")
    groups = {}
    for d in sorted(glob.glob(os.path.join(root, f"macro_*_{seed}"))):
        m = kre.match(os.path.basename(d))
        if m:
            groups.setdefault(m.group(1), {})[m.group(2)] = d

    w = csv.writer(sys.stdout)
    w.writerow(["組合", "模式", "來源", "A_t", "a_t", "D_t",
                "T_recovery", "未回復比例"])

    for ds_layer, by_mode in groups.items():
        ds = next((x for x in DS if ds_layer.startswith(x)), None)
        layer = ds_layer[len(ds) + 1:] if ds else ""
        raw_dir = by_mode.get("raw")
        if not raw_dir:
            continue
        raw_src = load(os.path.join(raw_dir, "DAMNET", "test_graphs.pkl"))
        raw_win = {mw: j for j, (mw, _) in
                   enumerate(test_meta(ds, "raw", len(raw_src)))}

        for mode in MODES:
            d = by_mode.get(mode)
            if not d:
                continue
            src = load(os.path.join(d, "DAMNET", "test_graphs.pkl"))
            meta = test_meta(ds, mode, len(src))
            idx = ([(i, raw_win[mw]) for i, (mw, _) in enumerate(meta)
                    if mw in raw_win] if mode != "raw" else [])

            rows = [("原始序列", src,
                     [(i, raw_src[j]) for i, j in idx])]
            for mm in MODELS:
                if want and mm not in want:
                    continue
                p = os.path.join(d, mm, "sampled_ts.pkl")
                if not os.path.exists(p):
                    continue
                gen = load(p)
                cp = os.path.join(raw_dir, mm, "sampled_ts.pkl")
                ctl = load(cp) if os.path.exists(cp) else None
                pr = ([(i, ctl[j]) for i, j in idx
                       if i < len(gen) and j < len(ctl)] if ctl else [])
                rows.append((mm, gen, pr))

            if md_root and (not want or "MoDiff" in want):
                mp = os.path.join(md_root,
                                  f"MoDiff_macro_{ds}_{mode}_{layer}_{seed}",
                                  "MoDiff", "sampled_ts.pkl")
                cp = os.path.join(md_root,
                                  f"MoDiff_macro_{ds}_raw_{layer}_{seed}",
                                  "MoDiff", "sampled_ts.pkl")
                if os.path.exists(mp):
                    gen = load(mp)
                    ctl = load(cp) if os.path.exists(cp) else None
                    pr = ([(i, ctl[j]) for i, j in idx
                           if i < len(gen) and j < len(ctl)] if ctl else [])
                    rows.append(("MoDiff", gen, pr))

            if gm_root and (not want or "GraphMaker" in want):
                gen = gm_loader.find(gm_root, ds, mode, layer, seed)
                ctl = gm_loader.find(gm_root, ds, "raw", layer, seed)
                if gen:
                    pr = ([(i, ctl[j]) for i, j in idx
                           if i < len(gen) and j < len(ctl)] if ctl else [])
                    rows.append(("GraphMaker", gen, pr))

            for name, seqs, pairs in rows:
                v = summarize(seqs, meta, pairs, max_seq)
                w.writerow([ds_layer, mode, name] +
                           ["" if x != x else f"{x:.6f}" for x in v])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
