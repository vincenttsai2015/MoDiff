"""把 data/macro 底下的 nx.Graph 就地轉成 nx.DiGraph。

MoDiff 的 graphs_to_MultiD_tensor_rotate 斷言輸入是 DiGraph——它的 Hermitian
鄰接用虛部編碼邊的方向不對稱。而四個 baseline 拿到的都是無向圖
（DAMNETS 的 sampler 取鄰接下三角，本來就只支援無向），為了比較一致，
這裡轉成雙向的 DiGraph，虛部因此為零。

執行：
    python to_digraph.py                  轉 data/macro 底下全部
    python to_digraph.py --dry-run        只列出要處理的檔案
"""
import argparse
import glob
import os
import pickle

import networkx as nx


def to_di(obj):
    """遞迴把 nx.Graph 換成 nx.DiGraph，回傳 (新物件, 轉換數)。"""
    if isinstance(obj, nx.DiGraph):
        return obj, 0
    if isinstance(obj, nx.Graph):
        g = nx.DiGraph()
        g.add_nodes_from(obj.nodes(data=True))
        g.add_edges_from((u, v, d) for u, v, d in obj.edges(data=True))
        g.add_edges_from((v, u, d) for u, v, d in obj.edges(data=True))
        return g, 1
    if isinstance(obj, list):
        n = 0
        out = []
        for x in obj:
            y, k = to_di(x)
            out.append(y)
            n += k
        return out, n
    return obj, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/macro")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "*", "*.pkl")))
    if not paths:
        raise SystemExit(f"{args.root} 底下找不到 pkl")

    print(f"{len(paths)} 個檔案")
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        new, n = to_di(obj)
        total += n
        print(f"  {os.path.relpath(p, args.root):58s} 轉換 {n:>7d} 張")
        if n and not args.dry_run:
            tmp = p + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)

    print(f"\n合計 {total} 張" + ("（dry-run，沒有寫入）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
