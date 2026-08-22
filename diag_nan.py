"""逐階段找出 NaN 從哪裡進來。

    python diag_nan.py macro_wiki_vote_burst_support

在運算節點上跑（要 GPU）。只讀資料、不訓練。

資料的組法與 dataloader_TD_train_comp 一致：V / R / T 三個檔會被當成
transform_type [1, 2, 3] 的三個角度，逐一取同一個索引組成 Hermitian 矩陣，
不是 train / val / test 的用法。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import top_k_eigen  # noqa: E402
from utils.graph_utils import (graphs_to_MultiD_tensor_rotate, init_features,  # noqa
                               node_flags)
from data.data_generators import load_dataset  # noqa: E402
from parsers.config import get_config  # noqa: E402
from sde import VPSDE  # noqa: E402

N_SAMPLE = 64


def rep(name, t):
    if not torch.is_tensor(t):
        print(f"  {name:24s} 不是 tensor: {type(t)}")
        return
    x = t.detach()
    base = x.real if x.is_complex() else x
    n_nan = torch.isnan(base).sum().item()
    n_inf = torch.isinf(base).sum().item()
    finite = x[torch.isfinite(base)]
    rng = (f"{finite.abs().min().item():.3g} ~ {finite.abs().max().item():.3g}"
           if finite.numel() else "無有限值")
    flag = "   <== NaN" if n_nan else ""
    print(f"  {name:24s} {str(tuple(x.shape)):20s} NaN {n_nan:>8d} "
          f"Inf {n_inf:>6d}  |值| {rng}{flag}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "macro_wiki_vote_burst_support"
    scale = name[len("macro_"):] if name.startswith("macro_") else name
    cfg = get_config(f"Macro/Macro_{scale}", 0)

    print(f"=== {name} ===")
    print(f"data.dir       {cfg.data.dir}")
    print(f"max_node_num   {cfg.data.max_node_num}")
    print(f"spec_dim       {cfg.data.spec_dim}")
    print(f"test_split     {cfg.data.test_split}")
    print(f"sde.adj beta   {cfg.sde.adj.beta_min} ~ {cfg.sde.adj.beta_max}")
    print()

    # 與 dataloader_TD_train_comp 相同：V / R / T 是三個角度
    lists = []
    for tag in ("V", "R", "T"):
        gl = load_dataset(data_dir=cfg.data.dir, file_name=cfg.data.file1 + tag)
        n_test = int(cfg.data.test_split * len(gl))
        train = gl[n_test:]
        e = [g.number_of_edges() for g in train]
        print(f"  {tag}: 全部 {len(gl)}、訓練 {len(train)}、"
              f"邊數中位 {sorted(e)[len(e)//2] if e else 0}、"
              f"空圖 {sum(1 for x in e if x == 0)}")
        lists.append(train[:N_SAMPLE])

    n = min(len(x) for x in lists)
    lists = [x[:n] for x in lists]
    print(f"  取前 {n} 張做檢查（型別 {type(lists[0][0]).__name__}）")
    print()

    print("--- 1. 鄰接張量 ---")
    adjs = graphs_to_MultiD_tensor_rotate(lists, cfg.data.max_node_num)
    rep("adjs_tensor", adjs)

    print("--- 2. 節點旗標 ---")
    flags = node_flags(adjs)
    rep("flags", flags)
    cnt = flags.sum(-1)
    print(f"  有效節點數 最小 {cnt.min().item():.0f}、"
          f"中位 {cnt.median().item():.0f}、最大 {cnt.max().item():.0f}")
    print(f"  有效節點數為 0 的張數 {(cnt == 0).sum().item()}/{len(cnt)}")

    print("--- 3. 特徵分解 ---")
    la, u = top_k_eigen(adjs, cfg.data.spec_dim)
    rep("la（特徵值）", la)
    rep("u（特徵向量）", u)

    print("--- 4. SDE ---")
    sde = VPSDE(cfg.sde.adj.beta_min, cfg.sde.adj.beta_max,
                cfg.sde.adj.num_scales)
    beta_type = os.environ.get("BETA_TYPE", "linear")   # parser.py 的預設
    print(f"  beta_type = {beta_type}")
    sde.select_type(beta_type)
    t = torch.rand(la.shape[0], device=la.device) * (sde.T - 1e-5) + 1e-5
    mean, std = sde.marginal_prob_adj(la, t, u, la)
    rep("mean_eigen", mean)
    rep("std_eigen", std)

    print("--- 5. 節點特徵 ---")
    x = init_features(cfg.data.init, adjs, cfg.data.max_feat_num)
    rep("x_tensor", x)

    print("--- 6. 擾動後的鄰接 ---")
    eigen_mask = torch.zeros((adjs.shape[0], adjs.shape[1]), device=adjs.device)
    for i in range(flags.shape[0]):
        c = int(torch.count_nonzero(flags[i]))
        eigen_mask[i, :c // 2] = 1
        eigen_mask[i, -c // 2:] = 1
    rep("eigen_mask", eigen_mask)
    z = torch.randn_like(la)
    pe = (mean * eigen_mask[:, :la.shape[1]] + std[:, None] * z)
    rep("perturbed_eigen", pe)
    pd = torch.diag_embed(pe).type(torch.complex64)
    padj = torch.matmul(torch.matmul(u, pd), torch.conj(u.transpose(-1, -2)))
    rep("perturbed_adj", padj)

    print()
    print("第一個標 <== NaN 的階段就是問題所在。")


if __name__ == "__main__":
    main()
