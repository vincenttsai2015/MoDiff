"""跑幾步訓練，抓出 NaN 從哪一步、哪個張量開始。

    python diag_train.py macro_wiki_vote_burst_support [步數]

在運算節點上跑。不存 checkpoint。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers.config import get_config  # noqa: E402
from utils.loader import (load_model_optimizer, load_loss_fn4DT,  # noqa: E402
                          load_batch2, load_model_params, load_data_TD_train_comp)


def bad(t):
    if not torch.is_tensor(t):
        return False
    b = t.real if t.is_complex() else t
    return bool(torch.isnan(b).any() or torch.isinf(b).any())


def scan_params(model, tag):
    n_p = sum(1 for p in model.parameters() if bad(p))
    n_g = sum(1 for p in model.parameters()
              if p.grad is not None and bad(p.grad))
    gn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.real if p.grad.is_complex() else p.grad
            if torch.isfinite(g).all():
                gn += float(g.norm()) ** 2
    print(f"    {tag:10s} 壞掉的參數 {n_p}，壞掉的梯度 {n_g}，"
          f"有限梯度範數 {gn ** 0.5:.4g}")
    return n_p, n_g


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "macro_wiki_vote_burst_support"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    scale = name[len("macro_"):] if name.startswith("macro_") else name

    cfg = get_config(f"Macro/Macro_{scale}", 0)
    cfg.scale = scale
    cfg.type = os.environ.get("BETA_TYPE", "linear")
    print(f"=== {name} ===")
    print(f"lr        x={cfg.train.lr}  grad_norm={cfg.train.grad_norm}")
    print(f"batch     {cfg.data.batch_size}")
    print(f"spec_dim  {cfg.data.spec_dim}   max_node_num {cfg.data.max_node_num}")
    print()

    device = [0]                      # 與 trainer.py 一致
    tr0, tr1, _, _ = load_data_TD_train_comp(cfg)
    params_x, params_adj = load_model_params(cfg)
    model_x, opt_x, _ = load_model_optimizer(params_x, cfg.train, device)
    model_adj, opt_adj, _ = load_model_optimizer(params_adj, cfg.train, device)
    loss_fn = load_loss_fn4DT(cfg)
    model_x.train()
    model_adj.train()

    print(f"{'步':>4s} {'loss_x':>12s} {'loss_adj':>12s}")
    i = 0
    for b0, b1 in zip(tr0, tr1):
        if i >= steps:
            break
        i += 1
        opt_x.zero_grad()
        opt_adj.zero_grad()
        x0, adj0, u0, la0 = load_batch2(b0, device)
        la1 = b1[3].to(f"cuda:{device[0]}")

        lx, ladj = loss_fn(model_x, model_adj, x0, adj0, u0, la0, la1)
        print(f"{i:>4d} {lx.item():>12.4e} {ladj.item():>12.4e}")

        if bad(lx) or bad(ladj):
            print()
            print(f"  第 {i} 步的 loss 就壞了。檢查輸入：")
            for nm, t in (("x0", x0), ("adj0", adj0), ("u0", u0),
                          ("la0", la0), ("la1", la1)):
                print(f"    {nm:5s} 壞 {bad(t)}  "
                      f"|值|最大 {t.abs().max().item():.4g}")
            scan_params(model_x, "model_x")
            scan_params(model_adj, "model_adj")
            break

        lx.backward()
        ladj.backward()
        p1, g1 = scan_params(model_x, "model_x")
        p2, g2 = scan_params(model_adj, "model_adj")
        if p1 or g1 or p2 or g2:
            print(f"  第 {i} 步 backward 之後參數或梯度壞掉")
            break

        torch.nn.utils.clip_grad_norm_(model_x.parameters(), cfg.train.grad_norm)
        torch.nn.utils.clip_grad_norm_(model_adj.parameters(), cfg.train.grad_norm)
        opt_x.step()
        opt_adj.step()

    print()
    print("loss 從第幾步變 NaN、以及是輸入壞還是梯度爆，決定要調什麼。")


if __name__ == "__main__":
    main()
