"""巨觀動態的拓撲指標。

四個量，都以「單層、單條序列」為單位計算：

    A_t   = |E_t △ E_{t-1}|                 edge change activity
    a_t   = #{active nodes at t} / N        active node ratio
    D_J(t)= 1 - |E_t ∩ E_pre| / |E_t ∪ E_pre|   對參考圖的 Jaccard 距離
    T_rec = min{t > t_p : D_J(t) <= δ}      回復時間，δ 取事件前波動的 95 百分位

`D_J` 的兩個彙總量：`H_residual = D_J(T-1)`（序列結束時仍與參考圖的距離）、
`H_persist = mean_{t > t_p} D_J(t)`（事件後的平均距離）。

核心函式只吃「每個 timestamp 的邊集合」，所以同一套程式可以用在原始序列與
各個 baseline 生成的圖上。節點屬性是選用的——只讀鄰接矩陣的模型沒有屬性，
那種情況用 `active_ratio_from_edges` 以「該 snapshot 有邊」定義 active。
"""
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Edge = Tuple[int, int]
EdgeSeq = Sequence[Set[Edge]]


def edge_change_activity(seq: EdgeSeq) -> np.ndarray:
    """A_t，長度 T-1，對應 t = 1..T-1。"""
    return np.array([len(seq[t] ^ seq[t - 1]) for t in range(1, len(seq))],
                    dtype=float)


def active_ratio_from_edges(seq: EdgeSeq, n_nodes: int) -> np.ndarray:
    """a_t，以「該 snapshot 至少有一條邊」定義 active。"""
    out = []
    for es in seq:
        nodes = set()
        for u, v in es:
            nodes.add(u)
            nodes.add(v)
        out.append(len(nodes) / n_nodes)
    return np.array(out, dtype=float)


def active_ratio_from_attr(active: np.ndarray) -> np.ndarray:
    """a_t，直接用節點屬性。active: (T, N) 的 0/1。"""
    return active.mean(axis=1).astype(float)


def jaccard_distance(a: Set[Edge], b: Set[Edge]) -> float:
    """兩張圖都沒有邊時距離定義為 0。"""
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def sliding_union(seq: EdgeSeq, window: int) -> List[Set[Edge]]:
    """把每個 t 換成最近 window 張的邊聯集。window=1 時原樣回傳。

    事件流切出來的 snapshot 很稀疏，相鄰時間點的邊幾乎不重疊，
    逐張比對的 Jaccard 距離會恆等於 1。累積成窗之後圖才有足夠的共同結構
    可以量「偏離參考狀態多遠」。
    """
    if window <= 1:
        return [set(s) for s in seq]
    out = []
    for t in range(len(seq)):
        acc: Set[Edge] = set()
        for s in seq[max(0, t - window + 1):t + 1]:
            acc |= s
        out.append(acc)
    return out


def residual_shift(seq: EdgeSeq, ref: int = 0) -> np.ndarray:
    """D_J(t)，長度 T。ref 是參考圖 G_pre 的 index。

    事件流切出來的 snapshot 之間邊幾乎不重複，這個定義在稀疏資料上會飽和到 1，
    分不出注入的動態。要量結構偏離請用 `deviation_from_control`。
    """
    g_pre = seq[ref]
    return np.array([jaccard_distance(es, g_pre) for es in seq], dtype=float)


def deviation_from_control(seq: EdgeSeq, control: EdgeSeq,
                           window: int = 1) -> np.ndarray:
    """D(t) = Jaccard(E_t, E_t^control)，對照組是同一窗口未注入動態的版本。

    參考基準從「事件前的自己」換成「沒有事件的自己」，量的就是注入造成的
    結構偏離：邊在事件後消失的話這條曲線回到 0，留下來的話維持在高檔。
    """
    a = sliding_union(seq, window)
    b = sliding_union(control, window)
    return np.array([jaccard_distance(a[t], b[t]) for t in range(len(a))],
                    dtype=float)


def recovery_time(d: np.ndarray, t_peak: int, delta: float) -> float:
    """事件後第一次回到 δ 以內的時間點；到序列結束都沒有則回傳 inf。"""
    for t in range(t_peak + 1, len(d)):
        if d[t] <= delta:
            return float(t)
    return float("inf")


def baseline_tolerance(d: np.ndarray, t_peak: int, ref: int = 0,
                       q: float = 95.0) -> float:
    """δ：事件之前 D_J 波動的 q 百分位。

    參考圖自己的 D_J 恆為 0，不納入。事件前沒有其他時間點時退回 0。
    """
    pre = [d[t] for t in range(len(d)) if t < t_peak and t != ref]
    return float(np.percentile(pre, q)) if pre else 0.0


def sequence_metrics(seq: EdgeSeq, t_peak: int, n_nodes: int,
                     active: Optional[np.ndarray] = None,
                     ref: Optional[int] = None,
                     window: int = 1,
                     control: Optional[EdgeSeq] = None) -> Dict[str, float]:
    """一條序列、一層的所有指標。

    `A_t` 與 `a_t` 用逐張的邊集合。結構偏離 `D` 在給了 `control` 時以對照組為
    基準，否則退回以序列自身的 `ref`（預設事件前一刻）為參考圖。

    回傳的 `A_t` / `a_t` / `D` 是逐時間點的序列，其餘是彙總值。
    """
    if ref is None:
        ref = max(0, t_peak - 1)
    a_t_edges = active_ratio_from_edges(seq, n_nodes)
    if control is not None:
        d = deviation_from_control(seq, control, window=window)
    else:
        d = residual_shift(sliding_union(seq, window), ref=ref)
    delta = baseline_tolerance(d, t_peak, ref=ref)
    post = d[t_peak + 1:]

    out = {
        "A_t": edge_change_activity(seq),
        "a_t_edges": a_t_edges,
        "D": d,
        "delta": delta,
        "H_residual": float(d[-1]),
        "H_persist": float(post.mean()) if len(post) else float("nan"),
        "T_recovery": recovery_time(d, t_peak, delta),
        "A_t_mean": float(edge_change_activity(seq).mean()),
        "A_t_peak": float(edge_change_activity(seq).max()),
        "a_t_mean": float(a_t_edges.mean()),
    }
    if active is not None:
        a_attr = active_ratio_from_attr(active)
        out["a_t_attr"] = a_attr
        out["a_t_attr_mean"] = float(a_attr.mean())
        out["a_t_attr_peak"] = float(a_attr.max())
    return out


# ── 從我們的序列檔讀取 ────────────────────────────────────────────────

def edges_from_npz(edges: np.ndarray, layer: int, T: int) -> List[Set[Edge]]:
    """edges: (M, 4) = (t, layer, u_local, v_local)，無向、去重。"""
    seq: List[Set[Edge]] = [set() for _ in range(T)]
    for t, l, u, v in edges:
        if int(l) == layer:
            u, v = int(u), int(v)
            seq[int(t)].add((u, v) if u < v else (v, u))
    return seq
