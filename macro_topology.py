"""巨觀動態的拓撲指標。

四個量，都以「單層、單條序列」為單位計算：

    A_t   = |E_t △ E_{t-1}|                  edge change activity
    a_t   = #{active nodes at t} / N         active node ratio
    D(t)  = 1 - |E_t ∩ E_ref| / |E_t ∪ E_ref|    對參考圖的 Jaccard 距離
    T_rec = min{t > t_p : D(t) <= δ}         回復時間

δ 取事件之前那段 D(t) 的 95 百分位。`D(t)` 的兩個彙總量是
`H_residual = D(T-1)` 與 `H_persist = mean_{t > t_p} D(t)`。

`D(t)` 的參考圖預設取同一窗口未注入動態的對照序列，量的是注入造成的結構偏離；
沒有對照序列時退回序列自身的第 `ref` 張。

核心函式只吃每個 timestamp 的邊集合。節點屬性是選用的，
沒有屬性時以「該 snapshot 有邊」定義 active。
"""
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Edge = Tuple[int, int]
EdgeSeq = Sequence[Set[Edge]]


# ── 核心 ──────────────────────────────────────────────────────────────

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
        out.append(len(nodes) / n_nodes if n_nodes else 0.0)
    return np.array(out, dtype=float)


def active_ratio_from_attr(active: np.ndarray) -> np.ndarray:
    """a_t，直接用節點屬性。active: (T, N) 的 0/1。"""
    return np.asarray(active, dtype=float).mean(axis=1)


def jaccard_distance(a: Set[Edge], b: Set[Edge]) -> float:
    """兩張圖都沒有邊時距離定義為 0。"""
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def sliding_union(seq: EdgeSeq, window: int) -> List[Set[Edge]]:
    """把每個 t 換成最近 window 張的邊聯集。window=1 時原樣回傳。"""
    if window <= 1:
        return [set(s) for s in seq]
    out = []
    for t in range(len(seq)):
        acc: Set[Edge] = set()
        for s in seq[max(0, t - window + 1):t + 1]:
            acc |= s
        out.append(acc)
    return out


def deviation_from_control(seq: EdgeSeq, control: EdgeSeq,
                           window: int = 1) -> np.ndarray:
    """D(t) = Jaccard(E_t, E_t^control)。兩邊必須是同一批節點、同一個窗口。"""
    a = sliding_union(seq, window)
    b = sliding_union(control, window)
    n = min(len(a), len(b))
    return np.array([jaccard_distance(a[t], b[t]) for t in range(n)], dtype=float)


def residual_shift(seq: EdgeSeq, ref: int = 0, window: int = 1) -> np.ndarray:
    """D(t)，以序列自身第 ref 張為參考圖。

    `ref` 超出序列長度時夾回最後一個合法索引，避免上游的事件時間點
    （例如跨資料集切分算出來的 t_peak）對不齊這條序列實際長度時整批崩潰。
    """
    s = sliding_union(seq, window)
    ref = max(0, min(ref, len(s) - 1))
    return np.array([jaccard_distance(es, s[ref]) for es in s], dtype=float)


def baseline_tolerance(d: np.ndarray, t_peak: int, ref: int = 0,
                       q: float = 95.0) -> float:
    """容許值 δ：事件之前那段 D(t) 的 q 百分位。參考點自身恆為 0，不納入。"""
    pre = [d[t] for t in range(len(d)) if t < t_peak and t != ref]
    return float(np.percentile(pre, q)) if pre else 0.0


def recovery_time(d: np.ndarray, t_peak: int, delta: float) -> float:
    """事件後第一次回到 δ 以內的時間點；到序列結束都沒有則回傳 inf。"""
    for t in range(t_peak + 1, len(d)):
        if d[t] <= delta:
            return float(t)
    return float("inf")


def sequence_metrics(seq: EdgeSeq, n_nodes: int, t_peak: Optional[int] = None,
                     active: Optional[np.ndarray] = None,
                     control: Optional[EdgeSeq] = None,
                     ref: Optional[int] = None,
                     window: int = 1) -> Dict[str, object]:
    """一條序列、一層的所有指標。

    `t_peak` 為 None 時只計算 `A_t` 與 `a_t`。
    `control` 為 None 時 `D(t)` 以序列自身的第 `ref` 張為參考圖。
    """
    a_edges = active_ratio_from_edges(seq, n_nodes)
    at = edge_change_activity(seq)
    out: Dict[str, object] = {
        "A_t": at,
        "A_t_mean": float(at.mean()) if len(at) else float("nan"),
        "A_t_peak": float(at.max()) if len(at) else float("nan"),
        "a_t_edges": a_edges,
        "a_t_edges_mean": float(a_edges.mean()),
    }
    if active is not None:
        a_attr = active_ratio_from_attr(active)
        out["a_t_attr"] = a_attr
        out["a_t_attr_mean"] = float(a_attr.mean())
        out["a_t_attr_peak"] = float(a_attr.max())

    if t_peak is None:
        return out

    if ref is None:
        ref = max(0, t_peak - 1)
    d = (deviation_from_control(seq, control, window=window)
         if control is not None else residual_shift(seq, ref=ref, window=window))
    delta = baseline_tolerance(d, t_peak, ref=ref)
    post = d[t_peak + 1:]
    out.update({
        "D": d,
        "delta": delta,
        "H_residual": float(d[-1]) if len(d) else float("nan"),
        "H_persist": float(post.mean()) if len(post) else float("nan"),
        "T_recovery": recovery_time(d, t_peak, delta),
    })
    return out


# ── 從不同來源取得邊集合 ──────────────────────────────────────────────

def edges_from_nx(graphs) -> List[Set[Edge]]:
    """一層的 T 張 networkx 圖，取出每張的無向邊集合。"""
    out = []
    for g in graphs:
        out.append({(min(int(u), int(v)), max(int(u), int(v)))
                    for u, v in g.edges()})
    return out


def active_from_nx(graphs, attr: str = "x") -> Optional[np.ndarray]:
    """從節點屬性取 active。`x` 是 one-hot，第二個通道為 1 表示 active。

    圖上沒有這個屬性時回傳 None。
    """
    rows = []
    for g in graphs:
        vals = []
        for _, data in g.nodes(data=True):
            v = data.get(attr)
            if v is None:
                return None
            v = np.asarray(v).ravel()
            vals.append(float(v[1] if v.size > 1 else v[0]))
        rows.append(vals)
    if not rows or not rows[0]:
        return None
    width = max(len(r) for r in rows)
    return np.array([r + [0.0] * (width - len(r)) for r in rows], dtype=float)


def edges_from_npz(edges: np.ndarray, layer: int, T: int) -> List[Set[Edge]]:
    """序列檔的 edges 欄位是 (t, layer, u_local, v_local)，形狀 (M, 4)。"""
    seq: List[Set[Edge]] = [set() for _ in range(T)]
    for t, l, u, v in edges:
        if int(l) == layer:
            u, v = int(u), int(v)
            seq[int(t)].add((u, v) if u < v else (v, u))
    return seq


# ── 對整批序列 ────────────────────────────────────────────────────────

def _as_layered(sequences):
    """統一成 `[序列][層][時間]`。`[序列][時間]` 視為單一層。"""
    out = []
    for seq in sequences:
        if seq and isinstance(seq[0], (list, tuple)):
            out.append(list(seq))
        else:
            out.append([list(seq)])
    return out


def summarize_sequences(graph_sequences, t_peaks=None, control_sequences=None,
                        window: int = 1, prefix: str = "") -> Dict[str, float]:
    """對整批序列算出各指標的中位數。

    `graph_sequences` 可以是 `[序列][層][時間]` 或 `[序列][時間]`。
    `t_peaks` 是每條序列的事件時間點，`control_sequences` 是同形狀的對照組。
    回傳的鍵加上 `prefix`，`T_recovery` 另附未回復的比例。
    """
    seqs = _as_layered(graph_sequences)
    ctls = _as_layered(control_sequences) if control_sequences is not None else None

    acc: Dict[str, List[float]] = {}
    for s, layers in enumerate(seqs):
        for k, graphs in enumerate(layers):
            if not graphs:
                continue
            seq = edges_from_nx(graphs)
            ctl = None
            if ctls is not None and s < len(ctls) and k < len(ctls[s]):
                ctl = edges_from_nx(ctls[s][k])
            m = sequence_metrics(
                seq, n_nodes=graphs[0].number_of_nodes(),
                t_peak=None if t_peaks is None else t_peaks[s],
                active=active_from_nx(graphs), control=ctl, window=window)
            for key, val in m.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    acc.setdefault(key, []).append(float(val))

    out = {}
    for key, vals in acc.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[f"{prefix}{key}"] = float(np.median(finite)) if finite.size else float("nan")
        if key == "T_recovery":
            out[f"{prefix}T_recovery_not_recovered"] = float(
                np.mean(~np.isfinite(arr)))
    return out
