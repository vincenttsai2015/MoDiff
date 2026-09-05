"""R(t) / ΔR(t) / C(t) / burst intensity / hysteresis intensity。

定義完全照論文 Sec. 4.4 與 MulDyDiff 的實作：

  Eq. 15  R_m^(l)(t) = | (1/N) Σ_i exp(j·θ_{i,m}^(l)(t)) |
          θ 由節點屬性在該 (t, l, 屬性通道) 內正規化到 [0,1] 後乘 π 得到
          （對應 _attribute_to_phase_tensor）

  Eq. 16  ΔR(t) = R(t+1) − R(t)
          burst intensity = mean_t[(ΔR − mean ΔR)²]

  p.8     C(t) = Σ_{τ<=t} ΔR(τ)
          hysteresis intensity H = mean_t[C(t)²]

最後兩項與 refs/MulDyDiff/diffusion/macro_dynamics.py 的
macro_dynamic_descriptor 同定義，test_descriptors.py 會交叉比對。
"""
import numpy as np

EPS = 1e-8


def attribute_to_phase(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """X: (T, L, N, M)，mask: (T, L, N) -> theta: (T, L, N, M)

    對每個 (t, l, 屬性通道) 在節點維度上正規化到 [0,1]，再乘 π。
    整個通道的值都相同時（denom 為 0）theta 取 0。
    """
    m = mask.astype(bool)[..., None]
    x_min = np.where(m, X, np.inf).min(axis=2, keepdims=True)
    x_max = np.where(m, X, -np.inf).max(axis=2, keepdims=True)
    denom = x_max - x_min
    valid = np.isfinite(x_min) & np.isfinite(x_max) & (denom > EPS)
    norm = np.where(valid, (X - x_min) / np.maximum(denom, EPS), 0.0)
    return np.pi * norm


def kuramoto_order(theta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """theta: (T, L, N, M)，mask: (T, L, N) -> R: (T, L, M)"""
    m = mask.astype(np.float64)[..., None]
    n_valid = np.maximum(m.sum(axis=2), 1.0)
    mean_cos = (np.cos(theta) * m).sum(axis=2) / n_valid
    mean_sin = (np.sin(theta) * m).sum(axis=2) / n_valid
    return np.sqrt(mean_cos ** 2 + mean_sin ** 2 + 1e-12)


def compute_R(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return kuramoto_order(attribute_to_phase(X, mask), mask)


def delta_R(R: np.ndarray) -> np.ndarray:
    """R: (T, ...) -> ΔR: (T-1, ...)"""
    return np.diff(R, axis=0)


def burst_intensity(R: np.ndarray) -> np.ndarray:
    d = delta_R(R)
    return ((d - d.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)


def hysteresis_trace(R: np.ndarray) -> np.ndarray:
    """C(t) = Σ_{τ<=t} ΔR(τ)，形狀 (T-1, ...)"""
    return np.cumsum(delta_R(R), axis=0)


def hysteresis_intensity(R: np.ndarray) -> np.ndarray:
    return (hysteresis_trace(R) ** 2).mean(axis=0)


def one_hot(active: np.ndarray) -> np.ndarray:
    """active: (T, L, N) 的 0/1 -> (T, L, N, 2)，inactive=[1,0]、active=[0,1]"""
    X = np.zeros(active.shape + (2,), dtype=np.float64)
    a = active.astype(bool)
    X[..., 0] = (~a).astype(np.float64)
    X[..., 1] = a.astype(np.float64)
    return X
