"""k-NN probe used by the f_err objective, with a CPU and a GPU backend that
agree bit-for-bit.

Why a dedicated module
----------------------
``f_err`` is ~99 % of the cost of one objective evaluation: it materialises a
|val| x |train| = 1000 x 2000 squared-distance matrix (16 MB) and then selects
the k nearest training points for every validation point.  On the CPU that is a
memory-bandwidth-bound loop, on the GPU it is 60 ms -> 1.5 ms.

Why the tie-break is explicit
-----------------------------
URL features are coarse: with a mask of a few count-like features, many distinct
training URLs share the *same* vector in the selected coordinates, so several
distances are *exactly* equal.  ``np.argpartition`` and ``torch.topk`` then
disagree about which of the tied neighbours to keep (measured: 62 of 300 random
masks, one of them by 0.32 in balanced accuracy).  The probe therefore orders by
``(distance, then training-row index)``, which is well defined and identical on
both backends, so a GPU run and a CPU run produce the same numbers.

The tie-break is implemented by adding ``row_index * 1e-12`` to the distances.
That is ~10^1 above float64 noise (measured max |CPU-GPU| on the raw matrix:
2e-13) and orders of magnitude below any genuine distance gap, so it only
decides true ties.
"""
from __future__ import annotations

import os

import numpy as np

TIE_EPS = 1e-12


def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same value as ``sklearn.metrics.balanced_accuracy_score`` (two classes),
    but ~50x faster because it skips building a confusion matrix.  The sklearn
    call was 1.5 ms of a 5.7 ms GPU probe, i.e. 25 % of the whole objective."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for cls in (0, 1):
        m = y_true == cls
        n = int(m.sum())
        if n == 0:
            continue
        recalls.append(float((y_pred[m] == cls).sum()) / n)
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def _backend_from_env() -> str:
    return os.environ.get("AMOFS_PROBE", "auto").strip().lower()


class KnnProbe:
    """Exact k-NN probe over a fixed train/validation subsample."""

    def __init__(self, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                 k: int = 5, backend: str | None = None):
        self.Xtr = np.ascontiguousarray(Xtr, dtype=np.float64)
        self.Xva = np.ascontiguousarray(Xva, dtype=np.float64)
        self.ytr = np.asarray(ytr).astype(np.int8)
        self.yva = np.asarray(yva).astype(np.int8)
        self.k = int(k)
        self.n_tr = self.Xtr.shape[0]
        self._bias = (np.arange(self.n_tr, dtype=np.float64) * TIE_EPS)[None, :]

        want = (backend or _backend_from_env())
        self.backend = "cpu"
        self._torch = None
        if want in ("gpu", "cuda", "auto"):
            try:
                self._init_gpu()
                self.backend = "gpu"
            except Exception as e:  # pragma: no cover - depends on the host
                if want != "auto":
                    raise
                print(f"[probe] GPU backend unavailable ({type(e).__name__}: {e}); using CPU")

    # ------------------------------------------------------------------ gpu
    def _init_gpu(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        self._torch = torch
        dev = torch.device("cuda")
        self._dev = dev
        self._Xtr_t = torch.from_numpy(self.Xtr).to(dev)
        self._Xva_t = torch.from_numpy(self.Xva).to(dev)
        self._ytr_t = torch.from_numpy(self.ytr.astype(np.int64)).to(dev)
        self._bias_t = torch.from_numpy(self._bias).to(dev)

    # ------------------------------------------------------------------ api
    def f_err(self, mask: np.ndarray) -> float:
        sel = np.flatnonzero(mask)
        if sel.size == 0:
            return 0.5
        pred = self._predict_gpu(sel) if self.backend == "gpu" else self._predict_cpu(sel)
        return 1.0 - balanced_accuracy_np(self.yva, pred)

    # ------------------------------------------------------------------ cpu
    def _predict_cpu(self, sel: np.ndarray) -> np.ndarray:
        A, B = self.Xva[:, sel], self.Xtr[:, sel]
        d2 = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
              - 2.0 * (A @ B.T) + self._bias)
        kk = min(self.k, B.shape[0])
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        return (self.ytr[nn].mean(1) >= 0.5).astype(int)

    # ------------------------------------------------------------------ gpu
    def _predict_gpu(self, sel: np.ndarray) -> np.ndarray:
        torch = self._torch
        idx = torch.as_tensor(sel, dtype=torch.long, device=self._dev)
        A = self._Xva_t.index_select(1, idx)
        B = self._Xtr_t.index_select(1, idx)
        d2 = torch.addmm(
            (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :],
            A, B.t(), beta=1.0, alpha=-2.0,
        )
        d2 = d2 + self._bias_t
        nn = torch.topk(d2, min(self.k, B.shape[0]), dim=1, largest=False, sorted=False).indices
        votes = self._ytr_t[nn].to(torch.float64).mean(1)
        return (votes >= 0.5).to(torch.int8).cpu().numpy()
