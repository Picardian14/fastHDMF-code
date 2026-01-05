"""
Observable strategies for HDMF outputs.

Design goals
- Keep it simple and pickleable (top-level classes, simple state).
- Config-driven: experiment code builds the pipeline from YAML/Dict and injects it.
- Simulator only calls `pipeline.compute(outputs, params, config)` and saves the result.

Conventions
- outputs is a dict that may contain keys: "bold", "rates", "fic" (numpy arrays).
- Time series shape is assumed (N, T) where rows are regions and columns are time.
  If your simulator uses (T, N), transpose before building observables or adjust here.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterable, Union
import numpy as np


# -----------------------------
# Base types and helpers
# -----------------------------

class BaseObservable:
    """Base class for observable extractors.

    Subclasses implement `compute` and return a dict of results.
    - `name` is a short identifier used in output keys and metadata.
    - `signal` selects the source variable from outputs: "bold", "rates", or "fic".
    """

    name: str = "observable"

    def __init__(self, signal: Union[str, Iterable[str]], **params: Any) -> None:
        # Always normalize to a tuple of strings; callers may pass a single string or a list of strings.
        if isinstance(signal, str):
            self.signal = (signal,)
        else:
            self.signal = tuple(signal)
        self.params = params or {}

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
        
    def spec(self) -> Dict[str, Any]:
        """Serializable spec of this observable (for metadata)."""
        return {"name": self.name, "signal": list(self.signal), "params": self.params}
    def needs(self, signal: str) -> bool:
        """Does this observable need the given signal variable from outputs?"""
        return signal in self.signal
    



def _get_ts(outputs: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    arr = outputs.get(key)
    if arr is None:
        return None
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception:
            return None
    return arr


# -----------------------------
# Concrete observables
# -----------------------------

class FCObservable(BaseObservable):
    """Functional connectivity via Pearson correlation of time series.

    Assumes input shape (N, T). Output is (N, N). If `zero_diag` is True, zeros the diagonal.
    """

    name = "fc"

    def __init__(self, signal: Union[str, Iterable[str]] = "bold", zero_diag: bool = True, **kwargs: Any) -> None:
        super().__init__(signal=signal, zero_diag=zero_diag, **kwargs)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is None or ts.ndim != 2:
                continue
            # Expect (N, T)
            fc = np.corrcoef(ts)
            if self.params.get("zero_diag", True) and fc.ndim == 2 and fc.shape[0] == fc.shape[1]:
                np.fill_diagonal(fc, 0.0)
            res[f"fc_{var}"] = fc
        return res


class MeanObservable(BaseObservable):
    """Mean across time per region. Assumes (N, T) input -> (N,) output."""

    name = "mean"

    def __init__(self, signal: Union[str, Iterable[str]], **kwargs: Any) -> None:
        super().__init__(signal=signal, **kwargs)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is None or ts.ndim < 1:
                continue
            # Assume (N, T) -> mean over time axis=-1; if 1D, mean over axis 0.
            axis = -1 if ts.ndim >= 2 else 0
            res[f"mean_{var}"] = np.mean(ts, axis=axis)
        return res


class RawObservable(BaseObservable):
    """Pass-through: include raw time series for selected variables.

    If `signal` is a list, returns each variable under key `raw_<var>`.
    If `signal` is a string, returns `raw_<signal>`.
    """

    name = "raw"

    def __init__(self, signal: Union[str, Iterable[str]] = ("bold",), **kwargs: Any) -> None:
        super().__init__(signal=signal, **kwargs)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is not None:
                res[f"raw_{var}"] = ts
        return res

from .helper_functions import compute_fcd as _compute_fcd

class FCDObservable(BaseObservable):
    """Functional Connectivity Dynamics (FCD) via sliding-window FC.

    For each selected variable in `signal`, computes windowed FC vectors over time.
    Input time series are assumed (N, T); they are transposed to (T, N) for the helper.
    Output per variable is a matrix of shape (M, W) where M=N*(N-1)/2 and W=#windows.
    """

    name = "fcd"

    def __init__(self, signal: Union[str, Iterable[str]] = "bold", window_size: int = 30, overlap: int = 29, **kwargs: Any) -> None:
        super().__init__(signal=signal, window_size=int(window_size), overlap=int(overlap), **kwargs)
        self.window_size = int(window_size)
        self.overlap = int(overlap)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is None or ts.ndim != 2:
                continue
            # Conventions: ts is (N, T) -> helper expects (T, N)
            N = ts.shape[0]            
            isubdiag = np.triu_indices(N, k=1)
            fcd = _compute_fcd(ts, self.window_size, self.overlap, isubdiag)
            fcd = np.corrcoef(fcd.T)  # Correlate FC vectors over time windows
            res[f"fcd_{var}"] = fcd
        return res


class FICNodeStrengthCorrelationObservable(BaseObservable):
    """Find time index where FIC values have maximal correlation with structural node strength.

    This observable computes the correlation between FIC values and structural connectivity 
    node strength (row/column sums of the connectivity matrix) at each time point, then 
    returns the time index where this correlation is maximal.
    
    The structural connectivity matrix is extracted from params['C'].
    Input FIC time series are assumed (N, T) where N=regions, T=time points.
    """

    name = "fic_max_corr"

    def __init__(self, signal: Union[str, Iterable[str]] = "fic", **kwargs: Any) -> None:
        super().__init__(signal=signal, **kwargs)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        
        if params is None:
            return res
            
        # Extract structural connectivity matrix
        sc_matrix = params.get('C')
        if sc_matrix is None:
            return res
            
        # Compute node strengths (sum of connections for each node)
        node_strengths = np.sum(sc_matrix, axis=1)
        
        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is None or ts.ndim != 2:
                continue
                
            N, T = ts.shape
            
            # Check dimensions match
            if N != len(node_strengths):
                continue
                
            # Compute correlation at each time point
            correlations = np.zeros(T)
            for t in range(T):
                fic_values = ts[:, t]
                # Skip if all FIC values are the same (correlation undefined)
                if np.std(fic_values) == 0 or np.std(node_strengths) == 0:
                    correlations[t] = 0.0
                else:
                    correlations[t] = np.corrcoef(fic_values, node_strengths)[0, 1]
            
            # Find time index with maximum absolute correlation
            max_corr_idx = np.argmax(np.abs(correlations))
            max_correlation = correlations[max_corr_idx]
            
            res[f"fic_max_corr_time_{var}"] = max_corr_idx
            res[f"fic_max_corr_value_{var}"] = max_correlation            
            
        return res


class MaxFreqPowerObservable(BaseObservable):
    """Compute maximum frequency and its power for each region using Welch PSD.

    Returns keys: `max_freqs_<var>`, `max_power_<var>`, and also includes the full
    `freqs_<var>` and `psd_<var>` if `params.get('return_psd', False)`.

    Default parameters (can be overridden via params):
    - fs: sampling frequency (default 1000)
    - nperseg: number of samples per segment (default 4000)
    - noverlap: overlap between segments (default 2000)
    - max_freq_cut: upper index (or frequency) limit when searching for max (default: first 100 bins)
    """

    name = "max_freq_power"

    def __init__(self, signal: Union[str, Iterable[str]] = "rates", **kwargs: Any) -> None:
        super().__init__(signal=signal, **kwargs)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        p = dict(fs=1000, nperseg=4 * 1000, noverlap=2 * 1000, max_freq_cut=100, return_psd=False)        
        if self.params is not None:
            p.update(self.params)

        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is None or ts.ndim != 2:
                continue
            # Expect (N, T): compute Welch along time axis=1 for each region (axis=1 -> axis=1 in welch when axis=1)
            # Import scipy.signal.welch locally so this module can be imported even when scipy isn't installed.
            try:
                from scipy.signal import welch
            except Exception:
                # If scipy not available, skip this observable silently.
                continue
            freqs, psd = welch(ts, fs=p["fs"], axis=1, nperseg=p["nperseg"], noverlap=p["noverlap"]) 
            # psd shape (N, F)
            # Determine index/limit to search for max: if max_freq_cut is int -> interpret as number of bins
            max_cut = p["max_freq_cut"]
            if isinstance(max_cut, (int,)):
                idx_cut = min(max_cut, psd.shape[1])
            else:
                # if provided frequency value, find nearest index
                idx_cut = np.searchsorted(freqs, float(max_cut))
                idx_cut = min(max(1, idx_cut), psd.shape[1])

            max_freq_id = np.argmax(psd[:, :idx_cut], axis=1)
            max_freqs = freqs[max_freq_id]
            max_power = np.max(psd[:, :idx_cut], axis=1)

            res[f"max_freqs_{var}"] = max_freqs
            res[f"max_power_{var}"] = max_power
            if p.get("return_psd"):
                res[f"freqs_{var}"] = freqs
                res[f"psd_{var}"] = psd

        return res


class GammaEntropyObservable(BaseObservable):
    """Fit a gamma distribution to each node's time series and return the entropy.

    Assumes input time series shape (N, T) where rows are regions and columns are time.
    Returns an array of length N under the key `gamma_entropy_<var>`.

    Optional params:
    - return_fit_params: bool (default False) -> also return fitted (alpha, loc, scale)
      arrays under keys `gamma_alpha_<var>`, `gamma_loc_<var>`, `gamma_scale_<var>`.
    """

    name = "gamma_entropy"

    def __init__(self, signal: Union[str, Iterable[str]] = "rates", **kwargs: Any) -> None:
        super().__init__(signal=signal, **kwargs)

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        p = dict(return_fit_params=False)
        if self.params:
            p.update(self.params)

        for var in self.signal:
            ts = _get_ts(outputs, var)
            if ts is None or ts.ndim != 2:
                continue

            N, T = ts.shape

            # Prepare output arrays
            entropies = np.empty(N, dtype=float)
            entropies.fill(np.nan)
            alphas = np.empty(N, dtype=float)
            locs = np.empty(N, dtype=float)
            scales = np.empty(N, dtype=float)
            alphas.fill(np.nan); locs.fill(np.nan); scales.fill(np.nan)

            # Import locally to avoid hard dependency at module import time
            try:
                from scipy.stats import gamma as _gamma
            except Exception:
                # SciPy not available: return nothing for this variable
                continue

            for node in range(N):
                rates = ts[node, :]
                # If constant signal, gamma fit is degenerate; set entropy to 0.
                if np.nanstd(rates) == 0:
                    entropies[node] = 0.0
                    alphas[node] = np.nan
                    locs[node] = np.nan
                    scales[node] = np.nan
                    continue

                try:
                    a, loc, scale = _gamma.fit(rates)
                    H = _gamma.entropy(a=a, loc=loc, scale=scale)
                except Exception:
                    # On any fitting failure, set NaNs and continue
                    a = np.nan; loc = np.nan; scale = np.nan; H = np.nan

                alphas[node] = a
                locs[node] = loc
                scales[node] = scale
                entropies[node] = H

            res[f"gamma_entropy_{var}"] = entropies
            if p.get("return_fit_params"):
                res[f"gamma_alpha_{var}"] = alphas
                res[f"gamma_loc_{var}"] = locs
                res[f"gamma_scale_{var}"] = scales

        return res


# -----------------------------
# Pipeline and factory
# -----------------------------

_REGISTRY = {
    FCObservable.name: FCObservable,
    FCDObservable.name: FCDObservable,
    MeanObservable.name: MeanObservable,
    RawObservable.name: RawObservable,
    FICNodeStrengthCorrelationObservable.name: FICNodeStrengthCorrelationObservable,
    MaxFreqPowerObservable.name: MaxFreqPowerObservable,
    GammaEntropyObservable.name: GammaEntropyObservable,
}


class ObservablesPipeline:
    """Holds a list of observable extractors and applies them in order."""

    def __init__(self, extractors: Optional[List[BaseObservable]] = None) -> None:
        self.extractors: List[BaseObservable] = list(extractors or [])

    def compute(self, outputs: Dict[str, np.ndarray], params: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for i, obs in enumerate(self.extractors):
            out = obs.compute(outputs, params, config) or {}
            # Merge with simple collision avoidance
            for k, v in out.items():
                if k in merged:
                    merged[f"{k}_{i}"] = v
                else:
                    merged[k] = v
        return merged

    def spec(self) -> List[Dict[str, Any]]:
        return [obs.spec() for obs in self.extractors]
    def needs(self, signal: str) -> bool:
        return any(obs.needs(signal) for obs in self.extractors)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "ObservablesPipeline":
        """Build pipeline from config dict.

        Expected shape (example):
        {
          "observables": [
            {"name": "fc", "signal": "bold", "params": {"zero_diag": true}},
            {"name": "mean", "signal": "rates"},
            {"name": "raw", "signal": ["bold", "rates"]}
          ]
        }

        If not provided, defaults to FC signal BOLD.
        """
        items = []
        obs_list = (config or {}).get("observables")
        if not obs_list:
            # Default: FC from BOLD
            items.append(FCObservable(signal="bold"))
            return cls(items)

        for spec in obs_list:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            signal = spec.get("signal")
            params = spec.get("params", {}) or {}
            ctor = _REGISTRY.get(name)
            if ctor is None:
                continue
            if signal is None:
                # Reasonable defaults if `signal` omitted
                default_on: Union[str, Iterable[str]] = "bold" if name in ("fc", "raw") else "rates"
                items.append(ctor(signal=default_on, **params))
            else:
                items.append(ctor(signal=signal, **params))
        return cls(items)

    @classmethod
    def default(cls) -> "ObservablesPipeline":
        """FC signal BOLD only."""
        return cls([FCObservable(signal="bold")])


__all__ = [
    "BaseObservable",
    "FCObservable",
    "FCDObservable",
    "MeanObservable",
    "RawObservable",
    "FICNodeStrengthCorrelationObservable",
    "MaxFreqPowerObservable",
    "GammaEntropyObservable",
    "ObservablesPipeline",
]
