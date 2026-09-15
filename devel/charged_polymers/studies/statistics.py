"""Shared autocorrelation and block-error estimators; original formulas unchanged."""
import numpy as np


def correlation_time(values):
    """Initial positive paired autocorrelation sum, in sampling intervals."""
    x = np.asarray(values, dtype=float)-np.mean(values)
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    c = np.fft.irfft(f*f.conj(), n=2*n)[:n]/np.arange(n,0,-1)
    if c[0] == 0:
        return .5
    r = c/c[0]
    pairs = r[1:2*((n-1)//2)+1].reshape(-1,2).sum(axis=1)
    first = np.flatnonzero(pairs <= 0)
    positive = pairs[:first[0]] if len(first) else pairs
    return .5+np.minimum.accumulate(positive).sum() if len(positive) else .5


def statistics(values, interval):
    values = np.asarray(values)
    means = values.mean(axis=0)
    block_errors = {}
    for count in [10, 20, 40]:
        if len(values) < count*20:
            continue
        blocks = np.array([v.mean(axis=0) for v in np.array_split(values, count)])
        block_errors[str(count)] = (blocks.std(axis=0, ddof=1)/np.sqrt(count)).tolist()
    tau = np.array([correlation_time(v) for v in values.T])
    effective = len(values)/(2*tau)
    autocorr_se = values.std(axis=0, ddof=1)/np.sqrt(effective)
    se = np.maximum.reduce([autocorr_se]+[np.array(v) for v in block_errors.values()])
    return dict(mean=means.tolist(), se=se.tolist(), block_errors=block_errors,
                autocorrelation_se=autocorr_se.tolist(),
                tau_physical=(tau*interval).tolist(), effective_samples=effective.tolist())
