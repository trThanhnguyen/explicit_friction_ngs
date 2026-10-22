"""fmetrics statistics (spec, P16): exact sign test (confirmatory), Wilcoxon (secondary,
pratt), Kendall tau_b, seeded cluster bootstrap, Holm over the two confirmatory endpoints, and
the twin-spread machinery (SC-8: no mu effect is reported without its twin spread).

Deterministic: seeded numpy Generator; scipy exact tests; all inputs are small per-sequence
arrays (fp64).
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sps


def sign_test_onesided(diffs: Sequence[float]):
    """Exact one-sided sign test of median(diff) > 0. Zeros dropped AND counted (P16)."""
    d = np.asarray([x for x in diffs if x is not None], dtype=np.float64)
    nz = d[d != 0.0]
    n_zero = int((d == 0.0).sum())
    if nz.size == 0:
        return {'p': None, 'n': 0, 'n_zero': n_zero, 'n_pos': 0, 'na_reason': 'all_zero_or_empty'}
    n_pos = int((nz > 0).sum())
    res = sps.binomtest(n_pos, nz.size, 0.5, alternative='greater')
    return {'p': float(res.pvalue), 'n': int(nz.size), 'n_zero': n_zero, 'n_pos': n_pos,
            'na_reason': None}


def wilcoxon_onesided(diffs: Sequence[float]):
    """Secondary, higher-power: one-sided Wilcoxon signed-rank, zero_method='pratt' (stated,
    P16). Assumption on record: symmetric difference distribution (tests the pseudomedian)."""
    d = np.asarray([x for x in diffs if x is not None], dtype=np.float64)
    if d.size < 3 or np.all(d == 0.0):
        return {'p': None, 'n': int(d.size), 'na_reason': 'too_few_or_all_zero'}
    res = sps.wilcoxon(d, alternative='greater', zero_method='pratt')
    return {'p': float(res.pvalue), 'n': int(d.size), 'na_reason': None,
            'assumption': 'symmetry (pseudomedian), zero_method=pratt'}


def kendall_tau_b(mus: Sequence[float], values: Sequence[Optional[float]]):
    """tau_b between mu and S; N/A if any value is missing or all equal (P16)."""
    if any(v is None for v in values):
        return {'tau_b': None, 'na_reason': 'missing_mu_point'}
    v = np.asarray(values, dtype=np.float64)
    if np.all(v == v[0]):
        return {'tau_b': None, 'na_reason': 'all_values_equal'}
    tau, _ = sps.kendalltau(np.asarray(mus, dtype=np.float64), v, variant='b')
    return {'tau_b': float(tau), 'na_reason': None}


def bootstrap_median_ci(values: Sequence[Optional[float]], B: int, seed: int, level=0.95):
    """Cluster bootstrap over sequences (the unit of replication): percentile CI of the median.
    Resamples ONLY valid sequences and reports valid-n (P10)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    n = v.size
    if n == 0:
        return {'median': None, 'ci_lo': None, 'ci_hi': None, 'valid_n': 0}
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, n, size=(B, n))], axis=1)
    a = (1.0 - level) / 2.0
    return {'median': float(np.median(v)), 'ci_lo': float(np.quantile(meds, a)),
            'ci_hi': float(np.quantile(meds, 1.0 - a)), 'valid_n': n}


def holm(pvals: Dict[str, Optional[float]]):
    """Holm step-down over the named confirmatory endpoints; None p-values excluded (counted)."""
    items = [(k, p) for k, p in pvals.items() if p is not None]
    m = len(items)
    out = {k: {'p_raw': p, 'p_holm': None} for k, p in pvals.items()}
    for rank, (k, p) in enumerate(sorted(items, key=lambda kv: kv[1])):
        out[k]['p_holm'] = min(1.0, float((m - rank) * p))
    # enforce monotonicity of adjusted p-values
    prev = 0.0
    for k, p in sorted(items, key=lambda kv: kv[1]):
        out[k]['p_holm'] = max(out[k]['p_holm'], prev)
        prev = out[k]['p_holm']
    return out


def twin_spread(per_twin_values: Dict[str, Dict[str, Optional[float]]]):
    """Twin machinery (SC-8): per_twin_values[replicate][seq_key] -> metric value.
    Returns per-seq max|pairwise diff| and its dataset median/max — THE noise floor that every
    mu effect must exceed. None where fewer than 2 valid replicates exist."""
    reps = list(per_twin_values)
    if len(reps) < 2:
        return {'per_seq': {}, 'median_spread': None, 'max_spread': None,
                'na_reason': 'fewer_than_2_replicates'}
    keys = set.intersection(*[set(per_twin_values[r]) for r in reps])
    per_seq = {}
    for k in sorted(keys):
        vals = [per_twin_values[r][k] for r in reps]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            per_seq[k] = float(max(vals) - min(vals))
    if not per_seq:
        return {'per_seq': {}, 'median_spread': None, 'max_spread': None,
                'na_reason': 'no_common_valid_sequences'}
    arr = np.asarray(list(per_seq.values()))
    return {'per_seq': per_seq, 'median_spread': float(np.median(arr)),
            'max_spread': float(arr.max()), 'na_reason': None}


def effect_with_spread(effect: Optional[float], spread: Optional[float]):
    """SC-8 gate: an effect is only 'clear' if it exceeds its twin spread. Never suppresses the
    number — annotates it."""
    if effect is None:
        return {'effect': None, 'twin_spread': spread, 'exceeds_twin_spread': None}
    if spread is None:
        return {'effect': effect, 'twin_spread': None, 'exceeds_twin_spread': None,
                'note': 'twin spread unavailable — effect NOT gated (SC-8 requires it)'}
    return {'effect': effect, 'twin_spread': spread,
            'exceeds_twin_spread': bool(abs(effect) > spread)}
