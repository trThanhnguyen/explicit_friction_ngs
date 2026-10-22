"""fmetrics report: per-sequence long-format CSV, summary/stats JSON, and the output-inspection
invariants (Phase-3 checklist: impossible values, coverage outside [0,1], tautological metrics).

Row schema (one row per number, never pooled-only — Phase-3 requirement):
  seq, arm, mu, metric, support, variant, eps_v, value, valid, na_reason, extra(json)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class RowSink:
    def __init__(self):
        self.rows = []

    def add(self, seq, arm, mu, metric, value, *, support='', variant='', eps_v=None,
            na_reason=None, **extra):
        self.rows.append(dict(seq=seq, arm=arm, mu=mu, metric=metric, support=support,
                              variant=variant, eps_v=eps_v,
                              value=(None if value is None else float(value)),
                              valid=value is not None, na_reason=na_reason,
                              extra=json.dumps(extra, default=_np_json) if extra else ''))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def values(self, metric, arm=None, mu=None, support='', variant='', eps_v=None):
        """{seq: value-or-None} selector for the stats layer."""
        out = {}
        for r in self.rows:
            if r['metric'] != metric or (arm is not None and r['arm'] != arm):
                continue
            if mu is not None and r['mu'] != mu:
                continue
            if r['support'] != support or r['variant'] != variant:
                continue
            if (eps_v is None) != (r['eps_v'] is None):
                continue
            if eps_v is not None and abs(r['eps_v'] - eps_v) > 1e-15:
                continue
            out[r['seq']] = r['value']
        return out


def _np_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


BOUNDED_01 = {'cov', 'fisc', 'fisc_C', 'ocr', 'onset_coverage', 'csa_ba', 'pen_rate',
              'w_end_pos_fraction', 'free_mass_frac'}
NONNEG = {'S', 'r_moll_mean', 'r_moll_p95', 'r_cone_mean', 'D_s', 'D_spec', 'M_c',
          'pen_depth_mean', 'pen_depth_max', 'accel_q99'}


def inspect_invariants(df: pd.DataFrame) -> list:
    """The mandated output inspection: impossible values, coverage outside [0,1], NaN/inf,
    negative nonnegatives. Returns a list of human-readable warnings (also serialized)."""
    warns = []
    valid = df[df['valid']]
    bad_num = valid[~np.isfinite(valid['value'].astype(float))]
    for _, r in bad_num.iterrows():
        warns.append(f"non-finite value: {r['metric']} {r['arm']} mu={r['mu']} seq={r['seq']}")
    for m in BOUNDED_01:
        sel = valid[valid['metric'] == m]
        bad = sel[(sel['value'] < -1e-9) | (sel['value'] > 1.0 + 1e-9)]
        for _, r in bad.iterrows():
            warns.append(f"[0,1] violation: {m}={r['value']:.6g} {r['arm']} mu={r['mu']} seq={r['seq']}")
    for m in NONNEG:
        sel = valid[valid['metric'] == m]
        bad = sel[sel['value'] < -1e-12]
        for _, r in bad.iterrows():
            warns.append(f"negative {m}={r['value']:.3g} {r['arm']} mu={r['mu']} seq={r['seq']}")
    return warns


def write_outputs(out_dir: Path, sink: RowSink, summary: dict, stats: dict, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = sink.frame()
    df.to_csv(out_dir / 'per_seq_metrics.csv', index=False)
    warns = inspect_invariants(df)
    summary = dict(summary)
    summary['invariant_warnings'] = warns
    summary['metric_config'] = config.to_dict()
    summary['metric_config_hash'] = config.config_hash()
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=_np_json)
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=_np_json)
    with open(out_dir / 'metric_config.json', 'w') as f:
        f.write(config.to_json())
    return warns
