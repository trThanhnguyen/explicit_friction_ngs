"""fmetrics: the serialized metric-suite configuration.

Every threshold, floor, convention label, and hash that any number in a results file depends on
lives here and is embedded verbatim into every output (per_seq_metrics.csv sidecar +
summary.json). A results file whose config hash differs from the sweep's must not be pooled
(P18); `assert_run_meta` enforces the model-constant agreements (r_c, eps_op, warm-up) between
this config and each rollout's friction_log_meta.

Unit/time-level conventions (single source: spec –):
  * h [s] frame step; all velocities = displacement/h.
  * eps_op [m/s] = the OPERATOR's mollifier band (bolt_on.eps_v) — a model constant, distinct
    from the metric's swept eps_v (P4).
  * r_c [m] = the operator's gate margin (bolt_on.eps), the contact-set radius (SC-6).
  * t0 = first metric pkl frame = 2 + warmup_steps (P17); velocity samples exist for
    t in [t0, T-2].
"""
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


SCHEMA_VERSION = 1


@dataclass
class MetricSuiteConfig:
    # time / frames
    h: float = 1.0 / 30.0
    warmup_steps: int = 30              # asserted == run meta; t0 = 2 + warmup_steps
    # contact convention (convention pinned per SC-6/D-3)
    r_c: float = 3e-3                   # operator gate margin bolt_on.eps [m]
    eps_op: float = 1e-2                # operator mollifier band bolt_on.eps_v [m/s]
    up_axis: int = 1                    # Y-up (P2)
    oracle_max_dist: float = 0.5        # convention-B query radius [m]
    # thresholds (stated in advance)
    eps_v_sweep: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    eps_v_central: float = 1e-3         # [m/s]
    v_act: float = 3e-3                 # [m/s] FROZEN BY PILOT (an internal tool); > eps_v_central
    K_act: int = 5
    K_stick: int = 5
    delta_onset: int = 15               # frames (0.5 s) —, D-7
    k_ring: int = 2                     # strict-FISC face k-ring (vertex-shared adjacency), D-5
    common_min: int = 3                 # q rule: contact in >= common_min of the mu runs (spec: 3-of-4)
    # residual scales / floors
    eta: float = 1e-8                   # [N]
    num_zero_guard: float = 1e-12       # [m/s] P4 numerical zero (supplement classifier only)
    lambda_min_frac: float = 0.01       # lambda_min = frac * median(pooled mu=0 force>0)  (D-6)
    lambda_min_value: Optional[float] = None  # resolved at runtime from mu=0 logs; serialized back
    s0_floor: float = 1e-4              # [m/s] S(0) floor below which dS_rel is N/A
    sat_denom_floor: float = 1e-5       # [m/s] saturation-ratio denominator floor
    contact_masstime_frac_floor: float = 0.002  # sequence N/A floor. USER REVISION 2026-07-16
    # (spec): the spec- stated in advance 1% sits mid-population at the operator band
    # r_c=3mm (G1 medians 0.9-3.6%, minima ~0.5%) and censors near-arbitrarily; 0.2% sits below
    # every G1 sequence while still N/A-ing genuinely graze-only support. 1%-floor G1 outputs
    # retained as the stated in advance reference (out_v13TC/out_v11c).
    free_mass_frac_floor: float = 0.05  # P15 freeze-ratio support floor
    # statistics
    B_boot: int = 2000
    boot_seed: int = 260715
    holm_endpoints: List[str] = field(default_factory=lambda: ['H1_S', 'H1_O'])
    # bookkeeping
    generation: str = 'G1'              # P18: one instrumented generation per comparison
    conventions: Dict[str, str] = field(default_factory=lambda: {
        'tier1': 'B(advected barycentric)', 'tier2_pred': 'A(operator centroid)',
        'tier2_obs': 'B(advected barycentric)', 'tier3': 'A(operator centroid)',
        'guardrails': 'B + root-frame(Kabsch)'})
    schema_version: int = SCHEMA_VERSION

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> 'MetricSuiteConfig':
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'MetricSuiteConfig: unknown fields {sorted(unknown)}')
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> 'MetricSuiteConfig':
        from omegaconf import OmegaConf
        return cls.from_dict(dict(OmegaConf.to_container(OmegaConf.load(path))))

    # ---------------------------------------------------------------- validation
    def validate(self):
        assert self.eps_v_central < self.v_act, \
            'spec: eps_v < v_act must hold (activity and stick thresholds must be separated)'
        assert self.eps_v_central in self.eps_v_sweep, 'central eps_v must be a sweep member'
        assert self.k_ring >= 0 and self.K_act >= 1 and self.K_stick >= 1 and self.delta_onset >= 1
        assert 0.0 < self.contact_masstime_frac_floor < 1.0
        return self

    def t0(self) -> int:
        return 2 + int(self.warmup_steps)

    def assert_run_meta(self, meta: dict, run_name: str = '?'):
        """P14/P18: the model constants this config claims must equal what the rollout ran with,
        and the force channel must be the cone-driving lambda (no smoothing / no render-only)."""
        if meta is None:
            raise ValueError(f'[{run_name}] rollout has no friction_log_meta (need --flog runs)')
        chk = [('r_c', self.r_c, float(meta.get('r_c_gate_eps', float("nan")))),
               ('eps_op', self.eps_op, float(meta.get('eps_op', float("nan")))),
               ('warmup_steps', self.warmup_steps, int(meta.get('warmup_steps', -1))),
               ('h', self.h, float(meta.get('h', float("nan"))))]
        for name, want, got in chk:
            if abs(float(want) - float(got)) > 1e-12:
                raise ValueError(f'[{run_name}] config/{name}={want} but rollout meta says {got}')
        if int(meta.get('lambda_smooth_iters', 0)) != 0 or float(meta.get('lambda_smooth_window', 0.0)) != 0.0:
            raise ValueError(f'[{run_name}] lambda smoothing was ON — force channel != cone lambda (caveat)')
        gen = str(meta.get('generation', 'unstamped'))
        if gen != str(self.generation):
            raise ValueError(f'[{run_name}] generation {gen!r} != config generation {self.generation!r} (P18)')
