"""fmetrics — the friction metric suite.

Offline, deterministic evaluation of PSR / FISC / OCR (Tier 1), CSA (Tier 2), Coulomb-graph
residuals + dissipation (Tier 3), and guardrails, from rollout pkls produced by
`evaluate.py --flog`. Entry point: an internal tool
"""
from .config import MetricSuiteConfig  # noqa: F401

__version__ = '0.1.0'
