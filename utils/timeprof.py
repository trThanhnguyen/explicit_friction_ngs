"""Per-stage time profiling for the GNN rollout — the counterpart of ppf's own instrumentation.

WHY THIS EXISTS
---------------
A rollout's plain wall-clock (`metrics['time']`) is not the method's cost. It includes the
harness's own instrumentation — the `ncoll` tri-tri safecheck, the trajectory record, the pickle
write — and it is measured on a host that may be running a second eval lane, so it overstates the
method by a large and variable factor.

This module provides the missing breakdown, so `method_total` (network + operator) can be quoted
against a solver's own `solver_total` and the harness cost can be named rather than silently
included. Time both sides the same way, on an uncontended host, or the ratio is meaningless.

WHAT IS MEASURED
----------------
Stages nest: run > init/phase > substage. `substage` rows sum INTO their parent phase and
`phase` rows sum INTO `method_total`, so summing every row double-counts — filter on `stage_kind`
before aggregating.

  run_total        whole valid_rollout                          <- ppf `run_total`
  method_total     model_fwd + operator: THE comparable number  <- ppf `solver_total`
  seq_init         _add_cloth_obj + prepare_sample + warm-up    <- ppf `initialize`
  model_fwd        the network step                             <- ppf `advance_step`
    icontour         ContourCraft ICM contour build
    prepare_inputs   feature + graph (world-edge) assembly
    learned_model    encode-process-decode
    get_position     accel -> pred_pos
  operator         the bolt-on friction projection (no ppf analogue: ppf's friction is in-solve)
    op_correspond    nearest body face (knn) + next-pose face geometry + the d<eps gate
    op_lambda lambda_n assembly (barrier or force-balance) + the mollifier
    op_cone          displacement-space block soft-threshold
    op_normal endpoint restore / surface march / multi-normal transport
    op_patch         R3 patch coupling: C1 compliant relaxation, or C0 spreading (mutually
                     exclusive by guard, so the stage is unambiguous within one run)
  safecheck_ncoll  calc_tritri_collisions2 — EVAL INSTRUMENTATION, excluded from method_total
  collect_sample   per-step sample assembly (harness)
  record           trajectory append (harness)

TWO CLOCKS, AND WHICH ONE TO BELIEVE
------------------------------------
Every stage is timed twice: `wall` (perf_counter on the host) and `cuda` (a CUDA event pair).
Neither alone is honest. On an async stream a host-side `wall` around a GPU region measures kernel
LAUNCH, not execution; a `cuda` event pair around a host-bound region measures ~0 because nothing
was enqueued between the markers. The rollout loop happens to be close to synchronous already —
the existing `TorchTimer(metrics_dict, 'hood_time')` synchronises once per step, `knn_points` and
the tri-tri safecheck read shapes back to the host, and `torch.isfinite(pred_pos).all()` is a host
branch — so the two clocks should agree. THAT AGREEMENT IS THE VALIDITY CHECK: the assembler emits
both, plus `unaccounted` = run_total - sum(top-level stages). If wall and cuda disagree by more
than a few percent on a GPU stage, the attribution is not trustworthy and must be said out loud.

ZERO-PERTURBATION DISCIPLINE
----------------------------
A profiler that synchronises per stage would serialise the pipeline and inflate the very total it
reports. So:
  * CUDA events are recorded and NEVER read inline. They are drained (one synchronize) only when
    the preallocated pool is exhausted, ~4 times per sequence rather than ~9000 times.
  * Counters that live on the GPU (`n_contact`, `n_altered`) are stashed as 0-dim tensors and read
    at end-of-sequence. Calling `.item()` inline would sync every step.
  * The whole thing is OFF by default. Disabled, `stage()` returns a shared null context manager
    and `count()` returns immediately — no events, no allocations, and no change to any tensor.
    The off path is byte-identical to the un-instrumented runner by construction.

Quantify the residual cost rather than asserting it: run the same cells with profiling on and
off and report the delta.
"""

import time

import torch

__all__ = ['PROF', 'TimeProfiler']

# Pool size in event PAIRS. ~9000 stage-entries per 718-step sequence (12 stages + operator's
# per-example entries), so 4096 pairs drains ~2-3 times per long sequence.
_POOL_PAIRS = 4096


class _NullStage:
    """The disabled-path context manager. Allocation-free and shared."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _NullStage()


class _Stage:
    """One timing scope. Instances are pooled per name to avoid per-step allocation.

    All the state lives on the profiler's `_open` stack, so the same object can be re-entered
    (nested or recursive) without clobbering anything.
    """

    __slots__ = ('prof', 'name')

    def __init__(self, prof, name):
        self.prof = prof
        self.name = name

    def __enter__(self):
        self.prof.begin(self.name)
        return self

    def __exit__(self, *exc):
        self.prof.end(self.name)
        return False


class TimeProfiler:
    """Deferred-readback stage timer. A single module-level instance (`PROF`) is used everywhere.

    Nothing here mutates simulation state; the class only reads shapes and stashes 0-dim tensors.
    """

    def __init__(self):
        self.enabled = False
        self._free = []          # [(start_event, end_event)] available for a new scope
        self._open = []          # stack of scopes currently inside `with`
        self._pending = []       # (name, ev_start, ev_end) awaiting a drain
        self._wall = {}          # name -> [ms]
        self._cuda = {}          # name -> [ms]
        self._counts = {}        # name -> [float]           (host-side, immediate)
        self._counts_t = {}      # name -> [0-dim tensor]    (device-side, deferred)
        self._stages = {}        # name -> pooled _Stage
        self._run_meta = {}      # run-level provenance, seeded into every sequence's meta
        self._meta = {}
        self._t0 = None
        self._n_drains = 0

    # ---- lifecycle ------------------------------------------------------------------------- #

    def enable(self, device=None):
        """Turn profiling on and preallocate the event pool (one-off cost, before any timing)."""
        self.enabled = True
        if not self._free and torch.cuda.is_available():
            if device is not None:
                torch.cuda.set_device(device)
            self._free = [(torch.cuda.Event(enable_timing=True),
                           torch.cuda.Event(enable_timing=True))
                          for _ in range(_POOL_PAIRS)]

    def set_run_meta(self, **meta):
        """Run-level provenance (checkpoint, config, mu, gpu, host, concurrency, ...).

        Carried into every sequence's sidecar so no timing number can travel without the settings
        that produced it — the same rule `p42_timeprof.py` enforces by replaying each cell's argv.
        """
        self._run_meta.update(meta)

    def begin_sequence(self, **meta):
        """Reset all accumulators. Called once per rollout; harmless when disabled."""
        if not self.enabled:
            return
        self._recycle_pending()
        self._open.clear()
        self._wall, self._cuda = {}, {}
        self._counts, self._counts_t = {}, {}
        self._meta = dict(self._run_meta, **meta)
        self._n_drains = 0
        self._t0 = time.perf_counter()

    def end_sequence(self):
        """Drain the events, read the deferred counters, and return the summary dict.

        Returns None when disabled, so a caller can `if (s := PROF.end_sequence()): ...`.
        """
        if not self.enabled:
            return None
        run_wall_ms = (time.perf_counter() - self._t0) * 1e3 if self._t0 else 0.0
        self._drain()
        counts = {k: list(v) for k, v in self._counts.items()}
        for k, tensors in self._counts_t.items():
            if tensors:
                counts[k] = [float(x) for x in torch.stack(tensors).reshape(-1).cpu()]
        return {
            'meta': dict(self._meta),
            'run_wall_ms': run_wall_ms,
            'n_drains': self._n_drains,
            'wall_ms': {k: list(v) for k, v in self._wall.items()},
            'cuda_ms': {k: list(v) for k, v in self._cuda.items()},
            'counters': counts,
        }

    # ---- measurement ----------------------------------------------------------------------- #

    def stage(self, name):
        """`with PROF.stage('op_cone'): ...` — the preferred form where indentation is free."""
        if not self.enabled:
            return _NULL
        s = self._stages.get(name)
        if s is None:
            s = self._stages[name] = _Stage(self, name)
        return s

    def begin(self, name):
        """Open a scope without a `with`. For long blocks inside code that must not be re-indented
        (`utils/contact.py`'s operator body), where wrapping would touch ~130 lines of a module
        whose off-path byte-identity is a working agreement. Must be paired with `end(name)`."""
        if not self.enabled:
            return
        ev_s, ev_e = self._take_pair()
        if ev_s is not None:
            ev_s.record()
        self._open.append((name, ev_s, ev_e, time.perf_counter()))

    def end(self, name):
        if not self.enabled or not self._open:
            return
        if self._open[-1][0] != name:
            # Mismatched pairing (an early `return`/`raise` inside a begin/end block skipped its
            # end). Unwind to the matching name rather than silently attributing to the wrong
            # stage; discard the abandoned scopes, they measured an aborted path.
            names = [o[0] for o in self._open]
            if name not in names:
                return
            while self._open[-1][0] != name:
                _, s_ev, e_ev, _ = self._open.pop()
                if s_ev is not None:
                    self._free.append((s_ev, e_ev))
        nm, ev_s, ev_e, t0 = self._open.pop()
        self._wall.setdefault(nm, []).append((time.perf_counter() - t0) * 1e3)
        if ev_e is not None:
            ev_e.record()
            self._pending.append((nm, ev_s, ev_e))

    def count(self, name, value):
        """Record a counter sample. A tensor is stashed and read at end_sequence (no sync)."""
        if not self.enabled:
            return
        if torch.is_tensor(value):
            self._counts_t.setdefault(name, []).append(value.detach().reshape(()).float())
        else:
            self._counts.setdefault(name, []).append(float(value))

    # ---- internals ------------------------------------------------------------------------- #

    def _take_pair(self):
        """Hand out an event pair. A pair is OWNED by its scope until the next drain reads it.

        Open scopes simply hold theirs out of the free list, so a `with` spanning a drain can never
        have its events reused underneath it — the failure mode a bare index-and-reset pool has.
        """
        if not self._free:
            self._drain()
        if not self._free:
            # Every pair is held by a still-open scope (nesting deeper than the pool, i.e. never in
            # practice) or CUDA is unavailable. Grow rather than mis-time.
            if not torch.cuda.is_available():
                return None, None
            return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        return self._free.pop()

    def _drain(self):
        """Synchronize once, read every CLOSED event pair into `_cuda`, and recycle it."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for name, ev_s, ev_e in self._pending:
            try:
                self._cuda.setdefault(name, []).append(ev_s.elapsed_time(ev_e))
            except RuntimeError:
                pass                       # event not ready / not recorded — drop the sample
        self._recycle_pending()
        self._n_drains += 1

    def _recycle_pending(self):
        for _, ev_s, ev_e in self._pending:
            if ev_s is not None:
                self._free.append((ev_s, ev_e))
        self._pending.clear()


PROF = TimeProfiler()
