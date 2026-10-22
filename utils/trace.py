"""Execution breadcrumbs, for reading a run's path through the code (see README.md).

Plain prints, the repo's own convention -- no logging framework, no flag. Nothing here computes
anything, so removing a line cannot change a number.

    at('utils/contact.py', 'bolt_on_friction_projection', 'applying the contact operator')
    -> in utils/contact.py (bolt_on_friction_projection), applying the contact operator

`once` keeps a breadcrumb inside a per-step function out of an eval study's log: a 40-cell lane
prints it once for the process instead of once per rollout step.
"""

_SEEN = set()


def at(file, func, doing):
    print(f'in {file} ({func}), {doing}')


def once(key):
    """True the first time it is asked for `key` in this process, False afterwards."""
    if key in _SEEN:
        return False
    _SEEN.add(key)
    return True
