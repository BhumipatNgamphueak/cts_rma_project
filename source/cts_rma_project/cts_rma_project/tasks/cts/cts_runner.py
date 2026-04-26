# tasks/cts/cts_runner.py
"""
CTS runner — thin wrapper around RSL-RL OnPolicyRunner.

Curriculum removed: the CTS novelty is concurrent teacher-student training,
not velocity-command scheduling. Command ranges are identical to Baseline and
RMA (full SharedEnvCfg range from step 0) so the comparison is fair.
"""
from __future__ import annotations
from rsl_rl.runners import OnPolicyRunner  # type: ignore


class CTSRunner(OnPolicyRunner):
    """Alias for OnPolicyRunner — no curriculum logic."""
    pass
