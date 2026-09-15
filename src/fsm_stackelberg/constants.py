"""Single source of truth for classification thresholds.

The strict-success rule ("Practical Optimal") requires the solver to report
OPTIMAL *and* the relative objective gap |z_hat - z*| / |z*| against the
reference optimum to stay within GAP_THRESHOLD; OPTIMAL with a larger gap
is a "Spurious Optimal" and triggers diagnosis. Runtime classification
(solver executor, experiment result) and offline re-analysis (summary
scripts) must all import the threshold from here.
"""

from typing import Optional

# Relative gap |z_hat - z*| / |z*| allowed under the strict-success rule.
GAP_THRESHOLD = 1e-3


def within_threshold(gap_percent: Optional[float],
                     fallback: Optional[bool] = None) -> Optional[bool]:
    """Strict-success gap test from a stored gap in percent units.

    Stored manifest flags (practical_optimal / status / verified hits) were
    classified under whatever threshold was live at run time, so offline
    re-analysis re-derives the verdict from the measured gap. When no gap is
    measurable (no ground-truth optimum), the stored classification is kept.
    """
    if gap_percent is None:
        return fallback
    return gap_percent <= GAP_THRESHOLD * 100
