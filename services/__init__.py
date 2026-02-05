# services/__init__.py

# Import core functions from model.py
from .model import (
    precompute_thresholds,
    migrate_one,
    migrate_many,
    migrate_many_to_indices,
)

# Import core functions from risk_metrics.py
from .risk_metrics import (
    values_to_losses,
    expected_value,
    var,
    es,
    summarize_losses,
    summarize_case_metrics,
    es_over_var_ratio,
)

# Import core functions from simulation.py
from .simulation import (
    simulate_one_factor_asset_returns,
    simulate_portfolio_values_t1,
)

# Import core functions from validation.py
from .validation import (
    bbb_default_threshold_analytic,
    assert_bbb_default_threshold_consistency,
    convergence_check_var995,
    find_stable_N,
)

# Optional: define public API explicitly
__all__ = [
    # model
    "precompute_thresholds",
    "migrate_one",
    "migrate_many",
    "migrate_many_to_indices",
    # risk_metrics
    "values_to_losses",
    "expected_value",
    "var",
    "es",
    "summarize_losses",
    "summarize_case_metrics",
    "es_over_var_ratio",
    # simulation
    "simulate_one_factor_asset_returns",
    "simulate_portfolio_values_t1",
    # validation
    "bbb_default_threshold_analytic",
    "assert_bbb_default_threshold_consistency",
    "convergence_check_var995",
    "find_stable_N",
]
