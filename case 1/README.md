# Credit Complexity and Systemic Risk - Case 1

Quantitative analysis of credit portfolio risk, migration, and systemic effects using Monte Carlo simulation and stress testing.

## Project Overview

This project implements a comprehensive framework for analyzing credit risk in a bond portfolio using:
- **Credit migration models**: Merton-style asset return framework
- **Monte Carlo simulation**: Portfolio value distribution under correlated defaults
- **Risk metrics**: VaR, Expected Shortfall (ES), and convergence diagnostics
- **Validation framework**: Threshold consistency checks and convergence analysis

## Project Structure

```
.
├── data/
│   └── data.py              # Constants: ratings, transition matrix, bond values
├── services/
│   ├── model.py             # Credit migration: Z-thresholds and asset returns
│   ├── simulation.py        # Monte Carlo portfolio value simulation
│   ├── risk_metrics.py      # VaR, ES, and loss analysis
│   └── validation.py        # Threshold validation and convergence checks
├── test/
│   ├── test_data.py         # Tests for data module (36 tests)
│   ├── test_model.py        # Tests for model module (36 tests)
│   ├── test_simulation.py   # Tests for simulation module (35 tests)
│   ├── test_risk_metrics.py # Tests for risk metrics module (60 tests)
│   └── test_validation.py   # Tests for validation module (36 tests)
└── README.md                # This file
```

## Core Modules

### `data.py` - Case Data and Ratings
Defines canonical ratings order and case primitives:
- **RATINGS**: AAA, AA, A, BBB, BB, B, CCC, D (best to worst)
- **TRANSITION**: 7×8 migration matrix (Case 1, Table 1)
- **BOND_VALUE_T0/T1**: Bond prices at t=0 and t=1 (per 100 notional)
- **make_portfolio()**: Portfolio creation with validation
- **portfolio_value_t0()**: Deterministic portfolio value calculation

### `model.py` - Credit Migration
One-factor Merton-style credit migration model:
- **precompute_thresholds()**: Convert migration probabilities to Z-score thresholds
- **migrate_one()**: Single issuer asset return → rating migration
- **migrate_many()**: Vectorized migration for N issuers
- **migrate_many_to_indices()**: Efficient index-based migration for value lookups

### `simulation.py` - Monte Carlo Simulation
Portfolio value simulation under correlated defaults:
- **simulate_one_factor_asset_returns()**: X_i = √ρ·Y + √(1-ρ)·ε_i
- **simulate_portfolio_values_t1()**: Main simulation engine
  - Controls: notional, concentration (n_issuers_per_rating), correlation (rho), N scenarios
  - Returns: array of portfolio values at t=1

### `risk_metrics.py` - Risk Quantification
Loss-based risk analysis:
- **values_to_losses()**: Convert V0, V1 → Loss = V0 - V1
- **var()**: Value-at-Risk at confidence level α
- **es()**: Expected Shortfall (conditional VaR)
- **summarize_case_metrics()**: Combined E[V1], VaR, ES report
- **es_over_var_ratio()**: Tail fatness indicator

### `validation.py` - Quality Assurance
Model validation and convergence diagnostics:
- **bbb_default_threshold_analytic()**: Analytic BBB→D threshold
- **assert_bbb_default_threshold_consistency()**: Validate simulated thresholds
- **convergence_check_var995()**: Multi-seed VaR stability analysis
- **find_stable_N()**: Determine sufficient N for stable estimates

## Usage Examples

### Basic Portfolio Analysis

```python
from data import make_portfolio, portfolio_value_t0
from simulation import simulate_portfolio_values_t1
from risk_metrics import summarize_case_metrics

# Define portfolio
portfolio = make_portfolio({"AAA": 0.6, "AA": 0.3, "BBB": 0.1})
v0 = portfolio_value_t0(portfolio, total_notional=1500.0)

# Run simulation
v1 = simulate_portfolio_values_t1(
    portfolio=portfolio,
    total_notional=1500.0,
    n_issuers_per_rating=100,    # diversified
    rho=0.33,                     # intermediate correlation
    N=10000,                      # scenarios
    seed=42
)

# Compute risk metrics
metrics = summarize_case_metrics(v0, v1, alphas=(0.90, 0.95, 0.995))
print(f"E[V1]: {metrics['expected_value']:.2f}")
print(f"VaR(99.5%): {metrics['var_0.995']:.2f}")
print(f"ES(99.5%): {metrics['es_0.995']:.2f}")
```

### Concentration vs Diversification

```python
# Concentrated portfolio (1 issuer per rating)
v1_conc = simulate_portfolio_values_t1(
    portfolio=portfolio, total_notional=1500.0, 
    n_issuers_per_rating=1, rho=0.33, N=5000
)

# Diversified portfolio (100 issuers per rating)
v1_div = simulate_portfolio_values_t1(
    portfolio=portfolio, total_notional=1500.0,
    n_issuers_per_rating=100, rho=0.33, N=5000
)

# Concentrated has higher volatility and tail risk
print(f"Concentrated std: {np.std(v1_conc):.2f}")
print(f"Diversified std:  {np.std(v1_div):.2f}")
```

### Convergence Analysis

```python
from validation import find_stable_N

# Find N with relative VaR dispersion <= 1%
result = find_stable_N(
    total_notional=1500.0,
    rho=0.33,
    n_issuers_per_rating=10,
    portfolio_weights={"BBB": 1.0},
    seeds=[1, 2, 3, 4, 5],
    N_grid=[1000, 2000, 5000, 10000],
    rel_tol=0.01
)

print(f"Recommended N: {result['N']}")
print(f"Status: {result['status']}")
```

## Testing

**203 tests** across 5 test modules provide comprehensive coverage:

```bash
# Run all tests
pytest test/ -v

# Run specific module
pytest test/test_data.py -v
pytest test/test_model.py -v
pytest test/test_simulation.py -v
pytest test/test_risk_metrics.py -v
pytest test/test_validation.py -v

# Run with coverage
pytest test/ --cov=data --cov=services --cov-report=html
```

### Test Coverage

| Module | Tests | Focus |
|--------|-------|-------|
| test_data.py | 36 | Constants, ratings, portfolios, bond values |
| test_model.py | 36 | Thresholds, migration, distribution preservation |
| test_simulation.py | 35 | One-factor model, convergence, diversification |
| test_risk_metrics.py | 60 | VaR, ES, tail behavior, consistency |
| test_validation.py | 36 | Threshold validation, convergence diagnostics |

## Key Features

### One-Factor Merton Model
Asset returns with systematic and idiosyncratic components:

$$X_i = \sqrt{\rho} \cdot Y + \sqrt{1-\rho} \cdot \epsilon_i$$

where:
- $Y \sim N(0,1)$ is the common factor (systematic risk)
- $\epsilon_i \sim N(0,1)$ are idiosyncratic shocks
- $\rho \in [0,1]$ controls default correlation

### Credit Migration via Thresholds
Each current rating maps to migration thresholds computed from transition probabilities:

$$\text{Threshold}_k = \Phi^{-1}\left(\sum_{j \leq k} P(\text{rate}_j | \text{current})\right)$$

Issuer defaults if $X_i \leq \text{Threshold}_D$.

### Loss Convention

$$\text{Loss} = V_0 - V_1$$

Positive losses represent portfolio deterioration.

### Risk Metrics
- **VaR**: Portfolio loss quantile at confidence level $\alpha$
- **ES**: Expected loss in worst $1-\alpha$ tail scenarios
- **ES/VaR ratio**: Tail shape indicator (normal ≈ 1.05-1.15, fat tails > 1.5)

## Configuration Parameters

| Parameter | Range | Meaning |
|-----------|-------|---------|
| **rho** | [0, 1] | Asset correlation (0=idiosyncratic, 1=perfect) |
| **N** | 1,000+ | Number of MC scenarios (larger=more stable) |
| **n_issuers_per_rating** | 1+ | Concentration (1=concentrated, 100+= diversified) |
| **alpha** | (0, 1) | Confidence level for VaR/ES (0.995=99.5%) |

## Performance Notes

- **Memory**: Simulation scales ~O(N × n_ratings × n_issuers_per_rating)
- **Time**: Monte Carlo is the bottleneck; ~10k scenarios on modern CPU ≈ 1-2s
- **Accuracy**: VaR converges at ~O(1/√N); use N ≥ 5,000 for stability

## Dependencies

- numpy >= 1.20
- scipy >= 1.7
- pytest >= 7.0 (for testing)

## References

Case 1 based on:
- **Merton (1974)**: Structural credit risk model via asset values
- **Vasicek (1987)**: One-factor Gaussian copula for default correlation
- **Jorion (2006)**: Value-at-Risk and Expected Shortfall risk metrics

## Author Notes

This implementation emphasizes:
1. **Clarity**: Explicit conventions (e.g., Loss = V0 - V1, ratings AAA→D order)
2. **Modularity**: Separate data, model, simulation, metrics, validation layers
3. **Testing**: 203 unit tests ensuring robustness and catching sign/logic errors
4. **Flexibility**: Parameterized portfolio, correlation, concentration for scenario analysis
