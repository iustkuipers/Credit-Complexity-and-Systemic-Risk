## Q1 – Simple CDS Model (Continuous Premium Approximation)

Assumptions: continuous premium payments, constant (average) hazard rate up to maturity, and a fixed LGD.

For maturity $T$ with par CDS spread $R(T)$ (in decimal), the rule-of-thumb implies:

$$\lambda_{\text{avg}}(T) = \frac{R(T)}{LGD}$$

This yields the survival probability:

$$S(T)=Q(\tau>T)=\exp\{-\lambda_{\text{avg}}(T)\,T\}$$

and cumulative default probability $PD(0,T)=1-S(T)$. Forward hazards over $(T_{i-1},T_i]$ are implied by

$$\lambda_{fwd,i}=-\frac{\ln(S(T_i)/S(T_{i-1}))}{T_i-T_{i-1}}$$

and forward default probabilities by $PD(T_{i-1},T_i)=S(T_{i-1})-S(T_i)$. These outputs serve as a benchmark for the exact iterative stripping in Q2.
