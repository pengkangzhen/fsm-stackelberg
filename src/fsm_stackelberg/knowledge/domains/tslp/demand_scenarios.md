---
name: tslp-demand-scenarios
description: Scenario set Omega, probabilities pi, random export demand eta
---

# Demand scenarios

- Export demand \(\eta_{n,t}^\omega\) is given per `(scenario, node, period)` in the data.
- Probabilities \(\pi_\omega\) sum to 1 (equal weight for the smoke instance).
- Nominal means \(\eta_{n,t}\) (`eta_bar`) are provided for reference; the DEP must use the scenario table `eta`, not only the mean.
