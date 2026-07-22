---
name: tslp-two-stage-structure
description: Window DEP structure [D1]–[D2]; first-stage y vs second-stage inland recourse
---

# Two-stage structure (TSLP ECR)

For one planning window \(\mathcal{T}\):

1. **Stage 1 (non-anticipative):** hub sea flows \(y^{\mathrm{in}}, y^{\mathrm{out}}\) chosen *before* demand scenarios are known.
2. **Stage 2 (scenario-wise):** for each \(\omega\in\Omega\), inland \(x^\omega\), inventory \(I^\omega\), lease \(r^\omega\), spill \(s^\omega\) after \(\eta^\omega\) is revealed.

Solve the **extensive-form DEP** (all scenarios in one LP). Variables are continuous \(\mathbb{R}_+\) (TEU). Empty supply \(\xi\) is deterministic; only demand is stochastic.
