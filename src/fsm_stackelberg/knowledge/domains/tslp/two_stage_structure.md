---
name: tslp-two-stage-structure
description: Window DEP structure [D1]–[D2]; first-stage y vs second-stage inland recourse
---

# Two-stage structure (TSLP ECR)

For one planning window \(\mathcal{T}\):

1. **Stage 1 (non-anticipative):** hub sea flows \(y^{\mathrm{in}}, y^{\mathrm{out}}\) chosen *before* demand scenarios are known.
2. **Stage 2 (scenario-wise):** for each \(\omega\in\Omega\), inland \(x^\omega\), inventory \(I^\omega\), lease \(r^\omega\), spill \(s^\omega\) after \(\eta^\omega\) is revealed.

Solve the **extensive-form DEP** (all scenarios in one LP). Variables are continuous \(\mathbb{R}_+\) (TEU). Empty supply \(\xi\) is deterministic; only demand is stochastic.

## Objective (extensive form)

\[
\begin{aligned}
\min\;
&\sum_{t}\sum_{i\in\mathrm{hubs}}\bigl(c^{\mathrm{sea,in}} y_{i,t}^{\mathrm{in}}+c^{\mathrm{sea,out}} y_{i,t}^{\mathrm{out}}\bigr) \\
&+\sum_{\omega}\pi_\omega\sum_{t}\Bigl(
\sum_{(i,j,\kappa)} c_{ij\kappa}^{\mathrm{repo}} x_{ij\kappa,t}^\omega
+\sum_n c_n^{\mathrm{hold}} I_{n,t}^\omega
+\sum_n c_n^{\mathrm{lease}} r_{n,t}^\omega
+\sum_n c^{\mathrm{spill}} s_{n,t}^\omega\Bigr).
\end{aligned}
\]

Cost names: sea in/out, inland **repositioning** (`repo`), holding, leasing, spill.
Request `tslp-stage1-sea`, `tslp-stage2-inland`, and `tslp-data-access` for
constraints and JSON field mapping.
