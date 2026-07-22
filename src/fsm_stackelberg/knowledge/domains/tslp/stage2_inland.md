---
name: tslp-stage2-inland
description: Scenario inland balance for hubs, spokes, dry ports; arcs, inventory, lease, spill
---

# Stage-2 inland recourse [D2]

Per scenario \(\omega\), minimize land + hold + lease + spill costs.

**Inventory balance (schematic):**

- **Hub:** \(I_t = I_{t-1}+\xi+r+y^{\mathrm{in}}-y^{\mathrm{out}}+\mathrm{inflow}-\mathrm{outflow}-\eta^\omega-s\)
- **Spoke:** same without \(y\), plus exogenous top-up \(E\)
- **Dry port:** inland only (no sea \(y\), no \(E\))

Inflows respect transit times \(\tau_{ij\kappa}\). Cap \(I\le U\), flows \(\le F\).  
Spill \(s\) with large \(c^{\mathrm{spill}}\) provides relatively complete recourse.
