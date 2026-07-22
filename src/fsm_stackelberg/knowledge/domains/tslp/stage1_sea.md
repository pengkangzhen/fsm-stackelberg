---
name: tslp-stage1-sea
description: Hub sea first-stage decisions y_in / y_out, vessel capacities, export floors
---

# Stage-1 sea decisions [D1]

Hubs only. For each hub \(i\) and period \(t\):

- \(y_{i,t}^{\mathrm{in}}\le B_i^{\mathrm{in}} V_{i,t}\)
- \(y_{i,t}^{\mathrm{out}}\le B_i^{\mathrm{out}} V_{i,t}\)
- \(y_{i,t}^{\mathrm{out}}\ge D_{i,t}^{\mathrm{ext,eff}} V_{i,t}\)

Sea cost: \(c^{\mathrm{sea,in}} y^{\mathrm{in}} + c^{\mathrm{sea,out}} y^{\mathrm{out}}\).  
\(y^{\mathrm{in}}\) uses **arrival-week** accounting (enters inventory in week \(t\)).
