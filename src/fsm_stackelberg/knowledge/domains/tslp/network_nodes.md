---
name: tslp-network-nodes
description: Hub / spoke seaports and dry ports; no shipper or consignee nodes
---

# Network nodes

Only three node classes:

| Set | Role |
|-----|------|
| `hubs` | Global hub seaports; stage-1 sea \(y\) allowed |
| `spokes` | Feeder seaports; inland + exogenous \(E\) |
| `dry_ports` | Inland dry ports |

There are **no** shipper/consignee street nodes. Hinterland links are encoded by `lambda_hl` and the arc list (road/rail).
