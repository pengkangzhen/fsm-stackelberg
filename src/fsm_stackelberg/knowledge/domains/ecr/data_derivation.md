---
name: ecr-data-derivation
description: ECR 数据推导规则——从基础数据（坐标、节点类型）推导距离矩阵、可用运输弧、运输成本、运输时间、腹地归属和 Street-Turn，含可直接调用的 Python 函数
---

# ECR 数据推导规则

原始数据仅提供节点坐标、供需量、成本系数等**基础信息**。以下参数需要从基础数据推导得出。

数据中 `nodes` 字段存储了每个节点的经纬度坐标和类型：
```
nodes = {"Dalian Port": {"type": "seaport", "location": {"lng": 121.63, "lat": 38.91}}, ...}
```

## 必须推导的参数

以下参数在基础数据中**不存在**，必须在 `optimize()` 函数内推导：

| 参数 | 含义 |
|------|------|
| `distance_matrix` | 节点间球面距离 (km) |
| `allowed_transport` | 哪些 (from, to, mode) 组合可行 |
| `transport_cost` | 每条可行弧的运输成本 = distance × unit_cost |
| `transit_time` | 每条可行弧的运输周期数 |
| `hinterland_consignee` / `hinterland_shipper` | 每个收发货人归属哪个陆港 |
| `street_turn` | 收货人→发货人直接转运的可行性 |

## 完整推导代码

将以下函数复制到 `optimize()` 内部，然后用推导出的参数构建 Gurobi 模型。

```python
import math

# ---------------------------------------------------------------------------
# 1. 球面距离 (Haversine 公式)
# ---------------------------------------------------------------------------
def haversine(lng1, lat1, lng2, lat2):
    """计算两个经纬度坐标之间的球面距离 (km)。"""
    R = 6371  # 地球半径 km
    lng1, lat1, lng2, lat2 = map(math.radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# 2. 从基础数据推导全部参数
# ---------------------------------------------------------------------------
def derive_all(data):
    """从基础数据 (节点坐标 + 供需 + 成本) 推导出全部优化所需参数。

    返回一个 dict，包含: distance, allowed_arcs, transport_cost,
    transit_time, hinterland_consignee, hinterland_shipper, street_turn。
    """
    nodes = data["nodes"]          # {name: {"type": ..., "location": {"lng", "lat"}}}
    all_nodes = data["all_nodes"]
    seaports = data["seaports"]
    dryports = data["dryports"]
    modes = data["transport_modes"]

    # 节点类型查找表
    node_type = {n: nodes[n]["type"] for n in all_nodes}

    # --- 2a. 距离矩阵 ---
    distance = {}
    for ni in all_nodes:
        for nj in all_nodes:
            if ni == nj:
                distance[(ni, nj)] = 0.0
            else:
                li, lj = nodes[ni]["location"], nodes[nj]["location"]
                distance[(ni, nj)] = haversine(li["lng"], li["lat"], lj["lng"], lj["lat"])

    # --- 2b. 运输方式规则: (from_type, to_type) -> 可用 mode 列表 ---
    mode_rules = {
        ("seaport", "seaport"): ["barge"],
        ("seaport", "dryport"): ["rail", "truck"],
        ("seaport", "shipper"): ["truck"],
        ("dryport", "seaport"): ["rail", "truck"],
        ("dryport", "dryport"): ["rail", "truck"],
        ("dryport", "shipper"): ["truck"],
        ("consignee", "seaport"): ["truck"],
        ("consignee", "dryport"): ["truck"],
        ("consignee", "shipper"): ["truck"],
    }

    # --- 2c. 腹地归属: 每个收发货人归属最近的陆港 ---
    hinterland_consignee = []   # [(dryport, consignee), ...]
    hinterland_shipper = []
    hinterland_of = {}          # {node: dryport}
    for node in all_nodes:
        if node_type[node] in ("consignee", "shipper"):
            best_dry = min(dryports,
                           key=lambda d: distance[(node, d)])
            hinterland_of[node] = best_dry
            if node_type[node] == "consignee":
                hinterland_consignee.append((best_dry, node))
            else:
                hinterland_shipper.append((best_dry, node))

    # --- 2d. Street-Turn: 同腹地的 consignee -> shipper 可行 ---
    street_turn = {}            # {(consignee, shipper): 0/1}
    consignees = [n for n in all_nodes if node_type[n] == "consignee"]
    shippers = [n for n in all_nodes if node_type[n] == "shipper"]
    for c in consignees:
        for s in shippers:
            street_turn[(c, s)] = 1 if hinterland_of[c] == hinterland_of[s] else 0

    # --- 2e. 可用弧 + 成本 + 运输时间 ---
    # unit_transport_cost: data["unit_transport_cost"] 列表，按 transport_modes 顺序
    utc = data["unit_transport_cost"]
    unit_cost = {modes[i]: utc[i] for i in range(len(modes))}

    # 运输速度 (km per period): truck=500, rail=300, barge=200
    speed = {"truck": 500, "rail": 300, "barge": 200}

    allowed_arcs = []           # [(from, to, mode), ...]
    transport_cost = {}         # {(from, to, mode): cost}
    transit_time = {}           # {(from, to, mode): periods}

    for ni in all_nodes:
        for nj in all_nodes:
            if ni == nj:
                continue  # 自环排除
            ti, tj = node_type[ni], node_type[nj]
            allowed_modes = mode_rules.get((ti, tj), [])
            for mode in allowed_modes:
                # Street-Turn 特殊过滤: consignee->shipper 仅限同腹地
                if (ti, tj) == ("consignee", "shipper") and street_turn[(ni, nj)] == 0:
                    continue
                allowed_arcs.append((ni, nj, mode))
                transport_cost[(ni, nj, mode)] = round(
                    distance[(ni, nj)] * unit_cost[mode], 2
                )
                # 运输时间: 海港间驳船固定 1 周期; 其余按距离/速度计算, 最小 0
                if ti == "seaport" and tj == "seaport" and mode == "barge":
                    transit_time[(ni, nj, mode)] = 1
                else:
                    transit_time[(ni, nj, mode)] = int(
                        distance[(ni, nj)] // speed[mode]
                    )

    return {
        "distance": distance,
        "allowed_arcs": allowed_arcs,
        "transport_cost": transport_cost,
        "transit_time": transit_time,
        "hinterland_consignee": hinterland_consignee,
        "hinterland_shipper": hinterland_shipper,
        "street_turn": street_turn,
        "hinterland_of": hinterland_of,
    }


# ---------------------------------------------------------------------------
# 3. 在 optimize() 中使用
# ---------------------------------------------------------------------------
# def optimize(data):
#     derived = derive_all(data)
#     allowed_arcs = derived["allowed_arcs"]
#     transport_cost = derived["transport_cost"]
#     transit_time = derived["transit_time"]
#     # ... 然后用这些参数定义 Gurobi 变量和约束 ...
```

## 使用要点

1. **第一步**：在 `optimize()` 开头调用 `derived = derive_all(data)` 获取推导参数
2. **变量定义**：运输变量 `x[i,j,k,t]` 的索引集 `(i,j,k)` 应取自 `derived["allowed_arcs"]`，不要自行枚举所有节点对×模式组合
3. **运输成本**：`transport_cost[(i,j,k)]` 已含距离×单位成本，直接用于目标函数
4. **运输时间**：`transit_time[(i,j,k)]` 表示从发出到到达需经过的周期数（0=同期到达）
5. **库存平衡**：仅 `seaports + dryports` 有库存变量；consignee 用流出等式，shipper 用流入等式
6. **Street-Turn**：`derived["allowed_arcs"]` 已过滤掉不同腹地的 consignee→shipper 弧，直接用即可
