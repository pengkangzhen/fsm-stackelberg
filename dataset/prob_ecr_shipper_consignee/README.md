# ECR Instance Generator

生成器用于从基座数据自动生成多组完整的 ECR 优化算例，支持场景化命名和批量生成。

## 目录结构

```
dataset/prob_ecr_shipper_consignee/
├── base/                    # 基座数据（长期不变）
│   ├── seaports.json        # 海港坐标
│   ├── dryports.json        # 陆港坐标
│   ├── candidate_nodes.json # 候选发货人/收货人池
│   └── costs_base.json      # 基准成本参数
├── config/                  # 配置文件
│   └── scenarios.yaml       # 预定义场景配置
└── instances/               # 生成的完整算例
    ├── small_5-3_5/
    ├── medium_10-6_10/
    ├── large_15-10_15/
    ├── high_cost_5-3_5/
    └── low_demand_5-3_5/
```

## 快速开始

### 批量生成预定义场景（5 个算例）

```bash
./scripts/generate_instances.sh
```

或手动运行：

```bash
PYTHONPATH=src poetry run python -m generator.cli batch \
    --config dataset/prob_ecr_shipper_consignee/config/scenarios.yaml \
    --output-dir dataset/prob_ecr_shipper_consignee/instances
```

### 生成单个算例

```bash
# 小规模算例
PYTHONPATH=src poetry run python -m generator.cli generate \
    --scale small --periods 5 \
    --output dataset/prob_ecr_shipper_consignee/instances/small_test

# 中规模算例
PYTHONPATH=src poetry run python -m generator.cli generate \
    --scale medium --periods 10 \
    --output dataset/prob_ecr_shipper_consignee/instances/medium_test

# 大规模算例
PYTHONPATH=src poetry run python -m generator.cli generate \
    --scale large --periods 15 \
    --output dataset/prob_ecr_shipper_consignee/instances/large_test
```

### 自定义参数生成

```bash
# 指定收货人/发货人数量
PYTHONPATH=src poetry run python -m generator.cli from-config \
    --consignees 8 --shippers 5 --periods 10 \
    --output dataset/prob_ecr_shipper_consignee/instances/custom_8-5_10
```

### 添加扰动

```bash
# 高运输成本场景（成本增加 50%）
PYTHONPATH=src poetry run python -m generator.cli generate \
    --scale small --periods 5 \
    --cost-perturbation 1.5 \
    --output dataset/prob_ecr_shipper_consignee/instances/high_cost_test

# 低需求场景（需求减少 30%）
PYTHONPATH=src poetry run python -m generator.cli generate \
    --scale small --periods 5 \
    --demand-perturbation 0.7 \
    --output dataset/prob_ecr_shipper_consignee/instances/low_demand_test
```

## 验证算例

```bash
PYTHONPATH=src poetry run python scripts/validate_instances.py \
    dataset/prob_ecr_shipper_consignee/instances/*
```

当前校验脚本按生成器的**现行输出格式**检查：
- records 格式：`supply`、`demand`、`distance_matrix`、`allowed_transport`、`arc_set`、`transit_time_matrix`、`transport_cost`、`hinterland_*`
- named dict 格式：`holding_cost`、`renting_cost`、`storage_capacity`、`initial_inventory`
- sets 格式：统一放在 `sample.json["sets"]`

## 算例命名规范

格式：`{scale}_{consignees}-{shippers}_{periods}_{variant}`

| 部分 | 说明 | 示例 |
|-----|------|------|
| `scale` | 规模等级 | `small`/`medium`/`large` |
| `consignees` | 收货人数量 | `5`, `10`, `15` |
| `shippers` | 发货人数量 | `3`, `6`, `10` |
| `periods` | 周期数 | `5`, `10`, `15` |
| `variant` | 变体标识（可选） | `high_cost`, `low_demand` |

## 预定义场景

配置文件 `config/scenarios.yaml` 定义了 5 个预定义场景：

1. **small_5-3_5**: 小规模基准算例（5 收货人 +3 发货人，5 周期）
2. **medium_10-6_10**: 中规模算例（10 收货人 +6 发货人，10 周期）
3. **large_15-10_15**: 大规模算例（15 收货人 +10 发货人，15 周期）
4. **high_cost_5-3_5**: 高运输成本场景（基于小规模，成本×1.5）
5. **low_demand_5-3_5**: 低需求场景（基于小规模，需求×0.7）

## 生成的算例结构

每个生成的算例包含以下文件：

```
instances/<name>/
├── sample.json      # 完整算例数据（含派生字段）
└── metadata.json    # 生成元信息
```

### sample.json 包含：

- `sets`: 所有集合定义，包含 `periods`、`transport_modes`、`seaports`、`dryports`、`shippers`、`consignees`、`storage_nodes`、`supply_nodes`、`demand_nodes`、`all_nodes`
- `nodes`: 扁平节点字典，键为节点名，值包含 `type`、`location` 等信息
- `supply`: records 格式，元素形如 `{"node": ..., "period": ..., "value": ...}`
- `demand`: records 格式，元素形如 `{"node": ..., "period": ..., "value": ...}`
- `holding_cost`: named dict，键为 storage node
- `renting_cost`: named dict，键为 storage node
- `storage_capacity`: named dict，键为 storage node
- `initial_inventory`: named dict，键为 storage node
- `unit_transport_cost`: 按 `transport_modes` 顺序排列的列表
- `distance_matrix`: records 格式，元素形如 `{"from": ..., "to": ..., "distance": ...}`
- `hinterland_consignee`: records 格式，元素形如 `{"dryport": ..., "consignee": ...}`
- `hinterland_shipper`: records 格式，元素形如 `{"dryport": ..., "shipper": ...}`
- `allowed_transport`: records 格式，元素形如 `{"from": ..., "to": ..., "mode": ...}`
- `arc_set`: records 格式，元素形如 `{"from": ..., "to": ...}`
- `transit_time_matrix`: records 格式，元素形如 `{"from": ..., "to": ..., "mode": ..., "time": ...}`
- `transport_cost`: records 格式，元素形如 `{"from": ..., "to": ..., "mode": ..., "cost": ...}`

说明：
- `transit_time_matrix` 与 `allowed_transport` 使用同一套弧集；零时滞弧也会显式写出。
- `transport_cost` 与 `transit_time_matrix` 都是稀疏 records，只存有效弧。

## 修改预定义场景

编辑 `config/scenarios.yaml` 文件添加或修改场景配置。

## 开发者信息

生成器模块位于 `src/generator/`：

- `data_loader.py`: 基座数据加载
- `derived_data.py`: 派生数据计算（距离、邻接矩阵等）
- `scenario_generator.py`: 场景生成逻辑
- `cli.py`: 命令行接口
