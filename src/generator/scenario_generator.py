"""Scenario generator for ECR instance generation.

Generates complete optimization instances from base data with configurable
perturbations for different scenarios.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .data_loader import BaseDataLoader
from .derived_data import DerivedDataCalculator


@dataclass
class InstanceConfig:
    """Configuration for a single instance generation.

    Attributes:
        name: Instance name.
        consignees_per_dryport: Number of consignees to select per dry port.
        shippers_per_dryport: Number of shippers to select per dry port.
        periods: Number of planning periods.
        cost_perturbation: Multiplier for transport costs (default 1.0).
        demand_perturbation: Multiplier for demand/supply (default 1.0).
        holding_cost_perturbation: Multiplier for holding costs (default 1.0).
        renting_cost_perturbation: Multiplier for renting costs (default 1.0).
        initial_inventory_ratio: Ratio of initial inventory relative to storage capacity (default 0.3).
        seed: Random seed for reproducibility.
    """

    name: str
    consignees_per_dryport: Dict[str, int] = field(default_factory=dict)
    shippers_per_dryport: Dict[str, int] = field(default_factory=dict)
    periods: int = 5
    cost_perturbation: float = 1.0
    demand_perturbation: float = 1.0
    holding_cost_perturbation: float = 1.0
    renting_cost_perturbation: float = 1.0
    initial_inventory_ratio: float = 0.3
    seed: Optional[int] = None


class ScenarioGenerator:
    """Generate ECR optimization instances from base data."""

    # Base supply/demand values for dry ports (per period averages)
    BASE_SUPPLY_DEMAND = {
        "Dalian Port": {"supply": 150, "demand": 180},
        "Yingkou Port": {"supply": 160, "demand": 160},
        "Dandong Port": {"supply": 160, "demand": 200},
        "Shenyang": {"supply": 200, "demand": 90},
        "Anshan": {"supply": 200, "demand": 150},
        "Changchun": {"supply": 180, "demand": 110},
        "Tonghua": {"supply": 180, "demand": 140},
        "Harbin": {"supply": 200, "demand": 130},
    }

    # Base supply/demand for shippers/consignees (per period)
    BASE_SHIPPER_SUPPLY = 12
    BASE_CONSIGNEE_DEMAND = 12

    def __init__(self, base_dir: str | Path):
        """Initialize the scenario generator.

        Args:
            base_dir: Directory containing base data files.
        """
        self.base_dir = Path(base_dir)
        self.data_loader = BaseDataLoader(base_dir)

        # Load base data
        self.seaports = self.data_loader.load_seaports()
        self.dryports = self.data_loader.load_dryports()
        self.candidate_nodes = self.data_loader.load_candidate_nodes()
        self.costs_base = self.data_loader.load_base_costs()

        self.seaport_names = [p["name"] for p in self.seaports]
        self.dryport_names = [p["name"] for p in self.dryports]

    def _select_nodes(
        self,
        candidates: List[Dict],
        num_to_select: int,
        rng: random.Random,
    ) -> List[Dict]:
        """Select a subset of nodes from candidates.

        Args:
            candidates: List of candidate nodes.
            num_to_select: Number of nodes to select.
            rng: Random number generator.

        Returns:
            Selected list of nodes.
        """
        if num_to_select >= len(candidates):
            return candidates.copy()

        return rng.sample(candidates, num_to_select)

    def _generate_supply_demand(
        self,
        selected_shippers: List[Dict],
        selected_consignees: List[Dict],
        periods: int,
        demand_perturbation: float,
        rng: random.Random,
    ) -> tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        """Generate supply and demand data for all nodes.

        Args:
            selected_shippers: List of selected shippers.
            selected_consignees: List of selected consignees.
            periods: Number of periods.
            demand_perturbation: Demand perturbation factor.
            rng: Random number generator.

        Returns:
            Tuple of (supply dict, demand dict).
        """
        supply = {}  # {node_name: [val_t1, val_t2, ...]}
        demand = {}  # {node_name: [val_t1, val_t2, ...]}

        # Generate for seaports and dryports
        all_ports = self.seaport_names + self.dryport_names
        for node_name in all_ports:
            base = self.BASE_SUPPLY_DEMAND.get(node_name, {"supply": 150, "demand": 150})
            supply[node_name] = []
            demand[node_name] = []

            for t in range(1, periods + 1):
                period_factor_supply = 0.8 + 0.4 * rng.random()  # [0.8, 1.2]
                period_factor_demand = 0.8 + 0.4 * rng.random()  # [0.8, 1.2]

                supply[node_name].append(int(
                    base["supply"] * period_factor_supply * demand_perturbation
                ))
                demand[node_name].append(int(
                    base["demand"] * period_factor_demand * demand_perturbation
                ))

        # Generate for selected shippers (demand only - shippers need empty containers for export)
        for shipper in selected_shippers:
            demand[shipper["name"]] = []
            for t in range(1, periods + 1):
                period_factor = 0.8 + 0.4 * rng.random()
                demand[shipper["name"]].append(int(
                    self.BASE_SHIPPER_SUPPLY * period_factor * demand_perturbation
                ))

        # Generate for selected consignees (supply only - consignees produce empty containers from import unloading)
        for consignee in selected_consignees:
            supply[consignee["name"]] = []
            for t in range(1, periods + 1):
                period_factor = 0.8 + 0.4 * rng.random()
                supply[consignee["name"]].append(int(
                    self.BASE_CONSIGNEE_DEMAND * period_factor * demand_perturbation
                ))

        return supply, demand

    def _generate_costs(
        self,
        cost_perturbation: float,
        holding_cost_perturbation: float,
        renting_cost_perturbation: float,
        rng: random.Random,
    ) -> tuple[Dict, Dict, Dict]:
        """Generate cost parameters with perturbation.

        Args:
            cost_perturbation: Transport cost perturbation factor.
            holding_cost_perturbation: Holding cost perturbation factor.
            renting_cost_perturbation: Renting cost perturbation factor.
            rng: Random number generator.

        Returns:
            Tuple of (unit_transport_cost, holding_cost, renting_cost).
        """
        # Unit transport costs with perturbation
        unit_transport_cost = {}
        for mode, base_cost in self.costs_base["unit_transport_cost"].items():
            mode_perturbation = cost_perturbation * (0.9 + 0.2 * rng.random())
            unit_transport_cost[mode] = round(base_cost * mode_perturbation, 2)

        # Holding costs
        holding_cost = {}
        for port in self.seaports:
            base = self.costs_base["holding_cost_base"]["seaport"]
            perturbation = holding_cost_perturbation * (0.9 + 0.2 * rng.random())
            holding_cost[port["name"]] = int(base * perturbation)

        for port in self.dryports:
            base = self.costs_base["holding_cost_base"]["dryport"]
            perturbation = holding_cost_perturbation * (0.9 + 0.2 * rng.random())
            holding_cost[port["name"]] = int(base * perturbation)

        # Renting costs
        renting_cost = {}
        for port in self.seaports:
            base = self.costs_base["renting_cost_base"]["seaport"].get(
                port["name"], 2000
            )
            perturbation = renting_cost_perturbation * (0.9 + 0.2 * rng.random())
            renting_cost[port["name"]] = int(base * perturbation)

        for port in self.dryports:
            base = self.costs_base["renting_cost_base"]["dryport"].get(
                port["name"], 4000
            )
            perturbation = renting_cost_perturbation * (0.9 + 0.2 * rng.random())
            renting_cost[port["name"]] = int(base * perturbation)

        return unit_transport_cost, holding_cost, renting_cost

    def _generate_storage_capacity(
        self,
        nodes_flat: Dict[str, Dict],
        rng: random.Random,
    ) -> Dict[str, int]:
        """Generate storage capacity for seaports and dryports.

        Args:
            nodes_flat: Flat nodes dict {name: {"type": ..., ...}}.
            rng: Random number generator.

        Returns:
            {node_name: capacity} for seaports and dryports.
        """
        storage_capacity_base = self.costs_base.get(
            "storage_capacity_base", {"seaport": 5000, "dryport": 3000}
        )
        result = {}
        for name, info in nodes_flat.items():
            ntype = info["type"]
            if ntype in storage_capacity_base:
                base = storage_capacity_base[ntype]
                perturbation = 0.9 + 0.2 * rng.random()  # [0.9, 1.1]
                result[name] = int(base * perturbation)
        return result

    def _generate_initial_inventory(
        self,
        storage_capacity: Dict[str, int],
        supply: Dict[str, List[int]],
        demand: Dict[str, List[int]],
        initial_inventory_ratio: float,
        rng: random.Random,
    ) -> Dict[str, int]:
        """Generate initial inventory for seaports and dryports.

        The initial inventory is calculated based on storage capacity with
        consideration for first period supply/demand balance to ensure feasibility.

        Formula: initial_inventory[i] = storage_capacity[i] × ratio × variation_factor
        where variation_factor ∈ [0.8, 1.2]

        Args:
            storage_capacity: Storage capacity dict {node_name: capacity}.
            supply: Supply dict {node_name: [val_t1, val_t2, ...]}.
            demand: Demand dict {node_name: [val_t1, val_t2, ...]}.
            initial_inventory_ratio: Base ratio of storage capacity (default 0.3).
            rng: Random number generator.

        Returns:
            {node_name: initial_inventory} for seaports and dryports.
        """
        result = {}

        for node_name, capacity in storage_capacity.items():
            # Base inventory: percentage of storage capacity
            base_inventory = int(capacity * initial_inventory_ratio)

            # Random variation: ±20% of base
            variation_factor = 0.8 + 0.4 * rng.random()  # [0.8, 1.2]
            adjusted_inventory = int(base_inventory * variation_factor)

            # Consider first period net supply/demand balance for feasibility
            first_period_supply = supply.get(node_name, [0])[0] if node_name in supply else 0
            first_period_demand = demand.get(node_name, [0])[0] if node_name in demand else 0
            net_flow = first_period_supply - first_period_demand

            # If net outflow (demand > supply), increase initial inventory
            if net_flow < 0:
                feasibility_adjustment = min(abs(net_flow), int(capacity * 0.1))
                adjusted_inventory += feasibility_adjustment

            # Ensure within bounds [0, capacity]
            result[node_name] = max(0, min(adjusted_inventory, capacity))

        return result

    @staticmethod
    def _build_flat_nodes(
        seaports: List[Dict],
        dryports: List[Dict],
        shippers: List[Dict],
        consignees: List[Dict],
    ) -> Dict[str, Dict]:
        """Build a flat nodes dict from grouped node lists.

        Returns:
            {node_name: {"type": ..., "name_zh": ..., "location": ..., ...}}
        """
        nodes_flat = {}
        for node in seaports:
            nodes_flat[node["name"]] = {"type": "seaport", **{k: v for k, v in node.items() if k != "name"}}
        for node in dryports:
            nodes_flat[node["name"]] = {"type": "dryport", **{k: v for k, v in node.items() if k != "name"}}
        for node in shippers:
            nodes_flat[node["name"]] = {"type": "shipper", **{k: v for k, v in node.items() if k != "name"}}
        for node in consignees:
            nodes_flat[node["name"]] = {"type": "consignee", **{k: v for k, v in node.items() if k != "name"}}
        return nodes_flat

    def generate_instance(self, config: InstanceConfig) -> Dict[str, Any]:
        """Generate a complete optimization instance.

        Args:
            config: Instance configuration.

        Returns:
            Complete instance dictionary.
        """
        # Initialize random generator
        rng = random.Random(config.seed)

        # Select shippers and consignees per dry port, building hinterland map
        selected_shippers = []
        selected_consignees = []
        hinterland_map: Dict[str, Dict[str, List]] = {}

        for dryport_name, num_shippers in config.shippers_per_dryport.items():
            candidates = self.candidate_nodes["shippers"].get(dryport_name, [])
            selected = self._select_nodes(candidates, num_shippers, rng)
            selected_shippers.extend(selected)
            hinterland_map.setdefault(dryport_name, {"shippers": [], "consignees": []})
            hinterland_map[dryport_name]["shippers"] = [s["name"] for s in selected]

        for dryport_name, num_consignees in config.consignees_per_dryport.items():
            candidates = self.candidate_nodes["consignees"].get(dryport_name, [])
            selected = self._select_nodes(candidates, num_consignees, rng)
            selected_consignees.extend(selected)
            hinterland_map.setdefault(dryport_name, {"shippers": [], "consignees": []})
            hinterland_map[dryport_name]["consignees"] = [c["name"] for c in selected]

        # Generate supply and demand
        supply, demand = self._generate_supply_demand(
            selected_shippers,
            selected_consignees,
            config.periods,
            config.demand_perturbation,
            rng,
        )

        # Generate costs
        (
            unit_transport_cost,
            holding_cost,
            renting_cost,
        ) = self._generate_costs(
            config.cost_perturbation,
            config.holding_cost_perturbation,
            config.renting_cost_perturbation,
            rng,
        )

        # Calculate derived data
        calculator = DerivedDataCalculator(
            seaports=self.seaports,
            dryports=self.dryports,
            shippers=selected_shippers,
            consignees=selected_consignees,
            costs_base=self.costs_base,
            hinterland_map=hinterland_map,
        )

        derived_data = calculator.compute_all(config.periods)

        # Build flat nodes dict
        nodes_flat = self._build_flat_nodes(
            self.seaports, self.dryports, selected_shippers, selected_consignees,
        )

        # Generate storage capacity
        storage_capacity = self._generate_storage_capacity(nodes_flat, rng)

        # Generate initial inventory
        initial_inventory = self._generate_initial_inventory(
            storage_capacity,
            supply,
            demand,
            config.initial_inventory_ratio,
            rng,
        )

        # Build ordered node lists for sets
        seaport_names = [p["name"] for p in self.seaports]
        dryport_names = [p["name"] for p in self.dryports]
        shipper_names = [s["name"] for s in selected_shippers]
        consignee_names = [c["name"] for c in selected_consignees]

        # storage_nodes = seaports + dryports
        storage_node_names = seaport_names + dryport_names

        # supply_nodes = seaports + dryports + consignees
        supply_node_names = seaport_names + dryport_names + consignee_names

        # demand_nodes = seaports + dryports + shippers
        demand_node_names = seaport_names + dryport_names + shipper_names

        # all_nodes = seaports + dryports + shippers + consignees
        all_node_names = seaport_names + dryport_names + shipper_names + consignee_names

        # Convert supply/demand to records format
        supply_records = [
            {"node": name, "period": t, "value": val}
            for name in supply_node_names
            for t, val in enumerate(supply[name], 1)
        ]
        demand_records = [
            {"node": name, "period": t, "value": val}
            for name in demand_node_names
            for t, val in enumerate(demand[name], 1)
        ]

        # Convert holding_cost, renting_cost, storage_capacity to named dict format (LLM-friendly)
        holding_cost_dict = {name: holding_cost[name] for name in storage_node_names}
        renting_cost_dict = {name: renting_cost[name] for name in storage_node_names}
        storage_capacity_dict = {name: storage_capacity[name] for name in storage_node_names}
        initial_inventory_dict = {name: initial_inventory[name] for name in storage_node_names}

        # Convert unit_transport_cost to list format
        transport_mode_names = list(self.costs_base["unit_transport_cost"].keys())
        unit_transport_cost_list = [unit_transport_cost[mode] for mode in transport_mode_names]

        # Convert distance_matrix to records format
        # Also keep matrix for transport cost calculation
        distance_matrix_list = [
            [derived_data["distance_matrix"][from_node].get(to_node, 0.0) for to_node in all_node_names]
            for from_node in all_node_names
        ]
        distance_records = [
            {"from": from_node, "to": to_node, "distance": distance_matrix_list[i][j]}
            for i, from_node in enumerate(all_node_names)
            for j, to_node in enumerate(all_node_names)
            if distance_matrix_list[i][j] > 0  # Only include non-zero distances
        ]

        # Convert hinterland_coverage to records format
        # hinterland_consignee: dryport × consignee (only include pairs where consignee is in hinterland)
        # hinterland_shipper: dryport × shipper (only include pairs where shipper is in hinterland)
        hinterland = derived_data["hinterland_coverage"]
        hinterland_consignee_records = [
            {"dryport": dryport, "consignee": consignee}
            for dryport in dryport_names
            for consignee in consignee_names
            if consignee in hinterland.get(dryport, {}).get("consignees", [])
        ]
        hinterland_shipper_records = [
            {"dryport": dryport, "shipper": shipper}
            for dryport in dryport_names
            for shipper in shipper_names
            if shipper in hinterland.get(dryport, {}).get("shippers", [])
        ]

        # Calculate allowed_transport and transit_time matrices
        allowed_transport_matrix = calculator.calc_allowed_transport_matrix(
            all_node_names, nodes_flat, transport_mode_names
        )
        transit_time_matrix = calculator.calc_transit_time_matrix(
            all_node_names, nodes_flat, transport_mode_names
        )

        # Calculate transport cost matrix: distance * unit_cost
        transport_cost_matrix = calculator.calc_transport_cost_matrix(
            all_node_names,
            transport_mode_names,
            distance_matrix_list,
            unit_transport_cost_list,
        )

        # Calculate arc_set matrix from allowed_transport
        arc_set_matrix = calculator.calc_arc_set_matrix(allowed_transport_matrix)

        # Convert 3D matrices to records format
        allowed_transport_records = [
            {"from": all_node_names[i], "to": all_node_names[j], "mode": transport_mode_names[k]}
            for i in range(len(all_node_names))
            for j in range(len(all_node_names))
            for k in range(len(transport_mode_names))
            if allowed_transport_matrix[i][j][k] == 1
        ]

        arc_set_records = [
            {"from": all_node_names[i], "to": all_node_names[j]}
            for i in range(len(all_node_names))
            for j in range(len(all_node_names))
            if arc_set_matrix[i][j] == 1
        ]

        transit_time_records = [
            {"from": all_node_names[i], "to": all_node_names[j], "mode": transport_mode_names[k], "time": transit_time_matrix[i][j][k]}
            for i in range(len(all_node_names))
            for j in range(len(all_node_names))
            for k in range(len(transport_mode_names))
            if allowed_transport_matrix[i][j][k] == 1
        ]

        transport_cost_records = [
            {"from": all_node_names[i], "to": all_node_names[j], "mode": transport_mode_names[k], "cost": transport_cost_matrix[i][j][k]}
            for i in range(len(all_node_names))
            for j in range(len(all_node_names))
            for k in range(len(transport_mode_names))
            if allowed_transport_matrix[i][j][k] == 1
        ]

        # Build complete instance (without _meta - it's stored externally)
        instance = {
            "sets": {
                "periods": list(range(1, config.periods + 1)),
                "transport_modes": transport_mode_names,
                "node_types": ["seaport", "dryport", "shipper", "consignee"],
                "seaports": seaport_names,
                "dryports": dryport_names,
                "shippers": shipper_names,
                "consignees": consignee_names,
                # storage_nodes removed: derivable as seaports + dryports
                # supply_nodes/demand_nodes removed: derivable from supply/demand arrays
                "all_nodes": all_node_names,
            },
            "nodes": nodes_flat,
            # Records format (LLM-friendly)
            "supply": supply_records,
            "demand": demand_records,
            # Named dict format (1D parameters, LLM-friendly)
            "holding_cost": holding_cost_dict,
            "renting_cost": renting_cost_dict,
            "storage_capacity": storage_capacity_dict,
            "initial_inventory": initial_inventory_dict,
            "unit_transport_cost": unit_transport_cost_list,  # Keep as list (ordered by transport_modes)
            # Records format (sparse data)
            "distance_matrix": distance_records,
            "hinterland_consignee": hinterland_consignee_records,
            "hinterland_shipper": hinterland_shipper_records,
            "allowed_transport": allowed_transport_records,
            "arc_set": arc_set_records,
            "transit_time_matrix": transit_time_records,
            "transport_cost": transport_cost_records,
        }

        return instance

    def generate_and_save(
        self,
        config: InstanceConfig,
        output_dir: str | Path,
        include_metadata: bool = True,
    ) -> Path:
        """Generate an instance and save it to disk.

        Args:
            config: Instance configuration.
            output_dir: Output directory for the instance.
            include_metadata: Whether to include metadata file.

        Returns:
            Path to the generated instance directory.
        """
        output_path = Path(output_dir) / config.name
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate instance
        instance = self.generate_instance(config)

        # Save main instance file
        instance_file = output_path / "sample.json"
        with open(instance_file, "w", encoding="utf-8") as f:
            json.dump(instance, f, indent=4, ensure_ascii=False)

        # Save metadata
        if include_metadata:
            nodes = instance["nodes"]
            metadata = {
                "instance_name": config.name,
                "scale": self._infer_scale(config),
                "num_shippers": sum(1 for v in nodes.values() if v["type"] == "shipper"),
                "num_consignees": sum(1 for v in nodes.values() if v["type"] == "consignee"),
                "periods": config.periods,
                "perturbations": {
                    "cost": config.cost_perturbation,
                    "demand": config.demand_perturbation,
                    "holding_cost": config.holding_cost_perturbation,
                    "renting_cost": config.renting_cost_perturbation,
                },
                "seed": config.seed,
            }
            metadata_file = output_path / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_path

    def _infer_scale(self, config: InstanceConfig) -> str:
        """Infer scale label from configuration.

        Args:
            config: Instance configuration.

        Returns:
            Scale label: 'small', 'medium', or 'large'.
        """
        total_consignees = sum(config.consignees_per_dryport.values())
        total_shippers = sum(config.shippers_per_dryport.values())

        if total_consignees <= 5 and total_shippers <= 3:
            return "small"
        elif total_consignees <= 10 and total_shippers <= 6:
            return "medium"
        else:
            return "large"


def generate_from_yaml_config(
    config_path: str | Path, base_dir: str | Path, output_dir: str | Path
) -> List[Path]:
    """Generate instances from a YAML configuration file.

    Args:
        config_path: Path to YAML config file.
        base_dir: Directory containing base data.
        output_dir: Output directory for instances.

    Returns:
        List of paths to generated instance directories.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML config support. Install with: poetry add pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generator = ScenarioGenerator(base_dir)
    output_paths = []

    scenarios = config.get("scenarios", {})

    for name, scenario_config in scenarios.items():
        # Check for base scenario to inherit from
        if "base" in scenario_config:
            base_name = scenario_config["base"]
            if base_name not in scenarios:
                raise ValueError(f"Base scenario '{base_name}' not found in config")

            base_config = scenarios[base_name]
            # Merge configurations
            merged_config = {**base_config, **scenario_config}
        else:
            merged_config = scenario_config

        # Create InstanceConfig
        instance_config = InstanceConfig(
            name=name,
            consignees_per_dryport=merged_config.get("consignees_per_dryport", {}),
            shippers_per_dryport=merged_config.get("shippers_per_dryport", {}),
            periods=merged_config.get("periods", 5),
            cost_perturbation=merged_config.get("cost_perturbation", 1.0),
            demand_perturbation=merged_config.get("demand_perturbation", 1.0),
            holding_cost_perturbation=merged_config.get("holding_cost_perturbation", 1.0),
            renting_cost_perturbation=merged_config.get("renting_cost_perturbation", 1.0),
            seed=merged_config.get("seed"),
        )

        # Generate and save
        output_path = generator.generate_and_save(instance_config, output_dir)
        output_paths.append(output_path)
        print(f"Generated instance: {output_path}")

    return output_paths
