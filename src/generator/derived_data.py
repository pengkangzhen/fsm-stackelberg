"""Derived data calculator for ECR instance generator.

Computes distance matrices, transport modes, and other derived data from base
node coordinates and cost parameters.
"""

import math
from typing import Dict, List


def haversine_distance(
    lng1: float, lat1: float, lng2: float, lat2: float
) -> float:
    """Calculate the great-circle distance between two points using Haversine formula.

    Args:
        lng1, lat1: Longitude and latitude of point 1 in degrees.
        lng2, lat2: Longitude and latitude of point 2 in degrees.

    Returns:
        Distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers

    lng1_rad = math.radians(lng1)
    lng2_rad = math.radians(lng2)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlng = lng2_rad - lng1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class DerivedDataCalculator:
    """Calculate derived data for ECR optimization instances."""

    def __init__(
        self,
        seaports: List[Dict],
        dryports: List[Dict],
        shippers: List[Dict],
        consignees: List[Dict],
        costs_base: Dict,
        hinterland_map: Dict[str, Dict[str, List[str]]] | None = None,
    ):
        """Initialize the calculator.

        Args:
            seaports: List of seaport data.
            dryports: List of dry port data.
            shippers: List of shipper data.
            consignees: List of consignee data.
            costs_base: Base cost parameters.
            hinterland_map: Mapping from dryport name to selected shipper/consignee names.
                e.g. {"Shenyang": {"shippers": ["Shenyang_Sujiatun"], "consignees": ["Shenyang_Tiexi"]}}
        """
        self.seaports = seaports
        self.dryports = dryports
        self.shippers = shippers
        self.consignees = consignees
        self.costs_base = costs_base
        self.hinterland_map = hinterland_map or {}

        # Build ordered node list
        self.all_nodes = seaports + dryports + shippers + consignees
        self.seaport_names = [p["name"] for p in seaports]
        self.dryport_names = [p["name"] for p in dryports]
        self.shipper_names = [s["name"] for s in shippers]
        self.consignee_names = [c["name"] for c in consignees]

    def calc_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate the distance matrix between all nodes as a named nested dict.

        Returns:
            {node_name_i: {node_name_j: distance_km, ...}, ...}
        """
        result = {}
        for node_i in self.all_nodes:
            name_i = node_i["name"]
            loc_i = node_i["location"]
            row = {}
            for node_j in self.all_nodes:
                name_j = node_j["name"]
                if name_i == name_j:
                    row[name_j] = 0.0
                else:
                    loc_j = node_j["location"]
                    row[name_j] = haversine_distance(
                        loc_i["lng"], loc_i["lat"],
                        loc_j["lng"], loc_j["lat"],
                    )
            result[name_i] = row
        return result

    def calc_hinterland_coverage(self) -> Dict[str, Dict[str, List[str]]]:
        """Calculate hinterland coverage from the hinterland map.

        Returns:
            {dryport_name: {"shippers": [...], "consignees": [...]}, ...}
        """
        return self.hinterland_map

    def calc_transport_modes(self) -> List[Dict]:
        """Calculate transport mode rules by node type combination.

        Returns:
            List of rules, each with from_node_type, to_node_type, transport_mode.
        """
        mode_map = {
            ("seaport", "seaport"): ["barge"],
            ("seaport", "dryport"): ["rail", "truck"],
            ("seaport", "shipper"): ["truck"],
            ("dryport", "seaport"): ["rail", "truck"],
            ("dryport", "dryport"): ["rail", "truck"],
            ("dryport", "shipper"): ["truck"],
            # Street-Turn arcs are further filtered by shared dryport hinterland
            # in calc_allowed_transport_matrix().
            ("consignee", "shipper"): ["truck"],
            ("consignee", "seaport"): ["truck"],
            ("consignee", "dryport"): ["truck"],
        }

        rules = []
        for (from_type, to_type), modes in mode_map.items():
            rules.append({
                "from_node_type": from_type,
                "to_node_type": to_type,
                "transport_mode": modes,
            })
        return rules

    def _shared_hinterland_pairs(self) -> set[tuple[str, str]]:
        """Return the allowed Street-Turn consignee->shipper pairs.

        A Street-Turn move is allowed only when the consignee and shipper belong
        to the same dryport hinterland.
        """
        pairs: set[tuple[str, str]] = set()
        for coverage in self.hinterland_map.values():
            consignees = coverage.get("consignees", [])
            shippers = coverage.get("shippers", [])
            for consignee in consignees:
                for shipper in shippers:
                    pairs.add((consignee, shipper))
        return pairs

    def calc_transit_time(self) -> List[Dict]:
        """Calculate transit time rules by node type and transport mode.

        Only seaport-to-seaport via barge has non-zero transit time (1 period).
        All other combinations are immediate (transit_time = 0, not listed).

        Returns:
            List of rules with non-zero transit times.
        """
        return [
            {
                "from_node_type": "seaport",
                "to_node_type": "seaport",
                "transport_mode": "barge",
                "transit_time": 1,
            }
        ]

    def calc_allowed_transport_matrix(
        self,
        all_node_names: List[str],
        nodes_flat: Dict[str, Dict],
        transport_modes: List[str],
    ) -> List[List[List[int]]]:
        """Calculate allowed transport matrix for all node pairs.

        Args:
            all_node_names: Ordered list of all node names.
            nodes_flat: Dict mapping node name to node info (including 'type').
            transport_modes: Ordered list of transport mode names.

        Returns:
            3D binary matrix: allowed[from_idx][to_idx][mode_idx] = 0 or 1
        """
        # Build node type lookup
        node_types = {name: nodes_flat[name]["type"] for name in all_node_names}

        # Get transport mode rules
        mode_rules = self.calc_transport_modes()

        # Build lookup: (from_type, to_type) -> set of allowed modes
        allowed_modes = {}
        for rule in mode_rules:
            key = (rule["from_node_type"], rule["to_node_type"])
            allowed_modes[key] = set(rule["transport_mode"])

        # Build the 3D matrix
        n = len(all_node_names)
        k = len(transport_modes)
        mode_indices = {mode: idx for idx, mode in enumerate(transport_modes)}
        shared_hinterland_pairs = self._shared_hinterland_pairs()

        matrix = [
            [
                [0] * k  # Initialize all modes to 0
                for _ in range(n)
            ]
            for _ in range(n)
        ]

        for i, from_node in enumerate(all_node_names):
            from_type = node_types[from_node]
            for j, to_node in enumerate(all_node_names):
                if i == j:  # Skip self-loops
                    continue
                to_type = node_types[to_node]
                key = (from_type, to_type)
                if key in allowed_modes:
                    if key == ("consignee", "shipper") and (
                        from_node,
                        to_node,
                    ) not in shared_hinterland_pairs:
                        continue
                    for mode in allowed_modes[key]:
                        mode_idx = mode_indices[mode]
                        matrix[i][j][mode_idx] = 1

        return matrix

    def calc_transit_time_matrix(
        self,
        all_node_names: List[str],
        nodes_flat: Dict[str, Dict],
        transport_modes: List[str],
    ) -> List[List[List[int]]]:
        """Calculate transit time matrix for all node pairs.

        Args:
            all_node_names: Ordered list of all node names.
            nodes_flat: Dict mapping node name to node info (including 'type').
            transport_modes: Ordered list of transport mode names.

        Returns:
            3D matrix: transit_time[from_idx][to_idx][mode_idx] = time (0 if not applicable)
        """
        # Build node type lookup
        node_types = {name: nodes_flat[name]["type"] for name in all_node_names}

        # Get transit time rules
        transit_rules = self.calc_transit_time()

        # Build lookup: (from_type, to_type, mode) -> transit_time
        transit_lookup = {}
        for rule in transit_rules:
            key = (rule["from_node_type"], rule["to_node_type"], rule["transport_mode"])
            transit_lookup[key] = rule["transit_time"]

        # Build the 3D matrix
        n = len(all_node_names)
        k = len(transport_modes)
        mode_indices = {mode: idx for idx, mode in enumerate(transport_modes)}

        matrix = [
            [
                [0] * k  # Initialize all to 0
                for _ in range(n)
            ]
            for _ in range(n)
        ]

        for i, from_node in enumerate(all_node_names):
            from_type = node_types[from_node]
            for j, to_node in enumerate(all_node_names):
                if i == j:  # Skip self-loops (consistent with allowed_transport)
                    continue
                to_type = node_types[to_node]
                for mode_idx, mode in enumerate(transport_modes):
                    key = (from_type, to_type, mode)
                    if key in transit_lookup:
                        matrix[i][j][mode_idx] = transit_lookup[key]

        return matrix

    def compute_all(self, periods: int) -> Dict:
        """Compute all derived data.

        Args:
            periods: Number of planning periods.

        Returns:
            Dictionary containing all derived data.
        """
        return {
            "distance_matrix": self.calc_distance_matrix(),
            "hinterland_coverage": self.calc_hinterland_coverage(),
            "allowed_routes": self.calc_transport_modes(),
            "transit_time": self.calc_transit_time(),
        }

    def calc_transport_cost_matrix(
        self,
        all_node_names: List[str],
        transport_modes: List[str],
        distance_matrix: List[List[float]],
        unit_transport_cost: List[float],
    ) -> List[List[List[float]]]:
        """Calculate transport cost matrix: distance[i][j] * unit_cost[mode].

        Args:
            all_node_names: Ordered list of all node names.
            transport_modes: Ordered list of transport mode names.
            distance_matrix: 2D distance matrix [i][j] in km.
            unit_transport_cost: List of unit costs per mode (cost per km per container).

        Returns:
            3D matrix: transport_cost[from_idx][to_idx][mode_idx] = cost
        """
        n = len(all_node_names)
        k = len(transport_modes)

        matrix = [
            [
                [round(distance_matrix[i][j] * unit_transport_cost[m], 2)
                 for m in range(k)]
                for j in range(n)
            ]
            for i in range(n)
        ]
        return matrix

    def calc_arc_set_matrix(
        self,
        allowed_transport: List[List[List[int]]],
    ) -> List[List[int]]:
        """Calculate arc set matrix from allowed_transport.

        Arc set is a 2D binary matrix indicating valid arcs.
        A[i][j] = 1 iff exists k: allowed_transport[i][j][k] = 1

        Args:
            allowed_transport: 3D binary matrix [i][j][k]

        Returns:
            2D binary matrix [i][j] indicating arc validity
        """
        n = len(allowed_transport)
        k = len(allowed_transport[0][0]) if n > 0 else 0

        arc_set = [
            [1 if any(allowed_transport[i][j][m] == 1 for m in range(k)) else 0
             for j in range(n)]
            for i in range(n)
        ]

        return arc_set
