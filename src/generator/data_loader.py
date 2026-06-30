"""Base data loader for ECR instance generator."""

import json
from pathlib import Path
from typing import Dict, List, Any


class BaseDataLoader:
    """Load base data files for ECR instance generation."""

    def __init__(self, base_dir: str | Path):
        """Initialize the data loader.

        Args:
            base_dir: Directory containing base data files.
        """
        self.base_dir = Path(base_dir)

    def _load_json(self, filename: str) -> Any:
        """Load a JSON file from the base directory.

        Args:
            filename: Name of the JSON file.

        Returns:
            Loaded JSON data.
        """
        filepath = self.base_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_seaports(self) -> List[Dict]:
        """Load seaport data.

        Returns:
            List of seaport dictionaries with name, name_zh, and location.
        """
        return self._load_json("seaports.json")

    def load_dryports(self) -> List[Dict]:
        """Load dry port data.

        Returns:
            List of dry port dictionaries with name, name_zh, adcode, and location.
        """
        return self._load_json("dryports.json")

    def load_candidate_nodes(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load candidate shippers and consignees grouped by dry port.

        Returns:
            Dictionary with 'shippers' and 'consignees' keys, each containing
            a dictionary mapping dry port names to lists of candidate nodes.
        """
        return self._load_json("candidate_nodes.json")

    def load_base_costs(self) -> Dict:
        """Load base cost parameters.

        Returns:
            Dictionary containing unit transport costs, holding costs,
            renting costs, and other cost-related parameters.
        """
        return self._load_json("costs_base.json")

    def load_all(self) -> Dict[str, Any]:
        """Load all base data.

        Returns:
            Dictionary containing all base data with keys:
            'seaports', 'dryports', 'candidate_nodes', 'costs_base'.
        """
        return {
            "seaports": self.load_seaports(),
            "dryports": self.load_dryports(),
            "candidate_nodes": self.load_candidate_nodes(),
            "costs_base": self.load_base_costs(),
        }
