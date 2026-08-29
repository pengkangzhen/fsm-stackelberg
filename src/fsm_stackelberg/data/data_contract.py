"""Deterministic data catalog and DataEngineer semantic-contract validation.

The LLM is allowed to choose mathematical symbols, but it is not allowed to
invent physical data locations.  ``build_data_catalog`` derives the allowed
locations from ``sample.json``; ``validate_data_engineer_output`` checks that
each DE mapping selects one of those locations and uses compatible index sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..schemas import DataEngineerOutput


_VALUE_FIELD_NAMES = {
    "value",
    "linked",
    "calls",
    "transit_time",
    "distance_km",
    "capacity",
    "unit_cost",
    "probability",
}

_FIELD_SET_HINTS = {
    "hub": "hubs",
    "spoke": "spokes",
    "sea": "seas",
    "dry": "dry_ports",
    "node": "nodes",
    "from": "nodes",
    "to": "nodes",
    "period": "periods",
    "mode": "modes",
    "scenario": "scenarios",
}


@dataclass(frozen=True)
class ContractViolation:
    """One deterministic mismatch between a DE output and the data catalog."""

    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.code} at {self.location}: {self.message}"


def _access_pattern(path: Sequence[str]) -> str:
    return "data" + "".join(f"[{part!r}]" for part in path)


def _symbol_base(symbol: str) -> str:
    return re.split(r"\[", symbol or "", maxsplit=1)[0].strip()


def _set_values(sample: Mapping[str, Any]) -> Dict[str, set[Any]]:
    raw_sets = sample.get("sets", {})
    if not isinstance(raw_sets, Mapping):
        return {}
    return {
        str(name): set(values)
        for name, values in raw_sets.items()
        if isinstance(values, list)
    }


def _infer_index_source(
    field_name: str,
    values: Iterable[Any],
    set_values: Mapping[str, set[Any]],
) -> Optional[str]:
    """Infer the most specific declared set containing a record-key column."""
    observed = set(values)
    if not observed:
        return None

    hinted = _FIELD_SET_HINTS.get(field_name.lower())
    if hinted in set_values and observed.issubset(set_values[hinted]):
        return hinted

    # Prefer exact singular/plural name matches where available, then choose
    # the smallest containing set.  The latter handles hub→hubs, node→nodes,
    # period→periods, and subset-valued fields without domain prompt rules.
    normalized = field_name.lower().rstrip("s")
    candidates = [
        name
        for name, values_set in set_values.items()
        if observed.issubset(values_set)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda name: (
            0 if name.lower().rstrip("s") == normalized else 1,
            len(set_values[name]),
            name,
        )
    )
    return candidates[0]


def _entry(
    *,
    data_id: str,
    path: Sequence[str],
    kind: str,
    index_fields: Sequence[str] = (),
    index_sources: Sequence[Optional[str]] = (),
    value_field: Optional[str] = None,
    record_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "data_id": data_id,
        "path": list(path),
        "kind": kind,
        "access_pattern": _access_pattern(path),
        "index_fields": list(index_fields),
        "index_sources": list(index_sources),
        "value_field": value_field,
        "record_fields": list(record_fields),
    }


def build_data_catalog(sample: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the only physical data locations DE may reference.

    Catalog IDs are runtime paths:

    - sets promoted by preprocessing use bare IDs, e.g. ``hubs``;
    - nested record collections use dotted IDs, e.g. ``topology.arcs``;
    - values inside records use dotted field IDs, e.g.
      ``topology.arcs.capacity``.
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    set_values = _set_values(sample)

    # auto_preprocess promotes raw sets to top-level runtime keys.
    for name in sorted(set_values):
        catalog[name] = _entry(
            data_id=name,
            path=(name,),
            kind="set",
        )

    def walk(value: Any, path: tuple[str, ...]) -> None:
        data_id = ".".join(path)
        if isinstance(value, Mapping):
            if not value:
                catalog[data_id] = _entry(
                    data_id=data_id,
                    path=path,
                    kind="mapping",
                )
            for child_name, child in value.items():
                walk(child, (*path, str(child_name)))
            return

        if isinstance(value, list) and value and isinstance(value[0], Mapping):
            first = value[0]
            record_fields = list(first)
            value_fields = [
                field for field in record_fields if field in _VALUE_FIELD_NAMES
            ]
            key_fields = [
                field for field in record_fields if field not in value_fields
            ]
            index_sources = [
                _infer_index_source(
                    field,
                    (record.get(field) for record in value if field in record),
                    set_values,
                )
                for field in key_fields
            ]
            catalog[data_id] = _entry(
                data_id=data_id,
                path=path,
                kind="records",
                index_fields=key_fields,
                index_sources=index_sources,
                record_fields=record_fields,
            )
            for field in value_fields:
                field_id = f"{data_id}.{field}"
                catalog[field_id] = _entry(
                    data_id=field_id,
                    path=path,
                    kind="record_value",
                    index_fields=key_fields,
                    index_sources=index_sources,
                    value_field=field,
                    record_fields=record_fields,
                )
            return

        if isinstance(value, list):
            catalog[data_id] = _entry(
                data_id=data_id,
                path=path,
                kind="list",
            )
            return

        catalog[data_id] = _entry(
            data_id=data_id,
            path=path,
            kind="scalar",
        )

    for block, value in sample.items():
        if block in {"sets", "_meta"}:
            continue
        walk(value, (str(block),))

    return catalog


def format_data_catalog(catalog: Mapping[str, Mapping[str, Any]]) -> str:
    """Render a compact, value-free catalog for the DE/PD prompts."""
    lines = [
        "## Deterministic Data Catalog",
        "",
        "Use the exact `data_id` below as DE `source`; do not invent paths.",
        "Mathematical symbols may be aliases because `source` is explicit.",
        "Derived parameters belong in `derived_parameters`, not `parameters`.",
        "",
        "| data_id | kind | index fields → set sources | value field | runtime access |",
        "|---|---|---|---|---|",
    ]
    for data_id in sorted(catalog):
        item = catalog[data_id]
        index_pairs = ", ".join(
            f"{field}→{source or '?'}"
            for field, source in zip(
                item.get("index_fields", []),
                item.get("index_sources", []),
            )
        )
        lines.append(
            "| `{}` | {} | {} | {} | `{}` |".format(
                data_id,
                item.get("kind", ""),
                index_pairs or "—",
                item.get("value_field") or "—",
                item.get("access_pattern", ""),
            )
        )
    return "\n".join(lines)


def _resolved_catalog_metadata(
    source: str,
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    symbol: str,
) -> Dict[str, Any]:
    """Copy physical access metadata for one exact DE-selected catalog ID."""
    item = catalog.get(source)
    if item is None:
        raise ValueError(
            f"Cannot resolve symbol {symbol!r}: source {source!r} is not in "
            "the deterministic data catalog"
        )

    return {
        "source": source,
        "kind": item.get("kind"),
        "access_pattern": item.get("access_pattern"),
        "index_fields": list(item.get("index_fields", [])),
        "index_sources": list(item.get("index_sources", [])),
        "value_field": item.get("value_field"),
        "record_fields": list(item.get("record_fields", [])),
    }


def build_symbol_access_guide(
    output: DataEngineerOutput,
    catalog: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve DE symbols to exact catalog-backed runtime access metadata.

    The DE chooses mathematical aliases, while the catalog owns every physical
    path.  This function only joins those two contracts through the explicit
    ``source`` field; it never derives a physical key from a symbol.
    """
    guide: Dict[str, Any] = {
        "contract": "resolved_symbol_access_v1",
        "sets": {},
        "parameters": {},
        "derived_parameters": {},
    }
    model_inputs = output.model_inputs

    for set_def in model_inputs.sets:
        if set_def.symbol in guide["sets"]:
            raise ValueError(f"Duplicate set symbol {set_def.symbol!r}")
        metadata = _resolved_catalog_metadata(
            set_def.source or "",
            catalog,
            symbol=set_def.symbol,
        )
        if metadata["kind"] != "set":
            raise ValueError(
                f"Cannot resolve set symbol {set_def.symbol!r}: source "
                f"{set_def.source!r} is not a catalog set"
            )
        guide["sets"][set_def.symbol] = {
            **metadata,
            "index": set_def.index,
            "description": set_def.description,
            "sparse": False,
            "access_hint": None,
        }

    for parameter in model_inputs.parameters:
        if parameter.symbol in guide["parameters"]:
            raise ValueError(f"Duplicate parameter symbol {parameter.symbol!r}")
        if not parameter.source:
            raise ValueError(
                f"Cannot resolve parameter symbol {parameter.symbol!r}: "
                "physical parameters require an explicit catalog source"
            )
        metadata = _resolved_catalog_metadata(
            parameter.source,
            catalog,
            symbol=parameter.symbol,
        )
        if metadata["kind"] == "set":
            raise ValueError(
                f"Cannot resolve parameter symbol {parameter.symbol!r}: "
                f"source {parameter.source!r} is a catalog set"
            )
        guide["parameters"][parameter.symbol] = {
            **metadata,
            "indices": [
                {"symbol": index.symbol, "set": index.set}
                for index in parameter.indices
            ],
            "description": parameter.description,
            "sparse": bool(
                parameter.sparse
                or metadata["kind"] in {"records", "record_value"}
            ),
            "access_hint": parameter.access_hint,
        }

    for derived in model_inputs.derived_parameters:
        if derived.symbol in guide["derived_parameters"]:
            raise ValueError(f"Duplicate derived parameter symbol {derived.symbol!r}")
        guide["derived_parameters"][derived.symbol] = {
            "kind": "derived",
            "source": None,
            "access_pattern": None,
            "indices": [
                {"symbol": index.symbol, "set": index.set}
                for index in derived.indices
            ],
            "description": derived.description,
            "derivation_logic": derived.derivation_logic,
            "source_parameters": list(derived.source_parameters),
        }

    return guide


def build_pd_data_access_context(
    output: DataEngineerOutput,
    catalog: Mapping[str, Mapping[str, Any]],
    physical_schema_context: Optional[str] = None,
) -> str:
    """Render the one resolved guide used by both PD forward and backward."""
    resolved = json.dumps(
        build_symbol_access_guide(output, catalog),
        indent=2,
        sort_keys=True,
    )
    sections = [
        "## Resolved Symbol Access Guide (authoritative)",
        resolved,
    ]
    if physical_schema_context:
        sections.extend(
            [
                "## Physical Schema Context (supplementary)",
                physical_schema_context,
            ]
        )
    return "\n\n".join(sections)


def validate_data_engineer_output(
    output: DataEngineerOutput,
    catalog: Mapping[str, Mapping[str, Any]],
) -> List[ContractViolation]:
    """Validate DE's semantic mapping against deterministic physical data."""
    violations: List[ContractViolation] = []
    model_inputs = output.model_inputs

    declared_sets_by_symbol = {
        item.symbol: item for item in model_inputs.sets
    }
    declared_sets_by_source = {
        item.source: item
        for item in model_inputs.sets
        if item.source
    }

    if len(declared_sets_by_symbol) != len(model_inputs.sets):
        violations.append(
            ContractViolation(
                "duplicate_set_symbol",
                "model_inputs.sets",
                "set symbols must be unique",
            )
        )

    seen_set_indices: set[str] = set()
    for position, set_def in enumerate(model_inputs.sets):
        location = f"model_inputs.sets[{position}]"
        source = set_def.source or ""
        item = catalog.get(source)
        if not source or item is None or item.get("kind") != "set":
            violations.append(
                ContractViolation(
                    "unknown_set_source",
                    f"{location}.source",
                    f"{source!r} is not a catalog set data_id",
                )
            )
        if set_def.index in seen_set_indices:
            violations.append(
                ContractViolation(
                    "duplicate_set_index",
                    f"{location}.index",
                    f"index {set_def.index!r} is already assigned to another set",
                )
            )
        seen_set_indices.add(set_def.index)

    seen_parameter_symbols: set[str] = set()
    for position, parameter in enumerate(model_inputs.parameters):
        location = f"model_inputs.parameters[{position}]"
        if parameter.symbol in seen_parameter_symbols:
            violations.append(
                ContractViolation(
                    "duplicate_parameter_symbol",
                    f"{location}.symbol",
                    f"parameter symbol {parameter.symbol!r} is duplicated",
                )
            )
        seen_parameter_symbols.add(parameter.symbol)

        source = parameter.source
        if not source:
            violations.append(
                ContractViolation(
                    "derived_parameter_in_parameters",
                    f"{location}.source",
                    "source is null; move this item to derived_parameters and "
                    "provide derivation_logic",
                )
            )
            continue
        item = catalog.get(source)
        if item is None or item.get("kind") == "set":
            violations.append(
                ContractViolation(
                    "unknown_parameter_source",
                    f"{location}.source",
                    f"{source!r} is not a catalog parameter data_id",
                )
            )
            continue

        expected_sources = list(item.get("index_sources", []))
        if len(parameter.indices) != len(expected_sources):
            violations.append(
                ContractViolation(
                    "index_arity_mismatch",
                    f"{location}.indices",
                    f"{parameter.symbol!r} declares {len(parameter.indices)} "
                    f"indices but {source!r} requires {len(expected_sources)}",
                )
            )
            continue

        for index_position, (index, expected_source) in enumerate(
            zip(parameter.indices, expected_sources)
        ):
            index_location = f"{location}.indices[{index_position}]"
            set_def = declared_sets_by_symbol.get(index.set)
            if set_def is None:
                violations.append(
                    ContractViolation(
                        "unknown_index_set",
                        f"{index_location}.set",
                        f"{index.set!r} is not a declared set symbol",
                    )
                )
                continue
            if expected_source and set_def.source != expected_source:
                expected_set = declared_sets_by_source.get(expected_source)
                expected_label = (
                    expected_set.symbol if expected_set else expected_source
                )
                violations.append(
                    ContractViolation(
                        "index_domain_mismatch",
                        index_location,
                        f"catalog field "
                        f"{item.get('index_fields', [])[index_position]!r} "
                        f"requires set {expected_label!r} "
                        f"(source {expected_source!r}), got {index.set!r}",
                    )
                )

    all_symbols = seen_parameter_symbols | {
        item.symbol for item in model_inputs.derived_parameters
    }
    for position, derived in enumerate(model_inputs.derived_parameters):
        location = f"model_inputs.derived_parameters[{position}]"
        if not derived.derivation_logic.strip():
            violations.append(
                ContractViolation(
                    "missing_derivation",
                    f"{location}.derivation_logic",
                    f"derived parameter {_symbol_base(derived.symbol)!r} "
                    "requires an explicit formula",
                )
            )
        for source_symbol in derived.source_parameters:
            if source_symbol not in all_symbols:
                violations.append(
                    ContractViolation(
                        "unknown_derivation_source",
                        f"{location}.source_parameters",
                        f"{source_symbol!r} is not a declared parameter symbol",
                    )
                )

    return violations


def format_contract_feedback(
    violations: Sequence[ContractViolation],
) -> str:
    """Format bounded retry feedback without adding domain-specific hints."""
    return "\n".join(f"- {violation}" for violation in violations)
