"""Deterministic data preprocessing — V2 adaptive compression.

Reads raw sample data (JSON), and produces either:
  Level 0 (small data, raw_chars < 240K / ~60K tokens):
      Returns raw JSON string verbatim — no compression needed.
  Level 1 (large data, raw_chars >= 240K):
      Produces a Schema declaration + pipe-delimited compact data table.
      All numeric values are preserved losslessly.  No domain knowledge is
      leaked (e.g., which node types have storage capability).

Both levels return the same tuple: ``(processed_dict, data_access_guide)``.
The ``processed_dict`` is always the full converted data (for code execution),
while ``data_access_guide`` is the text fed to the LLM prompt.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Threshold: raw JSON chars above which we switch to Level 1 compression.
# 150K compact chars ≈ 38K tokens.  At this scale, indented JSON reaches ~55K
# tokens which fills a significant portion of most LLM context windows and
# benefits from structured compression to leave room for prompt + reasoning.
_COMPRESS_THRESHOLD_CHARS = 150_000


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

_VALUE_FIELDS = {"value", "cost", "time", "distance"}


def _detect_format(data) -> str:
    if isinstance(data, dict):
        return "named_dict"
    if isinstance(data, list):
        if not data:
            return "list"
        if isinstance(data[0], dict):
            return "records"
        return "list"
    return "unknown"


def _infer_key_fields(records: list) -> List[str]:
    if not records:
        return []
    return [k for k in records[0] if k not in _VALUE_FIELDS]


def _infer_value_field(records: list) -> Optional[str]:
    if not records:
        return None
    for alt in ("value", "cost", "time", "distance"):
        if alt in records[0]:
            return alt
    return None


def _records_to_dict(
    records: list,
    key_fields: Optional[List[str]] = None,
    value_field: Optional[str] = None,
) -> Dict:
    """Convert records-format data to dict with tuple keys."""
    if not records:
        return {}

    if key_fields is None:
        key_fields = _infer_key_fields(records)
    if value_field is None:
        value_field = _infer_value_field(records)

    has_value = value_field and value_field in records[0]

    if len(key_fields) == 1:
        kf = key_fields[0]
        return (
            {r[kf]: r[value_field] for r in records}
            if has_value
            else {r[kf]: 1 for r in records}
        )
    else:
        key_fn = lambda r: tuple(r[f] for f in key_fields)  # noqa: E731
        return (
            {key_fn(r): r[value_field] for r in records}
            if has_value
            else {key_fn(r): 1 for r in records}
        )


# ---------------------------------------------------------------------------
# Arc reconciliation
# ---------------------------------------------------------------------------

def _reconcile_arc_parameters(processed: Dict) -> None:
    allowed_transport = processed.get("allowed_transport")
    if not isinstance(allowed_transport, dict) or not allowed_transport:
        return

    transport_cost = processed.get("transport_cost")
    if isinstance(transport_cost, dict):
        missing = sorted(set(allowed_transport) - set(transport_cost))
        if missing:
            examples = ", ".join(map(str, missing[:5]))
            raise ValueError(
                f"transport_cost missing {len(missing)} allowed arc(s), e.g. {examples}"
            )

    transit_time = processed.get("transit_time_matrix")
    if transit_time is None:
        processed["transit_time_matrix"] = {arc: 0 for arc in allowed_transport}
        return

    if isinstance(transit_time, dict):
        missing = set(allowed_transport) - set(transit_time)
        for arc in missing:
            transit_time[arc] = 0
        if missing:
            logger.info("Filled %d missing transit_time entries with 0", len(missing))


# ---------------------------------------------------------------------------
# Derived sets & parameters
# ---------------------------------------------------------------------------

def _derive_sets(processed: Dict, meta: Dict) -> None:
    """Compute derived sets (union relationships) from _meta or defaults."""
    sets_meta = meta.get("sets", {})
    for name, spec in sets_meta.items():
        if name in processed:
            continue
        union_of = spec.get("union")
        if union_of:
            merged = []
            seen = set()
            for member_set in union_of:
                for item in processed.get(member_set, []):
                    key = item if isinstance(item, str) else str(item)
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)
            processed[name] = merged
            logger.debug("Derived set %s = union(%s), %d items", name, union_of, len(merged))


def _derive_parameters(processed: Dict) -> None:
    """Compute derived parameters not present in raw data."""
    hc = processed.get("hinterland_consignee")
    hs = processed.get("hinterland_shipper")
    dryports = processed.get("dryports", [])

    if (
        isinstance(hc, dict)
        and isinstance(hs, dict)
        and dryports
        and "street_turn" not in processed
    ):
        consignees = set(c for (_, c) in hc.keys())
        shippers = set(s for (_, s) in hs.keys())
        street_turn = {}
        for p in consignees:
            for q in shippers:
                eligible = 0
                for j in dryports:
                    if hc.get((j, p), 0) and hs.get((j, q), 0):
                        eligible = 1
                        break
                street_turn[(p, q)] = eligible
        processed["street_turn"] = street_turn
        logger.debug("Derived street_turn: %d entries", len(street_turn))


# ---------------------------------------------------------------------------
# Level 1: Schema + Compact Data generation
# ---------------------------------------------------------------------------

def _build_schema_and_compact(processed: Dict) -> str:
    """Build Level 1 output: Schema + pipe-delimited compact data.

    Builds the Schema from the **processed** dict (sets promoted to top-level,
    parameters as tuple-keyed sparse dicts) so the guide matches what the
    solver receives.

    Design principles:
    - All numeric values are preserved losslessly.
    - Field names, key types, and entry counts are explicitly declared.
    - No domain knowledge is leaked (e.g. which node types have storage).
    """
    sections: list[str] = []

    # Classify processed keys into sets vs parameters
    set_keys: list[str] = []
    param_keys: list[str] = []
    for key, val in processed.items():
        if isinstance(val, list) and val and isinstance(val[0], str):
            set_keys.append(key)
        elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
            set_keys.append(key)
        else:
            param_keys.append(key)

    # --- Schema declaration ---
    schema_lines = [
        "## Data Schema",
        "",
        "The `data` dict passed to `optimize(data)` has the following structure:",
        "- All sets and parameters are top-level keys: `data['<name>']`",
        "- Parameters with composite keys are sparse dicts with tuple keys",
        "",
    ]

    # Sets
    if set_keys:
        schema_lines.append("### Sets")
        for key in set_keys:
            values = processed[key]
            if isinstance(values, list):
                if values and isinstance(values[0], (int, float)):
                    schema_lines.append(f"- `data['{key}']`: list of {len(values)} integers")
                else:
                    schema_lines.append(f"- `data['{key}']`: list of {len(values)} strings")
                    vals_str = ", ".join(str(v) for v in values[:10])
                    if len(values) > 10:
                        vals_str += ", ..."
                    schema_lines.append(f"  values: [{vals_str}]")
        schema_lines.append("")

    # Parameters
    schema_lines.append("### Parameters (sparse dicts with tuple keys)")
    schema_lines.append("")

    for key in param_keys:
        val = processed[key]

        if isinstance(val, dict):
            n_entries = len(val)
            if not val:
                schema_lines.append(f"- `data['{key}']`: dict (empty)")
                continue
            first_key = next(iter(val))
            first_val = val[first_key]
            if isinstance(first_key, tuple):
                dims = len(first_key)
                key_types = ", ".join(type(p).__name__ for p in first_key)
                schema_lines.append(
                    f"- `data['{key}']`: dict[{n_entries}], "
                    f"key: {dims}-tuple ({key_types}), "
                    f"value: {type(first_val).__name__}"
                )
            elif isinstance(first_key, str):
                schema_lines.append(
                    f"- `data['{key}']`: dict[{n_entries}], "
                    f"key: string, value: {type(first_val).__name__}"
                )
            else:
                schema_lines.append(f"- `data['{key}']`: dict[{n_entries}]")
        elif isinstance(val, list):
            schema_lines.append(f"- `data['{key}']`: list[{len(val)}]")
        else:
            schema_lines.append(f"- `data['{key}']`: {type(val).__name__} = {val}")

    schema_lines.append("")
    sections.append("\n".join(schema_lines))

    # --- Compact Data (pipe-delimited tables) ---
    compact_lines = ["## Compact Data", ""]
    compact_lines.append(
        "Each parameter below is encoded as a pipe-delimited table. "
        "Header line shows column names; data lines follow."
    )
    compact_lines.append("")

    for key in param_keys:
        val = processed[key]

        if isinstance(val, dict):
            compact_lines.append(f"### {key} [{len(val)} entries]")
            if not val:
                compact_lines.append("(empty)")
                compact_lines.append("")
                continue
            first_key = next(iter(val))
            if isinstance(first_key, tuple):
                dims = len(first_key)
                header_parts = [f"key{i+1}" for i in range(dims)] + ["value"]
                compact_lines.append("| " + " | ".join(header_parts) + " |")
                for k, v in val.items():
                    row = " | ".join(str(part) for part in k) + f" | {v}"
                    compact_lines.append(f"| {row} |")
            else:
                compact_lines.append("| key | value |")
                for k, v in val.items():
                    compact_lines.append(f"| {k} | {v} |")
            compact_lines.append("")

        elif isinstance(val, list):
            compact_lines.append(f"### {key}")
            compact_lines.append(str(val))
            compact_lines.append("")

    sections.append("\n".join(compact_lines))

    # --- Access rules ---
    access_lines = [
        "## Access Rules",
        "- Sets: `periods = data['periods']`, `all_nodes = data['all_nodes']`, etc.",
        "- Tuple-keyed dicts: iterate `for (node, period), value in data['supply'].items():`",
        "- String-keyed dicts: iterate `for node, cap in data['storage_capacity'].items():`",
        "- Always use `.get(key, default)` for safe access",
        "- If a parameter is NOT listed above, compute it from listed parameters",
        "",
    ]
    sections.append("\n".join(access_lines))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _is_fundamental_data(sample: Dict) -> bool:
    """Detect whether sample contains only fundamental data (no derived parameters).

    Fundamental data comes from CSV files and lacks derived parameters like
    distance_matrix, allowed_transport, transport_cost, etc.
    """
    _DERIVED_KEYS = {
        "distance_matrix", "allowed_transport", "arc_set",
        "transit_time_matrix", "transport_cost",
        "hinterland_consignee", "hinterland_shipper", "street_turn",
    }
    return not any(k in sample for k in _DERIVED_KEYS)


def _process_sample(sample: Dict) -> Dict:
    """Process raw sample into flat-sets + tuple-keyed sparse dict format.

    Always produces the same consistent structure regardless of input format:
    - Sets promoted to top-level keys
    - Records converted to tuple-keyed sparse dicts
    """
    meta = sample.get("_meta", {})
    processed: Dict = {}

    raw_sets = sample.get("sets", {})
    if isinstance(raw_sets, dict):
        for name, values in raw_sets.items():
            processed[name] = values

    is_fundamental = _is_fundamental_data(sample)
    # For fundamental data, keep "nodes" (coordinates needed for derivation);
    # for full data, skip it (derived params already exist).
    _SKIP_KEYS = {"sets", "_meta"}
    if not is_fundamental:
        _SKIP_KEYS.add("nodes")

    for key, raw_val in sample.items():
        if key in _SKIP_KEYS:
            continue
        if key in processed:
            continue

        fmt = _detect_format(raw_val)
        if fmt == "records":
            p_meta = meta.get("parameters", {}).get(key, {})
            kf = p_meta.get("key_fields") or _infer_key_fields(raw_val)
            vf = p_meta.get("value_field") or _infer_value_field(raw_val)
            processed[key] = _records_to_dict(raw_val, kf, vf)
        else:
            processed[key] = raw_val

    _derive_sets(processed, meta)

    # Only derive and reconcile if derived params are expected (full data)
    if not is_fundamental:
        _derive_parameters(processed)
        _reconcile_arc_parameters(processed)

    return processed


def auto_preprocess(sample: Dict) -> Tuple[Dict, str]:
    """Deterministic preprocessing of raw sample data.

    Always returns (processed_dict, data_access_guide) where:
    - processed_dict has flat sets and tuple-keyed sparse dicts
    - data_access_guide uses Schema+Compact format (consistent with processed_dict)

    Modes:
      - Fundamental data (CSV, no derived params): Schema+Compact of fundamental
        data only. LLM must derive missing params using knowledge modules.
      - Full data: Schema+Compact of processed data including derived params.

    Args:
        sample: Raw sample data dict (from sample.json or CSV loader).

    Returns:
        (data_dict, data_access_guide)
        - data_dict: Processed data with flat sets and tuple-keyed sparse dicts.
        - data_access_guide: Schema + Compact Data text for LLM prompts.
    """
    processed = _process_sample(sample)
    is_fundamental = _is_fundamental_data(sample)

    # Always build Schema+Compact guide from the processed dict
    data_access_guide = _build_schema_and_compact(processed)

    raw_chars = len(json.dumps(sample, ensure_ascii=False))
    guide_chars = len(data_access_guide)

    if is_fundamental:
        logger.info(
            "Fundamental data: %d fields, guide %d chars (~%d tokens), "
            "no derived parameters — LLM must derive from knowledge",
            len(processed), guide_chars, guide_chars // 4,
        )
    else:
        ratio = guide_chars / raw_chars * 100 if raw_chars > 0 else 0
        logger.info(
            "Full data: %d fields, %d → %d chars (~%d tokens), ratio %.1f%%",
            len(processed), raw_chars, guide_chars, guide_chars // 4, ratio,
        )

    return processed, data_access_guide
