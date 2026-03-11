"""
src/schema_discovery/property_analyzer.py
-----------------------------------------
Property type inference, constraint detection, and analysis.

**Paper basis**: PG-HIVE (Sideri et al., 2024) — Section 4.4
  - Property data type inference: INTEGER → FLOAT → DATE → STRING hierarchy
  - Property constraints: MANDATORY (appears in 100% of instances) vs OPTIONAL
  - Priority-based type inference from sampled values

**Paper basis**: Schema Inference (Lbath et al., 2021) — Section 4.2
  - MapReduce-based type inference
  - Complex nested property values
  - Kind-equivalence for type fusion

Algorithm (PG-HIVE §4.4)
-------------------------
For each property p of type T:
  1. Sample values v1, v2, ..., vk
  2. If v is integer -> INTEGER
  3. If v is real non-integer -> FLOAT
  4. If v is boolean -> BOOLEAN
  5. If v matches date/time ISO -> DATE
  6. Else -> STRING
  7. If f_T(p) = 1.0 -> MANDATORY, else OPTIONAL
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from src.utils.connector import Neo4jConnector

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class PropertyInfo:
    """Information about a single property of a node/edge type."""
    name: str
    data_type: str = "STRING"         # STRING, INTEGER, FLOAT, BOOLEAN, DATE, LIST
    constraint: str = "OPTIONAL"      # MANDATORY | OPTIONAL
    frequency: float = 0.0            # Fraction of instances that have this property
    unique_ratio: float = 0.0         # count(DISTINCT) / count(total)
    sample_values: list = field(default_factory=list)
    total_count: int = 0
    distinct_count: int = 0


@dataclass
class TypeProperties:
    """All property information for a single node or edge type."""
    label: str
    properties: dict[str, PropertyInfo] = field(default_factory=dict)
    instance_count: int = 0


# ──────────────────────────────────────────────────────────────
# Date regex patterns (from PG-HIVE)
# ──────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),                    # 2024-01-15
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),                    # 01/15/2024
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"),        # ISO datetime
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),                    # DD-MM-YYYY
    re.compile(r"^\d{1,2}\s+\w+\s+\d{4}$"),               # 15 January 2024
]


# ──────────────────────────────────────────────────────────────
# Main analyzer
# ──────────────────────────────────────────────────────────────

class PropertyAnalyzer:
    """
    Analyzes properties of node/edge types in a Neo4j graph.

    Infers data types, constraints (mandatory/optional), cardinality,
    and uniqueness for each property.

    Parameters
    ----------
    connector : Neo4jConnector
        Active Neo4j connection.
    sample_size : int
        Number of property values to sample for type inference (default 100).
    """

    def __init__(self, connector: Neo4jConnector, sample_size: int = 100):
        self.connector = connector
        self.sample_size = sample_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_node_properties(self, label: str) -> TypeProperties:
        """
        Analyze all properties of nodes with the given label.

        Implements PG-HIVE §4.4:
        - Discovers all property keys
        - Infers data type for each
        - Determines MANDATORY vs OPTIONAL
        - Computes uniqueness ratio
        """
        result = TypeProperties(label=label)

        # Get instance count
        row = self.connector.run_single(
            f"MATCH (n:`{label}`) RETURN count(n) AS cnt"
        )
        result.instance_count = row["cnt"] if row else 0
        if result.instance_count == 0:
            return result

        # Discover property keys
        prop_keys = self._discover_property_keys(label)

        # Analyze each property
        for key in prop_keys:
            info = self._analyze_property(label, key, result.instance_count)
            result.properties[key] = info

        return result

    def analyze_edge_properties(
        self, rel_type: str, source_label: str = "", target_label: str = ""
    ) -> TypeProperties:
        """Analyze all properties of edges with the given type."""
        result = TypeProperties(label=rel_type)

        # Match pattern
        src = f":`{source_label}`" if source_label else ""
        tgt = f":`{target_label}`" if target_label else ""
        pattern = f"({src})-[r:`{rel_type}`]->({tgt})"

        row = self.connector.run_single(
            f"MATCH {pattern} RETURN count(r) AS cnt"
        )
        result.instance_count = row["cnt"] if row else 0
        if result.instance_count == 0:
            return result

        # Discover edge property keys
        rows = self.connector.run(
            f"MATCH {pattern} UNWIND keys(r) AS k "
            f"RETURN DISTINCT k ORDER BY k"
        )
        prop_keys = [r["k"] for r in rows]

        for key in prop_keys:
            info = self._analyze_edge_property(
                rel_type, key, result.instance_count,
                source_label, target_label,
            )
            result.properties[key] = info

        return result

    # ------------------------------------------------------------------
    # Property key discovery
    # ------------------------------------------------------------------

    def _discover_property_keys(self, label: str) -> list[str]:
        """Discover all property keys used by nodes of this label."""
        rows = self.connector.run(
            f"MATCH (n:`{label}`) UNWIND keys(n) AS k "
            f"RETURN DISTINCT k ORDER BY k"
        )
        return [r["k"] for r in rows]

    # ------------------------------------------------------------------
    # Per-property analysis
    # ------------------------------------------------------------------

    def _analyze_property(
        self, label: str, key: str, total: int
    ) -> PropertyInfo:
        """Analyze a single property of a node type."""
        info = PropertyInfo(name=key, total_count=total)

        # Count non-null instances
        row = self.connector.run_single(
            f"MATCH (n:`{label}`) WHERE n.`{key}` IS NOT NULL "
            f"RETURN count(n) AS cnt"
        )
        non_null = row["cnt"] if row else 0
        info.frequency = non_null / total if total > 0 else 0

        # MANDATORY vs OPTIONAL (PG-HIVE §4.4)
        info.constraint = "MANDATORY" if info.frequency >= 1.0 else "OPTIONAL"

        # Count distinct values
        row = self.connector.run_single(
            f"MATCH (n:`{label}`) WHERE n.`{key}` IS NOT NULL "
            f"RETURN count(DISTINCT n.`{key}`) AS cnt"
        )
        info.distinct_count = row["cnt"] if row else 0
        info.unique_ratio = info.distinct_count / total if total > 0 else 0

        # Sample values for type inference
        rows = self.connector.run(
            f"MATCH (n:`{label}`) WHERE n.`{key}` IS NOT NULL "
            f"RETURN n.`{key}` AS val LIMIT {self.sample_size}"
        )
        info.sample_values = [r["val"] for r in rows]

        # Infer data type (PG-HIVE priority hierarchy)
        info.data_type = self._infer_type(info.sample_values)

        return info

    def _analyze_edge_property(
        self, rel_type: str, key: str, total: int,
        source_label: str = "", target_label: str = "",
    ) -> PropertyInfo:
        """Analyze a single property of an edge type."""
        info = PropertyInfo(name=key, total_count=total)

        src = f":`{source_label}`" if source_label else ""
        tgt = f":`{target_label}`" if target_label else ""
        pattern = f"({src})-[r:`{rel_type}`]->({tgt})"

        row = self.connector.run_single(
            f"MATCH {pattern} WHERE r.`{key}` IS NOT NULL "
            f"RETURN count(r) AS cnt"
        )
        non_null = row["cnt"] if row else 0
        info.frequency = non_null / total if total > 0 else 0
        info.constraint = "MANDATORY" if info.frequency >= 1.0 else "OPTIONAL"

        row = self.connector.run_single(
            f"MATCH {pattern} WHERE r.`{key}` IS NOT NULL "
            f"RETURN count(DISTINCT r.`{key}`) AS cnt"
        )
        info.distinct_count = row["cnt"] if row else 0
        info.unique_ratio = info.distinct_count / total if total > 0 else 0

        rows = self.connector.run(
            f"MATCH {pattern} WHERE r.`{key}` IS NOT NULL "
            f"RETURN r.`{key}` AS val LIMIT {self.sample_size}"
        )
        info.sample_values = [r["val"] for r in rows]
        info.data_type = self._infer_type(info.sample_values)

        return info

    # ------------------------------------------------------------------
    # Type inference (PG-HIVE §4.4 priority hierarchy)
    # ------------------------------------------------------------------

    def _infer_type(self, values: list) -> str:
        """
        Infer the data type from sampled values.

        Priority (PG-HIVE):
          BOOLEAN > INTEGER > FLOAT > DATE > STRING

        If mixed types, pick the most general compatible type.
        """
        if not values:
            return "STRING"

        type_counts = {"BOOLEAN": 0, "INTEGER": 0, "FLOAT": 0, "DATE": 0, "STRING": 0, "LIST": 0}

        for val in values:
            t = self._classify_value(val)
            type_counts[t] = type_counts.get(t, 0) + 1

        # If any list values, it's LIST
        if type_counts["LIST"] > 0:
            return "LIST"

        # Pick the type with majority
        total = sum(type_counts.values())
        for dtype in ["BOOLEAN", "INTEGER", "FLOAT", "DATE", "STRING"]:
            if type_counts[dtype] / total > 0.5:
                return dtype

        # Fallback: if mix of INTEGER and FLOAT, use FLOAT
        if type_counts["INTEGER"] + type_counts["FLOAT"] > total * 0.5:
            return "FLOAT"

        return "STRING"

    def _classify_value(self, val) -> str:
        """Classify a single value into a data type."""
        if isinstance(val, bool):
            return "BOOLEAN"
        if isinstance(val, int):
            return "INTEGER"
        if isinstance(val, float):
            return "FLOAT"
        if isinstance(val, list):
            return "LIST"
        if isinstance(val, str):
            # Check for boolean strings
            if val.lower() in ("true", "false"):
                return "BOOLEAN"
            # Check for date patterns
            for pattern in _DATE_PATTERNS:
                if pattern.match(val):
                    return "DATE"
            # Check for numeric strings
            try:
                int(val)
                return "INTEGER"
            except (ValueError, TypeError):
                pass
            try:
                float(val)
                return "FLOAT"
            except (ValueError, TypeError):
                pass
        return "STRING"
