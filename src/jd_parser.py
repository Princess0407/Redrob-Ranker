"""
jd_parser.py

Extracts a structured JDConfig from data/skill_aliases.json.
All downstream modules import parse_jd() — never rebuild this object at runtime.

No network calls. No datetime.now(). Pure parsing only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class JDConfig:
    """
    Structured representation of the Job Description requirements.
    Populated from data/skill_aliases.json, which is the authoritative taxonomy.
    """
    # Hard requirements (3x BM25 query weight) — dict: canonical_name -> alias set
    hard_requirements: Dict[str, List[str]] = field(default_factory=dict)

    # Preferred requirements (1x weight)
    preferred_requirements: Dict[str, List[str]] = field(default_factory=dict)

    # Negative signal skill groups (by group name -> alias list)
    negative_signals: Dict[str, List[str]] = field(default_factory=dict)

    # Production-context pass B keywords (per Section 3 of architecture)
    production_keywords: List[str] = field(default_factory=list)

    # Rare-term safety net (per Section 3 of architecture)
    rare_terms: List[str] = field(default_factory=list)

    # All aliases flattened for fast membership checks
    all_hard_aliases: Set[str] = field(default_factory=set)
    all_preferred_aliases: Set[str] = field(default_factory=set)
    all_negative_aliases: Set[str] = field(default_factory=set)

    def get_all_query_terms(self) -> List[str]:
        """Return all hard + preferred aliases for BM25 Pass A query."""
        terms = []
        for aliases in self.hard_requirements.values():
            terms.extend(aliases)
        for aliases in self.preferred_requirements.values():
            terms.extend(aliases)
        return list(set(terms))

    def hard_req_names(self) -> List[str]:
        """Canonical names for the hard requirements (for coverage scoring)."""
        return list(self.hard_requirements.keys())

    def preferred_req_names(self) -> List[str]:
        return list(self.preferred_requirements.keys())


def parse_jd(skill_aliases_path: str) -> JDConfig:
    """
    Parse data/skill_aliases.json into a JDConfig object.

    Args:
        skill_aliases_path: Absolute or relative path to skill_aliases.json.

    Returns:
        JDConfig with all fields populated.

    Raises:
        FileNotFoundError: If the aliases file doesn't exist.
        ValueError: If the file is malformed.
    """
    if not os.path.isfile(skill_aliases_path):
        raise FileNotFoundError(
            f"skill_aliases.json not found at: {skill_aliases_path}"
        )

    with open(skill_aliases_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    jd = JDConfig()

    # Parse JD requirements section
    jd_reqs = raw.get("jd_requirements", {})
    for canonical_name, req_data in jd_reqs.items():
        req_type = req_data.get("type", "preferred")
        aliases = [a.lower().strip() for a in req_data.get("aliases", [])]

        if req_type == "hard_requirement":
            jd.hard_requirements[canonical_name] = aliases
            jd.all_hard_aliases.update(aliases)
        else:
            # "preferred" and any other type treated as preferred
            jd.preferred_requirements[canonical_name] = aliases
            jd.all_preferred_aliases.update(aliases)

    # Parse negative signals section
    neg = raw.get("negative_signals", {})
    for group_name, alias_list in neg.items():
        if group_name.startswith("_"):
            continue  # skip comment keys
        jd.negative_signals[group_name] = [a.lower().strip() for a in alias_list]
        jd.all_negative_aliases.update(a.lower().strip() for a in alias_list)

    # Production keywords for BM25 Pass B (Section 3, architecture doc)
    # These are hardcoded from the architecture spec — not configurable
    jd.production_keywords = [
        "deployed", "scale", "serving", "latency",
        "production", "inference", "throughput", "real-time",
        "pipeline", "distributed"
    ]

    # Rare-term safety net (Section 3, architecture doc)
    jd.rare_terms = ["pinecone", "lambdarank"]

    return jd


def hard_req_coverage_score(candidate: dict, jd_config: JDConfig) -> float:
    """
    Compute fraction of hard requirements covered by candidate's skills.

    A hard requirement is "covered" if any of its aliases appears (case-insensitive)
    in the candidate's skill names. Falls back gracefully on missing/empty skills.

    Schema fields read: skills[].name

    Returns: float in [0.0, 1.0]
    """
    skills = candidate.get("skills", [])
    if not skills or not jd_config.hard_requirements:
        return 0.0

    # Build lowercase set of candidate skill names
    candidate_skill_names: Set[str] = set()
    for s in skills:
        name = s.get("name", "")
        if name:
            candidate_skill_names.add(name.lower().strip())

    # Also scan career_history descriptions for alias presence
    career_text = " ".join(
        (ch.get("description", "") or "").lower()
        for ch in candidate.get("career_history", [])
    )

    covered = 0
    total = len(jd_config.hard_requirements)

    for canonical_name, aliases in jd_config.hard_requirements.items():
        # Check skill name match first, then description match
        if any(alias in candidate_skill_names for alias in aliases):
            covered += 1
        elif any(alias in career_text for alias in aliases):
            covered += 1

    return covered / total if total > 0 else 0.0


if __name__ == "__main__":
    import sys

    base_dir = os.path.dirname(os.path.abspath(__file__))
    aliases_path = os.path.join(base_dir, "data", "skill_aliases.json")

    jd = parse_jd(aliases_path)

    print("=== JDConfig ===")
    print(f"\nHard Requirements ({len(jd.hard_requirements)}):")
    for name, aliases in jd.hard_requirements.items():
        print(f"  {name}: {len(aliases)} aliases")

    print(f"\nPreferred Requirements ({len(jd.preferred_requirements)}):")
    for name, aliases in jd.preferred_requirements.items():
        print(f"  {name}: {len(aliases)} aliases")

    print(f"\nNegative Signal Groups ({len(jd.negative_signals)}):")
    for group, aliases in jd.negative_signals.items():
        print(f"  {group}: {len(aliases)} aliases")

    print(f"\nProduction Keywords ({len(jd.production_keywords)}): {jd.production_keywords}")
    print(f"Rare Terms ({len(jd.rare_terms)}): {jd.rare_terms}")
    print(f"\nTotal hard aliases (flat set): {len(jd.all_hard_aliases)}")
    print(f"Total preferred aliases (flat set): {len(jd.all_preferred_aliases)}")
    print(f"Total query terms (Pass A): {len(jd.get_all_query_terms())}")
