"""Term-to-regulation mapping for case-derived query generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class TermType(StrEnum):
    """How tightly a term binds to its regulatory section."""

    ANCHOR = "anchor"
    CONTEXTUAL = "contextual"
    CROSS_CUTTING = "cross_cutting"


class MatchMode(StrEnum):
    """How to match a term in text."""

    EXACT = "exact"
    PHRASE = "phrase"
    SMART_PHRASE = "smart_phrase"
    REGEX = "regex"


@dataclass(frozen=True, slots=True)
class AnchorRef:
    """A reference to a regulatory section with weight."""

    ref: str  # key into TermMap.refs
    weight: float  # 0.0-1.0


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Configuration for how to match a term."""

    mode: MatchMode = MatchMode.SMART_PHRASE
    case_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class TermActions:
    """Actions to take when a term is matched."""

    retrieval_boost: float = 0.5
    adams_prefilter: bool = False


@dataclass(frozen=True, slots=True)
class RefEntry:
    """A regulatory reference entry."""

    label: str
    kind: str


@dataclass(frozen=True, slots=True)
class TermRecord:
    """A single entry in the term-to-regulation dictionary."""

    id: str
    canonical: str
    term_type: TermType
    anchors: list[AnchorRef]
    aliases: list[str] = field(default_factory=list)
    match: MatchConfig = field(default_factory=MatchConfig)
    actions: TermActions = field(default_factory=TermActions)
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class TermMap:
    """The complete term map with refs and terms."""

    version: str
    refs: dict[str, RefEntry]
    terms: list[TermRecord]


@dataclass(frozen=True, slots=True)
class TermMatch:
    """A dictionary term found in case content."""

    term: str
    term_id: str
    term_type: TermType
    anchors: list[AnchorRef]
    actions: TermActions
    frequency: int


# Default actions by term type
DEFAULT_ACTIONS: dict[TermType, TermActions] = {
    TermType.ANCHOR: TermActions(retrieval_boost=0.8, adams_prefilter=True),
    TermType.CONTEXTUAL: TermActions(retrieval_boost=0.4, adams_prefilter=False),
    TermType.CROSS_CUTTING: TermActions(retrieval_boost=0.1, adams_prefilter=False),
}


@dataclass(frozen=True, slots=True)
class TermMapper:
    """Loads a term-to-regulation dictionary and matches terms in text."""

    _term_map: TermMap
    _terms_index: dict[str, TermRecord]  # canonical + aliases (lowercase) -> TermRecord

    @classmethod
    def from_json(cls, path: Path) -> TermMapper:
        """Load and validate a term dictionary from a JSON file.

        Expected format:
        {
          "version": "0.1",
          "refs": {"cfr.10.50.46": {"label": "10 CFR 50.46", "kind": "cfr"}},
          "terms": [
            {
              "id": "term.eccs",
              "canonical": "ECCS",
              "aliases": ["emergency core cooling"],
              "type": "anchor",
              "anchors": [{"ref": "cfr.10.50.46", "weight": 1.0}],
              "match": {"mode": "smart_phrase", "case_sensitive": false},
              "actions": {"retrieval_boost": 1.0, "adams_prefilter": true},
              "notes": "..."
            }
          ]
        }

        Raises ValueError on malformed entries or duplicate aliases.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"Expected JSON object, got {type(raw).__name__}"
            raise ValueError(msg)

        # Validate version
        version = raw.get("version", "0.1")
        if not isinstance(version, str):
            msg = f"Version must be str, got {type(version).__name__}"
            raise ValueError(msg)

        # Validate and build refs
        refs_raw = raw.get("refs", {})
        if not isinstance(refs_raw, dict):
            msg = f"refs must be an object, got {type(refs_raw).__name__}"
            raise ValueError(msg)
        refs: dict[str, RefEntry] = {}
        for ref_id, ref_data in refs_raw.items():
            if not isinstance(ref_data, dict):
                msg = f"Ref '{ref_id}' must be an object"
                raise ValueError(msg)
            label = ref_data.get("label")
            kind = ref_data.get("kind")
            if not isinstance(label, str):
                msg = f"Ref '{ref_id}' missing or invalid 'label'"
                raise ValueError(msg)
            if not isinstance(kind, str):
                msg = f"Ref '{ref_id}' missing or invalid 'kind'"
                raise ValueError(msg)
            refs[ref_id] = RefEntry(label=label, kind=kind)

        # Validate and build terms
        terms_raw = raw.get("terms", [])
        if not isinstance(terms_raw, list):
            msg = f"terms must be an array, got {type(terms_raw).__name__}"
            raise ValueError(msg)

        terms: list[TermRecord] = []
        terms_index: dict[str, TermRecord] = {}
        seen_ids: set[str] = set()

        for entry in terms_raw:
            if not isinstance(entry, dict):
                msg = f"Each term entry must be an object, got {type(entry).__name__}"
                raise ValueError(msg)

            # Required fields
            term_id = entry.get("id")
            canonical = entry.get("canonical")
            term_type_raw = entry.get("type")
            anchors_raw = entry.get("anchors")

            if not isinstance(term_id, str):
                msg = f"Term entry missing or invalid 'id': {entry}"
                raise ValueError(msg)
            if term_id in seen_ids:
                msg = f"Duplicate term id: '{term_id}'"
                raise ValueError(msg)
            seen_ids.add(term_id)

            if not isinstance(canonical, str):
                msg = f"Term '{term_id}' missing or invalid 'canonical'"
                raise ValueError(msg)

            if not isinstance(term_type_raw, str):
                msg = f"Term '{term_id}' missing or invalid 'type'"
                raise ValueError(msg)
            try:
                term_type = TermType(term_type_raw)
            except ValueError:
                valid = [t.value for t in TermType]
                msg = f"Term '{term_id}' has invalid type '{term_type_raw}', must be one of {valid}"
                raise ValueError(msg) from None

            if not isinstance(anchors_raw, list) or len(anchors_raw) == 0:
                msg = f"Term '{term_id}' must have non-empty 'anchors'"
                raise ValueError(msg)

            # Validate anchors
            anchors: list[AnchorRef] = []
            for anchor in anchors_raw:
                if not isinstance(anchor, dict):
                    msg = f"Anchor in term '{term_id}' must be an object"
                    raise ValueError(msg)
                ref = anchor.get("ref")
                weight = anchor.get("weight", 1.0)
                if not isinstance(ref, str):
                    msg = f"Anchor in term '{term_id}' missing 'ref'"
                    raise ValueError(msg)
                if ref not in refs:
                    msg = f"Anchor ref '{ref}' in term '{term_id}' not found in refs"
                    raise ValueError(msg)
                if not isinstance(weight, (int, float)):
                    msg = f"Anchor weight in term '{term_id}' must be numeric"
                    raise ValueError(msg)
                anchors.append(AnchorRef(ref=ref, weight=float(weight)))

            # Optional fields
            aliases: list[str] = []
            for a in entry.get("aliases", []):
                if not isinstance(a, str):
                    msg = f"Alias in term '{term_id}' must be str"
                    raise ValueError(msg)
                aliases.append(a)

            # Match config with defaults
            match_raw = entry.get("match", {})
            if not isinstance(match_raw, dict):
                match_raw = {}
            mode_raw = match_raw.get("mode", "smart_phrase")
            case_sensitive = match_raw.get("case_sensitive", False)
            try:
                mode = MatchMode(mode_raw) if isinstance(mode_raw, str) else MatchMode.SMART_PHRASE
            except ValueError:
                valid = [m.value for m in MatchMode]
                msg = (
                    f"Term '{term_id}' has invalid match mode '{mode_raw}', must be one of {valid}"
                )
                raise ValueError(msg) from None
            match_config = MatchConfig(mode=mode, case_sensitive=bool(case_sensitive))

            # Actions with defaults based on term type
            actions_raw = entry.get("actions", {})
            if not isinstance(actions_raw, dict):
                actions_raw = {}
            default_actions = DEFAULT_ACTIONS.get(term_type, TermActions())
            retrieval_boost = actions_raw.get("retrieval_boost", default_actions.retrieval_boost)
            adams_prefilter = actions_raw.get("adams_prefilter", default_actions.adams_prefilter)
            if not isinstance(retrieval_boost, (int, float)):
                retrieval_boost = default_actions.retrieval_boost
            if not isinstance(adams_prefilter, bool):
                adams_prefilter = default_actions.adams_prefilter
            actions = TermActions(
                retrieval_boost=float(retrieval_boost),
                adams_prefilter=bool(adams_prefilter),
            )

            notes = entry.get("notes")
            if notes is not None and not isinstance(notes, str):
                notes = None

            record = TermRecord(
                id=term_id,
                canonical=canonical,
                term_type=term_type,
                anchors=anchors,
                aliases=aliases,
                match=match_config,
                actions=actions,
                notes=notes,
            )
            terms.append(record)

            # Build index and check for duplicate aliases
            # First add canonical, then check aliases against existing entries
            canonical_lower = canonical.lower()
            if canonical_lower in terms_index:
                existing = terms_index[canonical_lower]
                msg = (
                    f"Duplicate canonical '{canonical}' (case-insensitive): "
                    f"used by both '{existing.id}' and '{term_id}'"
                )
                raise ValueError(msg)
            terms_index[canonical_lower] = record

            # Now add aliases, checking for duplicates
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower in terms_index:
                    existing = terms_index[alias_lower]
                    msg = (
                        f"Duplicate alias '{alias}' (case-insensitive): "
                        f"used by both '{existing.id}' and '{term_id}'"
                    )
                    raise ValueError(msg)
                terms_index[alias_lower] = record

        term_map = TermMap(version=version, refs=refs, terms=terms)
        return cls(_term_map=term_map, _terms_index=terms_index)

    def lookup(self, term: str) -> TermRecord | None:
        """Return TermRecord for a term (case-insensitive), or None."""
        return self._terms_index.get(term.lower())

    def scan_content(self, content: str) -> list[TermMatch]:
        """Find all dictionary terms in content.

        Returns matches with frequency >= 2, sorted by descending frequency.
        Uses each term's configured match mode.
        """
        matches: list[TermMatch] = []
        matched_ids: set[str] = set()

        for record in self._term_map.terms:
            # Check canonical and all aliases
            all_forms = [record.canonical, *record.aliases]
            total_count = 0

            for form in all_forms:
                count = self._count_matches(content, form, record.match)
                total_count += count

            if total_count >= 2 and record.id not in matched_ids:
                matched_ids.add(record.id)
                matches.append(
                    TermMatch(
                        term=record.canonical,
                        term_id=record.id,
                        term_type=record.term_type,
                        anchors=list(record.anchors),
                        actions=record.actions,
                        frequency=total_count,
                    )
                )

        matches.sort(key=lambda m: (-m.frequency, m.term))
        return matches

    def _count_matches(self, content: str, term: str, config: MatchConfig) -> int:
        """Count occurrences of term in content using the specified match mode."""
        if config.mode == MatchMode.EXACT:
            # Exact match (case-sensitive or not)
            search_content = content if config.case_sensitive else content.lower()
            search_term = term if config.case_sensitive else term.lower()
            return search_content.count(search_term)

        if config.mode == MatchMode.REGEX:
            # Regex match
            flags = 0 if config.case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(term, flags)
                return len(pattern.findall(content))
            except re.error:
                return 0

        if config.mode == MatchMode.PHRASE:
            # Phrase match with word boundaries
            search_content = content if config.case_sensitive else content.lower()
            search_term = term if config.case_sensitive else term.lower()
            pattern = re.compile(r"\b" + re.escape(search_term) + r"\b")
            return len(pattern.findall(search_content))

        # SMART_PHRASE (default): phrase match with word boundaries
        search_content = content if config.case_sensitive else content.lower()
        search_term = term if config.case_sensitive else term.lower()
        pattern = re.compile(r"\b" + re.escape(search_term) + r"\b")
        return len(pattern.findall(search_content))

    @property
    def term_map(self) -> TermMap:
        """Return the underlying TermMap."""
        return self._term_map
