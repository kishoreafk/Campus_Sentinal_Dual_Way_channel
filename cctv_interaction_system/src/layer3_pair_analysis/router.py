"""Router — splits active tracklets into pairs (interaction) and singles (individual).

A tracklet can appear in BOTH a pair AND the individual branch (e.g., the
person is engaged in an interaction but we also want their individual
action label). However, the spec routes pairs exclusively to interaction,
so we route "singles" = tracklets NOT in any pair.
"""

from __future__ import annotations

from typing import List, Set

from src.common.schemas import PersonPair, Tracklet


class Router:
    """Splits tracklets into pairs (interaction) and singles (individual)."""

    def route(
        self,
        tracklets: List[Tracklet],
        pairs: List[PersonPair],
    ) -> tuple[List[PersonPair], List[Tracklet]]:
        paired_ids: Set[int] = set()
        for p in pairs:
            paired_ids.add(p.track_id_a)
            paired_ids.add(p.track_id_b)
        singles = [t for t in tracklets if t.track_id not in paired_ids]
        return pairs, singles
