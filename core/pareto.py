"""Pareto Archive and Hypervolume Calculation for ATLAS-MERGE.

Maintains non-dominated candidates across the multi-objective search space
and computes 2D hypervolume indicators.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import datetime

@dataclass
class ParetoCandidate:
    """Represents an evaluated model merge candidate."""
    candidate_id: str
    episode: int
    math_score: float
    code_score: float
    retention_score: float
    reward: float
    recipe: Dict[str, Dict[str, Any]]
    timestamp: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def dominates(cand_a: ParetoCandidate, cand_b: ParetoCandidate, eps: float = 1e-6) -> bool:
    """Check if candidate A dominates candidate B in Pareto sense.
    
    A dominates B iff A is at least as good as B in all objectives
    and strictly better in at least one objective.
    Objectives: math_score (maximize), code_score (maximize).
    """
    a_math, a_code = cand_a.math_score, cand_a.code_score
    b_math, b_code = cand_b.code_score, cand_b.code_score
    # Correct mapping:
    b_math = cand_b.math_score
    b_code = cand_b.code_score

    not_worse = (a_math >= b_math - eps) and (a_code >= b_code - eps)
    strictly_better = (a_math > b_math + eps) or (a_code > b_code + eps)

    return not_worse and strictly_better


def compute_2d_hypervolume(
    candidates: List[ParetoCandidate],
    reference_point: Tuple[float, float] = (0.0, 0.0),
) -> float:
    """Calculate mathematically exact 2D hypervolume enclosed by non-dominated front.
    
    reference_point: (ref_math, ref_code), default (0.0, 0.0)
    """
    ref_x, ref_y = reference_point

    # Filter points strictly better than reference point
    valid_pts = [
        (c.math_score, c.code_score)
        for c in candidates
        if c.math_score > ref_x and c.code_score > ref_y
    ]

    if not valid_pts:
        return 0.0

    # Sort by math_score ascending
    valid_pts.sort(key=lambda pt: pt[0])

    # Filter to ensure non-dominated staircase (y must be strictly decreasing)
    filtered_pts: List[Tuple[float, float]] = []
    max_y = -float("inf")
    # Traverse from right to left to keep only upper-right frontier
    for x, y in reversed(valid_pts):
        if y > max_y:
            filtered_pts.append((x, y))
            max_y = y
    filtered_pts.reverse()

    # Calculate exact disjoint rectangular decomposition
    hypervolume = 0.0
    prev_x = ref_x

    for x, y in filtered_pts:
        width = x - prev_x
        height = y - ref_y
        hypervolume += width * height
        prev_x = x

    return float(max(0.0, hypervolume))


class ParetoArchive:
    """Maintains the set of non-dominated merge candidates and tracks hypervolume trajectory."""

    def __init__(self, reference_point: Tuple[float, float] = (0.0, 0.0), max_size: int = 200):
        self.reference_point = reference_point
        self.max_size = max_size
        self.non_dominated_candidates: List[ParetoCandidate] = []
        self.all_evaluated_candidates: List[ParetoCandidate] = []
        self.hypervolume_history: List[Tuple[int, float]] = []

    def add_candidate(self, candidate: ParetoCandidate) -> bool:
        """Register a new candidate.
        
        Returns True if candidate enters the non-dominated Pareto archive, False otherwise.
        """
        self.all_evaluated_candidates.append(candidate)

        # Check if candidate is dominated by any existing member
        for existing in self.non_dominated_candidates:
            if dominates(existing, candidate):
                # Candidate is dominated, cannot enter archive
                current_hv = self.get_current_hypervolume()
                self.hypervolume_history.append((candidate.episode, current_hv))
                return False

        # Candidate is not dominated: remove all existing candidates it dominates
        new_archive = [
            existing for existing in self.non_dominated_candidates
            if not dominates(candidate, existing)
        ]

        # Avoid exact duplicate points
        duplicate = any(
            abs(existing.math_score - candidate.math_score) < 1e-4
            and abs(existing.code_score - candidate.code_score) < 1e-4
            for existing in new_archive
        )

        if not duplicate:
            new_archive.append(candidate)

        self.non_dominated_candidates = new_archive

        # Update hypervolume history
        current_hv = self.get_current_hypervolume()
        self.hypervolume_history.append((candidate.episode, current_hv))
        return True

    def get_current_hypervolume(self) -> float:
        return compute_2d_hypervolume(self.non_dominated_candidates, self.reference_point)

    def get_archive(self) -> List[ParetoCandidate]:
        # Return sorted by math_score ascending
        return sorted(self.non_dominated_candidates, key=lambda c: c.math_score)

    def clear(self) -> None:
        self.non_dominated_candidates = []
        self.all_evaluated_candidates = []
        self.hypervolume_history = []

