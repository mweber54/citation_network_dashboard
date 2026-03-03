from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import ceil
from typing import Dict, List, Tuple


Pair = Tuple[int, int]


def mine_pairs_for_year(
    transactions: List[List[int]],
    single_counts: Dict[int, int],
    n_year: int,
    minsup_frac: float,
    min_count: int,
    min_lift: float,
    eps: float = 1e-12,
) -> Dict[str, Dict[Pair, Dict[str, float]]]:
    pair_counts: Counter[Pair] = Counter()
    for tx in transactions:
        if len(tx) < 2:
            continue
        for a, b in combinations(tx, 2):
            pair_counts[(a, b)] += 1

    min_support_count = max(min_count, ceil(minsup_frac * n_year)) if n_year > 0 else min_count

    frequent_pairs: Dict[Pair, Dict[str, float]] = {}
    strong_pairs: Dict[Pair, Dict[str, float]] = {}

    for (a, b), count in pair_counts.items():
        if count < min_support_count:
            continue
        supp_ab = count / n_year if n_year > 0 else 0.0
        supp_a = (single_counts.get(a, 0) / n_year) if n_year > 0 else 0.0
        supp_b = (single_counts.get(b, 0) / n_year) if n_year > 0 else 0.0
        lift = supp_ab / (supp_a * supp_b + eps) if n_year > 0 else 0.0

        metrics = {
            "sup_count": float(count),
            "supp_frac": supp_ab,
            "lift": lift,
        }
        frequent_pairs[(a, b)] = metrics
        if lift >= min_lift:
            strong_pairs[(a, b)] = metrics

    return {
        "frequent_pairs": frequent_pairs,
        "strong_pairs": strong_pairs,
    }
