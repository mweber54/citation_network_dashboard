from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import ceil
from typing import Dict, List, Tuple


Pair = Tuple[int, int]
Triple = Tuple[int, int, int]


def mine_triples_for_year(
    transactions: List[List[int]],
    pair_metrics: Dict[Pair, Dict[str, float]],
    n_year: int,
    minsup_frac: float,
    min_count: int,
    min_lift: float,
) -> Dict[Triple, Dict[str, float]]:
    if n_year <= 0:
        return {}

    pair_set = set(pair_metrics.keys())
    triple_counts: Counter[Triple] = Counter()

    for tx in transactions:
        if len(tx) < 3:
            continue
        for triple in combinations(tx, 3):
            a, b, c = triple
            p1 = (a, b)
            p2 = (a, c)
            p3 = (b, c)
            if p1 in pair_set and p2 in pair_set and p3 in pair_set:
                triple_counts[triple] += 1

    min_support_count = max(min_count, ceil(minsup_frac * n_year))
    strong_triples: Dict[Triple, Dict[str, float]] = {}

    for triple, count in triple_counts.items():
        if count < min_support_count:
            continue
        a, b, c = triple
        p1 = (a, b)
        p2 = (a, c)
        p3 = (b, c)
        l1 = pair_metrics.get(p1, {}).get("lift", 0.0)
        l2 = pair_metrics.get(p2, {}).get("lift", 0.0)
        l3 = pair_metrics.get(p3, {}).get("lift", 0.0)
        lift_min = min(l1, l2, l3)
        if lift_min < min_lift:
            continue
        strong_triples[triple] = {
            "sup_count": float(count),
            "supp_frac": count / n_year,
            "lift_min": lift_min,
        }

    return strong_triples
