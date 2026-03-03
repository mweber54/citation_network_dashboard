from __future__ import annotations

import math
from statistics import mean
from typing import Dict, List, Sequence, Tuple


Itemset = Tuple[int, ...]


def _mean_non_nan(values: Sequence[float]) -> float:
    filtered = [v for v in values if not math.isnan(v)]
    return mean(filtered) if filtered else 0.0


def predict_residual_2026(
    itemsets: List[Itemset],
    years: List[int],
    residual_by_itemset: Dict[Itemset, List[float]],
    supp_series: Dict[Itemset, List[float]],
    residual_window_years: Tuple[int, int] = (2021, 2025),
) -> List[Dict[str, float]]:
    y0, y1 = residual_window_years
    year_to_idx = {year: idx for idx, year in enumerate(years)}

    results: List[Dict[str, float]] = []
    for itemset in itemsets:
        residuals = residual_by_itemset[itemset]
        supp = supp_series[itemset]

        idxs = [year_to_idx[y] for y in range(y0, y1 + 1) if y in year_to_idx]
        residual_slice = [residuals[i] for i in idxs]
        r_pred = _mean_non_nan(residual_slice)

        recent_idxs = [year_to_idx[y] for y in range(max(y0, y1 - 2), y1 + 1) if y in year_to_idx]
        recent_support = _mean_non_nan([supp[i] for i in recent_idxs])

        score = r_pred
        score_weighted = r_pred * (1.0 + recent_support)

        results.append(
            {
                "itemset": itemset,
                "pred_residual_2026": r_pred,
                "recent_support": recent_support,
                "score": score,
                "score_weighted": score_weighted,
            }
        )

    results.sort(key=lambda x: (x["score"], x["recent_support"]), reverse=True)
    return results


def evaluate_next_year_alignment(
    itemsets: List[Itemset],
    years: List[int],
    residual_by_itemset: Dict[Itemset, List[float]],
    k: int,
    start_year: int,
    end_year: int,
) -> List[Dict[str, float]]:
    year_to_idx = {year: idx for idx, year in enumerate(years)}
    out: List[Dict[str, float]] = []

    for year in range(start_year, end_year + 1):
        if year not in year_to_idx or (year + 1) not in year_to_idx:
            continue
        i = year_to_idx[year]
        j = year_to_idx[year + 1]

        pred_rank = []
        real_rank = []
        for itemset in itemsets:
            r_t = residual_by_itemset[itemset][i]
            r_next = residual_by_itemset[itemset][j]
            pred_rank.append((itemset, r_t if not math.isnan(r_t) else -1e9))
            real_rank.append((itemset, r_next if not math.isnan(r_next) else -1e9))

        pred_rank.sort(key=lambda x: x[1], reverse=True)
        real_rank.sort(key=lambda x: x[1], reverse=True)

        pred_top = [x[0] for x in pred_rank[:k]]
        real_top = [x[0] for x in real_rank[:k]]
        real_top_set = set(real_top)

        hits = sum(1 for it in pred_top if it in real_top_set)
        precision = hits / k if k > 0 else 0.0

        dcg = 0.0
        idcg = 0.0
        for rank, it in enumerate(pred_top, start=1):
            rel = 1.0 if it in real_top_set else 0.0
            dcg += rel / math.log2(rank + 1)
        for rank in range(1, min(k, len(real_top)) + 1):
            idcg += 1.0 / math.log2(rank + 1)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        out.append(
            {
                "year": float(year),
                "precision_at_k": precision,
                "ndcg_at_k": ndcg,
            }
        )

    return out
