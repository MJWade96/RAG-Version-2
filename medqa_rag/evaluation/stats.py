from __future__ import annotations

import math
import random


def accuracy(rows: list[dict], prediction_key: str = "prediction") -> float:
    if not rows:
        return 0.0
    scored = [row for row in rows if row.get("answer")]
    if not scored:
        return 0.0
    correct = sum(1 for row in scored if str(row.get(prediction_key) or "").upper() == str(row.get("answer") or "").upper())
    return correct / len(scored)


def contingency_counts(
    baseline_rows: list[dict],
    rag_rows: list[dict],
    prediction_key_baseline: str = "prediction",
    prediction_key_rag: str = "prediction",
) -> tuple[int, int, int, int]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    rag_by_id = {row["id"]: row for row in rag_rows}
    ids = sorted(set(baseline_by_id) & set(rag_by_id))

    n00 = n01 = n10 = n11 = 0
    for record_id in ids:
        baseline_correct = _is_correct(baseline_by_id[record_id], prediction_key_baseline)
        rag_correct = _is_correct(rag_by_id[record_id], prediction_key_rag)
        if baseline_correct and rag_correct:
            n11 += 1
        elif baseline_correct and not rag_correct:
            n10 += 1
        elif not baseline_correct and rag_correct:
            n01 += 1
        else:
            n00 += 1
    return n00, n01, n10, n11


def run_mcnemar(n00: int, n01: int, n10: int, n11: int, correction: bool = True) -> float:
    del n00, n11
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    numerator = abs(n01 - n10) - 1 if correction else abs(n01 - n10)
    chi_square = (numerator * numerator) / discordant
    return math.erfc(math.sqrt(chi_square / 2.0))


def bootstrap_diff(
    baseline_correct: list[bool],
    rag_correct: list[bool],
    n: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    if len(baseline_correct) != len(rag_correct):
        raise ValueError("Baseline and RAG arrays must have the same length.")
    if not baseline_correct:
        return 0.0, 0.0

    rng = random.Random(seed)
    size = len(baseline_correct)
    diffs: list[float] = []
    for _ in range(n):
        indices = [rng.randrange(size) for _ in range(size)]
        baseline_mean = sum(baseline_correct[index] for index in indices) / size
        rag_mean = sum(rag_correct[index] for index in indices) / size
        diffs.append(rag_mean - baseline_mean)
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return lo, hi


def correctness_vector(rows: list[dict], prediction_key: str = "prediction") -> list[bool]:
    return [_is_correct(row, prediction_key) for row in rows if row.get("answer")]


def _is_correct(row: dict, prediction_key: str) -> bool:
    return str(row.get(prediction_key) or "").upper() == str(row.get("answer") or "").upper()
