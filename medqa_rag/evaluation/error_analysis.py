from __future__ import annotations


def summarize_error_analysis(
    baseline_rows: list[dict],
    rag_rows: list[dict],
    prediction_key_baseline: str = "prediction",
    prediction_key_rag: str = "prediction",
) -> dict:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    rag_by_id = {row["id"]: row for row in rag_rows}
    shared_ids = sorted(set(baseline_by_id) & set(rag_by_id))

    helped = []
    hurt = []
    unchanged = []
    for record_id in shared_ids:
        baseline_row = baseline_by_id[record_id]
        rag_row = rag_by_id[record_id]
        baseline_correct = _is_correct(baseline_row, prediction_key_baseline)
        rag_correct = _is_correct(rag_row, prediction_key_rag)
        if not baseline_correct and rag_correct:
            helped.append(_pair_summary(baseline_row, rag_row))
        elif baseline_correct and not rag_correct:
            hurt.append(_pair_summary(baseline_row, rag_row))
        else:
            unchanged.append(_pair_summary(baseline_row, rag_row))

    return {
        "total": len(shared_ids),
        "rag_help": len(helped),
        "rag_hurt": len(hurt),
        "unchanged": len(unchanged),
        "helped_examples": helped[:10],
        "hurt_examples": hurt[:10],
    }


def _pair_summary(baseline_row: dict, rag_row: dict) -> dict:
    return {
        "id": baseline_row["id"],
        "answer": baseline_row.get("answer"),
        "baseline_pred": baseline_row.get("prediction"),
        "rag_pred": rag_row.get("prediction"),
        "question": baseline_row.get("question"),
        "retrieved_ids": rag_row.get("retrieved_ids", []),
    }


def _is_correct(row: dict, prediction_key: str) -> bool:
    return str(row.get(prediction_key) or "").upper() == str(row.get("answer") or "").upper()
