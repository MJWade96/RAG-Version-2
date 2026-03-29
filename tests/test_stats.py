from medqa_rag.evaluation.stats import bootstrap_diff, contingency_counts, run_mcnemar


def test_stats_functions_return_valid_outputs():
    baseline = [
        {"id": "1", "answer": "A", "prediction": "A"},
        {"id": "2", "answer": "B", "prediction": "A"},
    ]
    rag = [
        {"id": "1", "answer": "A", "prediction": "A"},
        {"id": "2", "answer": "B", "prediction": "B"},
    ]
    counts = contingency_counts(baseline, rag)
    assert counts == (0, 1, 0, 1)
    pvalue = run_mcnemar(*counts)
    assert 0.0 <= pvalue <= 1.0
    lo, hi = bootstrap_diff([True, False], [True, True], n=200, seed=7)
    assert lo <= hi
