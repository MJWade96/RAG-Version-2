"""Evaluation harnesses, statistics, and error analysis."""

from medqa_rag.evaluation.error_analysis import summarize_error_analysis
from medqa_rag.evaluation.harness import evaluate_baseline, evaluate_rag, write_results
from medqa_rag.evaluation.stats import accuracy, bootstrap_diff, run_mcnemar

__all__ = [
    "accuracy",
    "bootstrap_diff",
    "evaluate_baseline",
    "evaluate_rag",
    "run_mcnemar",
    "summarize_error_analysis",
    "write_results",
]
