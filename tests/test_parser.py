from medqa_rag.inference.parser import parse_answer_letter


def test_parser_handles_direct_output():
    assert parse_answer_letter("B") == "B"


def test_parser_handles_cot_output():
    text = "Reasoning about the options. Final answer: C"
    assert parse_answer_letter(text) == "C"
