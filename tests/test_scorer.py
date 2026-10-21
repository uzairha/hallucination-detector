from utils.scorer import score_results


def claims(*verdicts):
    return [{"verdict": v} for v in verdicts]


def test_all_supported():
    result = score_results(claims("Supported", "Supported", "Supported"))
    assert result["score"] == 1.0
    assert result["verdict"] == "Trustworthy"
    assert result["counts"] == {"Supported": 3, "Weakly Supported": 0, "Unsupported": 0}
    assert result["total"] == 3


def test_all_unsupported():
    result = score_results(claims("Unsupported", "Unsupported"))
    assert result["score"] == 0.0
    assert result["verdict"] == "Likely Hallucinated"
    assert result["counts"]["Unsupported"] == 2


def test_mixed():
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    result = score_results(claims("Supported", "Weakly Supported", "Unsupported"))
    assert result["score"] == 0.5
    assert result["verdict"] == "Partially Trustworthy"
    assert result["counts"] == {"Supported": 1, "Weakly Supported": 1, "Unsupported": 1}
    assert result["total"] == 3


def test_unknown_verdict_counts_as_unsupported():
    result = score_results([{"verdict": "Maybe"}, {}, {"verdict": "Supported"}])
    assert result["counts"] == {"Supported": 1, "Weakly Supported": 0, "Unsupported": 2}
    assert result["score"] == round(1 / 3, 3)
    assert result["verdict"] == "Likely Hallucinated"


def test_empty_list():
    result = score_results([])
    assert result == {"score": 0.0, "verdict": "No Claims Found", "counts": {}, "total": 0}
