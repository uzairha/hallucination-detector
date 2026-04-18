from typing import List, Dict


WEIGHTS = {
    "Supported": 1.0,
    "Weakly Supported": 0.5,
    "Unsupported": 0.0,
}


def score_results(verified_claims: List[Dict]) -> Dict:
    """
    Computes an overall hallucination score from a list of verified claims.

    Returns a dict with:
      - score: float 0.0–1.0 (1.0 = fully supported, 0.0 = fully hallucinated)
      - verdict: "Trustworthy" | "Partially Trustworthy" | "Likely Hallucinated"
      - counts: breakdown of each verdict type
      - total: total number of claims evaluated
    """
    if not verified_claims:
        return {
            "score": 0.0,
            "verdict": "No Claims Found",
            "counts": {},
            "total": 0,
        }

    counts = {"Supported": 0, "Weakly Supported": 0, "Unsupported": 0}
    for result in verified_claims:
        v = result.get("verdict", "Unsupported")
        if v in counts:
            counts[v] += 1
        else:
            counts["Unsupported"] += 1

    total = len(verified_claims)
    raw_score = sum(WEIGHTS[v] * counts[v] for v in counts) / total

    if raw_score >= 0.75:
        verdict = "Trustworthy"
    elif raw_score >= 0.4:
        verdict = "Partially Trustworthy"
    else:
        verdict = "Likely Hallucinated"

    return {
        "score": round(raw_score, 3),
        "verdict": verdict,
        "counts": counts,
        "total": total,
    }
