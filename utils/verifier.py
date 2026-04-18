import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from utils.retriever import retrieve

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def verify_claim(claim: str) -> Dict:
    """
    Retrieves relevant evidence for a claim and asks the LLM to judge it.
    Returns a dict with the claim, verdict, explanation, and evidence used.
    """
    # Get the most relevant chunks for this claim
    evidence_chunks = retrieve(claim, top_k=3)
    evidence_text = "\n\n".join([c["text"] for c in evidence_chunks])

    prompt = f"""You are a fact-checking assistant.

Using only the evidence provided below, evaluate whether the claim is supported.

Respond in exactly this format:
Verdict: <Supported | Weakly Supported | Unsupported>
Explanation: <one sentence explaining why>

Evidence:
{evidence_text}

Claim: {claim}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Parse the verdict and explanation from the response
    verdict = "Unsupported"
    explanation = raw

    for line in raw.splitlines():
        if line.startswith("Verdict:"):
            verdict = line.replace("Verdict:", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()

    return {
        "claim": claim,
        "verdict": verdict,
        "explanation": explanation,
        "evidence": [c["text"] for c in evidence_chunks]
    }


def verify_all_claims(claims: List[str]) -> List[Dict]:
    """
    Runs verify_claim on every claim and returns the full list of results.
    """
    results = []
    for claim in claims:
        result = verify_claim(claim)
        results.append(result)
        print(f"[{result['verdict']}] {claim}")
    return results
