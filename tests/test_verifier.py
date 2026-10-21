from types import SimpleNamespace

import pytest

from utils import verifier

EVIDENCE = [{"text": "Transformers use self-attention instead of recurrence."}]


@pytest.fixture
def fake_llm(monkeypatch):
    """
    Stubs out retrieval and the OpenAI call. Set fake_llm.reply to control
    what the "model" returns; fake_llm.prompts records what was sent.
    """
    state = SimpleNamespace(reply="", prompts=[])

    def fake_create(**kwargs):
        state.prompts.append(kwargs["messages"][0]["content"])
        message = SimpleNamespace(content=state.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(verifier, "retrieve", lambda claim, top_k=3: EVIDENCE)
    monkeypatch.setattr(verifier.client.chat.completions, "create", fake_create)
    return state


def test_parses_verdict_explanation_and_correct(fake_llm):
    fake_llm.reply = (
        "Verdict: Unsupported\n"
        "Explanation: The evidence says transformers do not use recurrence.\n"
        "Correct: Transformers use self-attention instead of recurrent layers."
    )

    result = verifier.verify_claim("Transformers rely on recurrent layers.")

    assert result == {
        "claim": "Transformers rely on recurrent layers.",
        "verdict": "Unsupported",
        "explanation": "The evidence says transformers do not use recurrence.",
        "correct": "Transformers use self-attention instead of recurrent layers.",
        "evidence": ["Transformers use self-attention instead of recurrence."],
    }
    # The retrieved evidence and the claim both make it into the prompt
    assert EVIDENCE[0]["text"] in fake_llm.prompts[0]
    assert "Claim: Transformers rely on recurrent layers." in fake_llm.prompts[0]


def test_parses_weakly_supported_with_surrounding_whitespace(fake_llm):
    fake_llm.reply = "\n  Verdict:   Weakly Supported  \nExplanation: Partly covered.\nCorrect: Some fact.\n"

    result = verifier.verify_claim("claim")

    assert result["verdict"] == "Weakly Supported"
    assert result["explanation"] == "Partly covered."
    assert result["correct"] == "Some fact."


def test_fallback_when_format_is_off(fake_llm):
    fake_llm.reply = "I think this claim is probably true, but I'm not sure."

    result = verifier.verify_claim("claim")

    # No recognised lines: default to Unsupported, keep the raw reply as the explanation
    assert result["verdict"] == "Unsupported"
    assert result["explanation"] == "I think this claim is probably true, but I'm not sure."
    assert result["correct"] == ""


def test_missing_correct_line_leaves_correct_empty(fake_llm):
    fake_llm.reply = "Verdict: Supported\nExplanation: Matches the evidence."

    result = verifier.verify_claim("claim")

    assert result["verdict"] == "Supported"
    assert result["explanation"] == "Matches the evidence."
    assert result["correct"] == ""
