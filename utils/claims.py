import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_claims(answer: str) -> List[str]:
    """
    Uses an LLM to break an AI-generated answer into individual factual claims.
    Returns a list of claim strings.
    """
    prompt = f"""You are a fact-checking assistant.

Your job is to extract every factual claim from the text below.
- Write each claim as a single, standalone sentence
- Only include checkable facts (not opinions or vague statements)
- Return one claim per line, with no bullet points or numbering

Text:
{answer}

Claims:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # low temperature = more consistent, less creative
    )

    raw = response.choices[0].message.content.strip()

    # Split the response into individual claims by line
    claims = [line.strip() for line in raw.splitlines() if line.strip()]

    return claims
