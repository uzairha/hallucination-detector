import os

# The OpenAI clients are created at import time and need a key to exist.
# Tests mock every API call, so a dummy value is enough.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
