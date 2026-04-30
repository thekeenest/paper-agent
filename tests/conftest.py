"""
Shared pytest fixtures.

API keys and configuration are read from the project .env file.
No keys are hardcoded here.  All tests are OFFLINE — external services
(OpenAlex, ROR, Semantic Scholar, OpenAI) are not called during the
structural test suite.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from the project root so OPENAI_API_KEY is available for import guards.
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")


@pytest.fixture(autouse=True)
def _require_openai_key() -> None:
    """Skip the whole session if the key is missing, with a clear message."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip(
            "OPENAI_API_KEY not set. Add it to .env (see .env.example).",
            allow_module_level=True,
        )
