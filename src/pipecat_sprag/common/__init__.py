"""Configuration shared by the Sprag Pipecat services."""

import os
from importlib.metadata import version

__version__ = version("pipecat-sprag")

SPRAG_BASE_URL = "https://api.sprag.ai/v1"
SPRAG_REALTIME_URL = "wss://api.sprag.ai/v1/realtime"
ATTRIBUTION_HEADERS = {"X-Sprag-Integration": f"pipecat/{__version__}"}


def resolve_api_key(api_key: str | None) -> str:
    """Return the given API key, or ``SPRAG_API_KEY`` from the environment.

    Raises:
        ValueError: If neither is set.
    """
    resolved = api_key or os.environ.get("SPRAG_API_KEY")
    if not resolved:
        raise ValueError(
            "SPRAG_API_KEY is required, either as the api_key argument or the "
            "SPRAG_API_KEY environment variable"
        )
    return resolved
