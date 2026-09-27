import pytest

from pipecat_sprag.common import ATTRIBUTION_HEADERS, __version__, resolve_api_key


def test_explicit_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPRAG_API_KEY", "from-env")
    assert resolve_api_key("explicit") == "explicit"


def test_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPRAG_API_KEY", "from-env")
    assert resolve_api_key(None) == "from-env"


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRAG_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-ours")
    with pytest.raises(ValueError, match="SPRAG_API_KEY"):
        resolve_api_key(None)


def test_attribution_header_names_the_package_version() -> None:
    assert ATTRIBUTION_HEADERS == {"X-Sprag-Integration": f"pipecat/{__version__}"}
