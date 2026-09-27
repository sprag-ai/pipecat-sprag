# pipecat-sprag

[Sprag](https://sprag.ai/) speech and language models for [Pipecat](https://github.com/pipecat-ai/pipecat).

This integration is maintained by Sprag.

| Package | Service | Model |
| --- | --- | --- |
| `pipecat-sprag-stt` | `SpragSTTService` | Rhythm (speech to text) |
| `pipecat-sprag-llm` | `SpragLLMService` | Symphony (LLM) |
| `pipecat-sprag-tts` | `SpragTTSService` | Chorus (text to speech) |

## Installation

Install the services your pipeline uses:

```bash
uv add pipecat-sprag-stt pipecat-sprag-llm pipecat-sprag-tts
```

Each service reads your API key from the `SPRAG_API_KEY` environment variable, or takes it as `api_key=`. You can generate a key from [API keys](https://sprag.ai/app/api-keys) in the Sprag app.

## Usage

Usage examples land with the services.

## Compatibility

Requires `pipecat-ai>=1.8,<2`. Tested with Pipecat v1.12.0.

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check
uv run mypy packages
uv run pytest
```

Live tests call api.sprag.ai and are skipped by default: `SPRAG_API_KEY=... uv run pytest -m live`.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
