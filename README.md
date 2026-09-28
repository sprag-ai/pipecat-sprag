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

Each service drops into a Pipecat pipeline in place of any other STT, LLM, or TTS service:

```python
from pipecat_sprag.llm import SpragLLMService
from pipecat_sprag.stt import SpragSTTService
from pipecat_sprag.tts import SpragTTSService

stt = SpragSTTService()
llm = SpragLLMService()
tts = SpragTTSService(settings=SpragTTSService.Settings(voice="wade"))

pipeline = Pipeline(
    [transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator]
)
```

- **`SpragSTTService`** streams audio to Rhythm over Sprag's realtime API. By default your pipeline's VAD decides when a turn ends; pass `turn_detection={"type": "server_vad"}` to let Sprag detect turns instead.
- **`SpragLLMService`** runs Symphony through Sprag's chat completions API, with function calling and token usage metrics.
- **`SpragTTSService`** streams Chorus speech over Sprag's realtime API, one sentence at a time. It defaults to the `chorus-clone` model and the `wade` voice; the available voices are listed in [voice presets](https://sprag.ai/docs/tts/voices/presets).

Each service takes `settings=<Service>.Settings(...)` for its model and other options, and `api_key=` if you don't use the `SPRAG_API_KEY` environment variable.

## Example

[`examples/voice_agent.py`](examples/voice_agent.py) is a complete voice agent running on Rhythm, Symphony, and Chorus. It uses Pipecat's development runner:

```bash
uv sync --group examples
SPRAG_API_KEY=... uv run examples/voice_agent.py
```

Then open the printed URL in your browser and talk to it.

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
