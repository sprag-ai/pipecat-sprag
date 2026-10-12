# pipecat-sprag

[Sprag](https://sprag.ai/) speech and language models for [Pipecat](https://github.com/pipecat-ai/pipecat).

This integration is maintained by Sprag.

| Service | Import | Model |
| --- | --- | --- |
| `SpragSTTService` | `pipecat_sprag.stt` | Rhythm (speech to text) |
| `SpragLLMService` | `pipecat_sprag.llm` | Symphony (LLM) |
| `SpragTTSService` | `pipecat_sprag.tts` | Chorus (text to speech) |

## Installation

```bash
uv add pipecat-sprag
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

- **`SpragSTTService`** streams audio to Rhythm over Sprag's streaming transcription socket (`wss://api.sprag.ai/v1/audio/transcriptions`). By default your pipeline's VAD decides when a turn ends; pass `turn_detection={"type": "server_vad"}` to let Sprag detect turns instead.
- **`SpragLLMService`** runs Symphony through Sprag's chat completions API, with function calling and token usage metrics.
- **`SpragTTSService`** streams Chorus speech over Sprag's streaming speech socket (`wss://api.sprag.ai/v1/audio/speech`), one sentence at a time. It defaults to the `chorus` model and the `wade` voice; the available voices are listed in [voice presets](https://sprag.ai/docs/tts/voices/presets).

Each service takes `settings=<Service>.Settings(...)` for its model and other options, and `api_key=` if you don't use the `SPRAG_API_KEY` environment variable.

## Example

[`examples/voice_agent.py`](https://github.com/sprag-ai/pipecat-sprag/blob/main/examples/voice_agent.py) is a complete voice agent running on Rhythm, Symphony, and Chorus. It uses Pipecat's development runner:

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
uv run mypy src
uv run pytest
```

Live tests call api.sprag.ai and are skipped by default: `SPRAG_API_KEY=... uv run pytest -m live`.

## License

BSD 2-Clause. See [LICENSE](https://github.com/sprag-ai/pipecat-sprag/blob/main/LICENSE).
