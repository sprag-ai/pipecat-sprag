# Changelog

## 0.2.0.dev1

### Added

- `examples/demo.py`: the voice agent behind a Sprag-branded browser client.
- `SpragTurnDetector`: Sprag's turn model as a processor that proposes the user's turn boundaries, for `ExternalUserTurnStrategies`.
- `SpragTurnAnalyzer`: Sprag's turn model as a streaming turn analyzer.
- `SpragAgentTextObserver`: gives either one the played words of the agent's last reply as context.

### Changed

- Requires `pipecat-ai` 1.12 or later.

## 0.1.0

### Added

- `SpragSTTService`: streaming transcription with Rhythm.
- `SpragLLMService`: Symphony, with function calling and token usage metrics.
- `SpragTTSService`: streaming speech with Chorus.
- `examples/voice_agent.py`: a voice agent running entirely on Sprag.
