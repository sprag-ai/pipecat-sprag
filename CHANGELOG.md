# Changelog

## 0.2.0.dev1

### Added

- `SpragSTTService(commit_on_turn_end=True)`: transcribes each turn when `SpragTurnDetector` ends it, so turn detection needs no VAD.
- `examples/demo.py`: the voice agent behind a Sprag-branded browser client, with turns decided by Sprag's turn model.
- `examples/turn_detection.py`: a voice agent whose turns are decided by Sprag's turn model.
- `SpragTurnDetector`: Sprag's turn model as a processor that proposes the user's turn boundaries, for `ExternalUserTurnStrategies`, reporting `TurnMetricsData` for each turn.
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
