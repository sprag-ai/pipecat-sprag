# Changelog

## Unreleased

### Fixed

- `SpragSTTService` marks each committed turn's transcript as finalized, so a turn analyzer such as `SpragTurnAnalyzer` or Smart Turn ends the turn when the transcript arrives instead of waiting for the STT timeout.
- `SpragTurnDetector` and `SpragTurnAnalyzer` read caller audio as one stream of bytes, so a chunk that ends in the middle of a sample no longer drops the session at input rates other than 16 kHz.

## 0.2.0

### Added

- `SpragTurnDetector`: Sprag's turn model as a processor that proposes the user's turn boundaries, for `ExternalUserTurnStrategies`, reporting `TurnMetricsData` for each turn.
- `SpragTurnAnalyzer`: Sprag's turn model as a streaming turn analyzer.
- `SpragAgentTextObserver`: gives either one the played words of the agent's last reply as context.
- `SpragTurnParams`: the session's settings, `eot_threshold`, `eager_eot_threshold`, `min_silence_ms`, and `eot_timeout_ms`, held to the server's bounds. Other values raise a `ValidationError` at construction.
- Event handlers on the detector and analyzer, each receiving a typed model: `on_turn_started`, `on_turn_update`, `on_eager_end`, `on_turn_resumed`, `on_turn_ended`, and `on_session_closed`.
- `usage` on the detector and analyzer: the model invocations in the latest connection's `session.closed`, or `None` while that connection is open or when it closed without one.
- `SessionConfigRefusedError`, the error a server refusal of the session's params reports, and `SessionRejectedError`, the error a rejected connection reports when a retry cannot clear it, such as an invalid API key or an unknown model. Neither is retried; network failures are.
- `SpragSTTService(commit_on_turn_end=True)`: transcribes each turn when `SpragTurnDetector` ends it, so turn detection needs no VAD.
- `examples/turn_detection.py`: a voice agent whose turns are decided by Sprag's turn model.
- `examples/demo.py`: the voice agent behind a Sprag-branded browser client, with turns decided by Sprag's turn model.

### Changed

- Requires `pipecat-ai` 1.12 or later.

## 0.1.0

### Added

- `SpragSTTService`: streaming transcription with Rhythm.
- `SpragLLMService`: Symphony, with function calling and token usage metrics.
- `SpragTTSService`: streaming speech with Chorus.
- `examples/voice_agent.py`: a voice agent running entirely on Sprag.
