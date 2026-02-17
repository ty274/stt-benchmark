# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STT Benchmark is a Speech-to-Text benchmarking framework that measures TTFS (Time To Final Segment) latency and Semantic WER (Word Error Rate) accuracy across 20+ STT providers. Built on Pipecat, it uses synthetic audio playback through real STT service pipelines.

## Commands

```bash
uv sync                                          # Install dependencies
uv sync --group dev                              # Install with dev dependencies
uv run stt-benchmark download --num-samples 100  # Download audio samples
uv run stt-benchmark run --services deepgram     # Run benchmarks
uv run stt-benchmark ground-truth                # Generate ground truth (Gemini)
uv run stt-benchmark wer                         # Calculate semantic WER (Claude)
uv run stt-benchmark report                      # View results

ruff check src/                                  # Lint
ruff format src/                                 # Format
mypy src/                                        # Type check
pytest tests/                                    # Run tests (asyncio_mode=auto)
```

## Architecture

The system is a multi-stage pipeline:

1. **Download** — Fetch audio from HuggingFace (`pipecat-ai/smart-turn-data-v3.1-train`), convert to 16-bit PCM at 16kHz
2. **Benchmark** — Play audio through a Pipecat pipeline with a real STT service, collecting TTFS and transcriptions via observers
3. **Ground Truth** — Generate reference transcriptions via Gemini with anti-hallucination prompting
4. **Semantic WER** — Claude evaluates transcription accuracy, ignoring differences that don't affect LLM understanding
5. **Report** — Aggregate statistics (mean, median, P50/P90/P95/P99) and generate comparison tables

### Core Pipeline (`src/stt_benchmark/pipeline/`)

`BenchmarkRunner` orchestrates each sample through a Pipecat pipeline:
- `SyntheticInputTransport` plays audio at real-time pace with Silero VAD for speech boundary detection
- The STT service (created via factory pattern from `services.py`) processes the audio stream
- Two observers capture results: `MetricsCollectorObserver` (TTFS from MetricsFrame) and `TranscriptionCollectorObserver` (final transcript from TranscriptionFrame)

### TTFS Measurement

TTFS = final_transcript_time − speech_end_time, where speech_end_time = VADStoppedSpeaking timestamp − VAD stop_secs. This measures the delay between the user finishing speech and receiving the final transcription.

### Service Factory (`src/stt_benchmark/services.py`)

Each STT provider has a factory function returning a configured Pipecat FrameProcessor. `STT_SERVICES` dict maps `ServiceName` enum values to `ServiceDefinition` (factory function, required env vars, aiohttp flag). Adding a new service means adding a factory function and an entry in this dict.

### Key Modules

- **models.py** — Pydantic models: `ServiceName` enum, `AudioSample`, `BenchmarkResult`, `GroundTruth`, `WERMetrics`, `SemanticWERTrace`
- **config.py** — Pydantic Settings loaded from `.env` (all API keys, audio params, VAD config, timeouts)
- **storage/database.py** — Async SQLite (aiosqlite) with tables: samples, benchmark_results, ground_truths, wer_metrics, semantic_wer_traces
- **evaluation/semantic_wer.py** — Claude-powered WER using tool-use for text normalization and alignment
- **ground_truth/gemini_transcriber.py** — Gemini transcription with rate limiting (60 req/min default)
- **cli/** — Typer commands: download, run (benchmark), ground-truth, wer, report, export
- **analysis/statistics.py** — Aggregate statistics computation with duration-bucketed analysis

## Configuration

Copy `env.example` to `.env` for API keys. Configuration is managed via Pydantic Settings in `config.py` with defaults for audio (16kHz, 20ms chunks), VAD (0.2s stop threshold), and timeouts (10s).

## Code Style

- Python 3.10+, Ruff for linting/formatting (line length 100), mypy for type checking
- Async/await throughout (Pipecat framework, aiosqlite, CLI uses asyncio.run)
- Logging via loguru, console output via rich
- Data stored in `stt_benchmark_data/` directory (SQLite DB + audio files)
