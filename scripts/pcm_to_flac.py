#!/usr/bin/env python3
"""Convert raw PCM files to FLAC format.

Audio files are assumed to be 16-bit signed PCM (little-endian), mono, at 16kHz
(matching the stt-benchmark default configuration).
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def convert(input_dir: Path, output_dir: Path, sample_rate: int) -> None:
    pcm_files = sorted(input_dir.glob("*.pcm"))
    if not pcm_files:
        print(f"No .pcm files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(pcm_files, 1):
        dst = output_dir / src.with_suffix(".flac").name
        audio = np.frombuffer(src.read_bytes(), dtype=np.int16)
        sf.write(str(dst), audio, samplerate=sample_rate, subtype="PCM_16")
        print(f"[{i}/{len(pcm_files)}] {src.name} -> {dst.name}")

    print(f"\nDone. {len(pcm_files)} files written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw PCM files to FLAC")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write FLAC files to",
    )
    parser.add_argument(
        "--input-dir",
        default="stt_benchmark_data/audio",
        help="Directory containing .pcm files (default: stt_benchmark_data/audio)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the PCM audio in Hz (default: 16000)",
    )
    args = parser.parse_args()

    convert(Path(args.input_dir), Path(args.output_dir), args.sample_rate)


if __name__ == "__main__":
    main()
