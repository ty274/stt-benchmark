#!/usr/bin/env python3
"""Export ground-truth and STT transcripts as a TSV file."""

import argparse
import csv
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("stt_benchmark_data/results.db")


def export_tsv(service: str, db_path: Path, output: Path, normalized: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))

    if normalized:
        cursor = conn.execute(
            """
            SELECT s.audio_path, w.normalized_reference, w.normalized_hypothesis
            FROM wer_metrics w
            JOIN samples s ON w.sample_id = s.sample_id
            WHERE w.service_name = ?
              AND w.normalized_reference IS NOT NULL
              AND w.normalized_hypothesis IS NOT NULL
            ORDER BY s.dataset_index
            """,
            (service,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT s.audio_path, g.text, r.transcription
            FROM results r
            JOIN samples s ON r.sample_id = s.sample_id
            JOIN ground_truth g ON r.sample_id = g.sample_id
            WHERE r.service_name = ?
              AND r.transcription IS NOT NULL
              AND r.transcription != ''
            ORDER BY s.dataset_index
            """,
            (service,),
        )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        label = "normalized wer_metrics" if normalized else "results"
        print(f"No {label} found for service '{service}' in {db_path}")
        return

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file_path", "target", "prediction"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export transcripts as TSV")
    parser.add_argument("--service", required=True, help="STT service name (e.g., assemblyai)")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database")
    parser.add_argument("--output", "-o", default=None, help="Output TSV path")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Export normalized_reference/normalized_hypothesis from wer_metrics instead of raw transcripts",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    output = Path(args.output) if args.output else Path(f"{args.service}_transcripts.tsv")
    export_tsv(args.service, db_path, output, normalized=args.normalized)


if __name__ == "__main__":
    main()
