#!/usr/bin/env python3
"""Calculate WER (word error rate) from a TSV file with alignment visualization.

Usage:
    python scripts/calc_wer.py --tsvfile assemblyai_transcripts.tsv --outputfile assemblyai_wer.txt
"""

import argparse
import csv
import os
import sys

import jiwer

# Allow importing normalizer from scripts/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer.basic import BasicTextNormalizer


def read_tsv(path: str) -> tuple[list[str], list[list[str]]]:
    """Read a TSV file and return (header, rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        data = list(reader)
    return header, data


def main(args: argparse.Namespace) -> None:
    header, data = read_tsv(args.tsvfile)

    # Set up text normalizer
    if args.text_normalizer == "Whisper":
        tn = BasicTextNormalizer()
    elif args.text_normalizer == "English":
        from normalizer.english import EnglishTextNormalizer
        tn = EnglishTextNormalizer()
    elif args.text_normalizer == "None":
        tn = lambda s: s.translate(str.maketrans("", "", "\u201c\u201d\u201e\u2018\u2019\u2014\u2013\u2026")).replace("...", "")
    else:
        raise ValueError(f"Unknown text normalizer option: {args.text_normalizer}")

    # Resolve column indices
    cols = ["file_path", "target", args.hypothesis_column_name]
    indices = [header.index(c) for c in cols]

    # Accumulators for aggregate WER
    ncor = 0
    nsub = 0
    nins = 0
    ndel = 0

    output_str = []

    for row in data:
        values = {c: row[i] for c, i in zip(cols, indices)}

        ref = tn(values["target"])
        hyp = tn(values[args.hypothesis_column_name])

        result = jiwer.process_words(ref, hyp)

        ncor += result.hits
        nsub += result.substitutions
        nins += result.insertions
        ndel += result.deletions

        vis = jiwer.visualize_alignment(result, show_measures=False, skip_correct=False)
        vis = vis.splitlines()
        vis = vis[1:]  # remove "sentence 1" header
        vis = [
            values["file_path"],
            f"WER: {result.wer * 100:.1f}%",
        ] + vis + [""]

        output_str.append("\n".join(vis))

    # Aggregate WER
    wcount = ncor + nsub + ndel
    wer = float(nsub + nins + ndel) / wcount if wcount > 0 else 0.0
    output_str.append("(Average)")
    output_str.append(f"Word count: {wcount}")
    output_str.append(f"WER: {wer * 100:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.outputfile)), exist_ok=True)

    with open(args.outputfile, "w") as f:
        print("\n".join(output_str), file=f)

    print(f"WER: {wer * 100:.1f}% ({wcount} words) -> {args.outputfile}")


def make_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate WER (word error rate) from a TSV file."
    )
    parser.add_argument(
        "--tsvfile", metavar="<file>", required=True,
        help="Input TSV file with columns: file_path, target, prediction.",
    )
    parser.add_argument(
        "--outputfile", metavar="<file>", required=True,
        help="Output file for WER report and alignments.",
    )
    parser.add_argument(
        "--hypothesis_column_name", metavar="<column>", default="prediction",
        help="Column name for recognition hypotheses (default: prediction).",
    )
    parser.add_argument(
        "--text_normalizer", choices=["None", "Whisper", "English"], default="Whisper",
        help="Text normalizer for reference and hypothesis (default: Whisper).",
    )
    return parser


if __name__ == "__main__":
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
