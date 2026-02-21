# Semantic Word Error Rate (SWER)

Semantic WER is a variant of the standard Word Error Rate metric that counts only transcription
errors that would change how a downstream LLM agent understands or responds to the user. Surface
differences that an LLM would interpret identically — punctuation, capitalization, singular/plural
variations, possessives, filler words, minor grammar — are not counted.

---

## How It Works

Each sample goes through a two-actor pipeline: Claude reasons through the comparison using a
structured 5-step process, then Python executes the final arithmetic.

```
Ground truth + STT transcript
        │
        ▼
┌───────────────────────┐
│   Claude (Sonnet)     │  ← System prompt with 13 normalization rules + 8 few-shot examples
│                       │
│  1. NORMALIZE         │  Apply text normalization to both reference and hypothesis
│  2. ALIGN             │  Word-by-word edit-distance alignment
│  3. SEMANTIC CHECK    │  Per-difference: would an LLM respond differently? YES/NO
│  4. COUNT             │  Sum only the YES differences as S / D / I
│  5. Call tool ──────────────────────────────────────────────────┐
└───────────────────────┘                                         │
                                                                  ▼
                                                   ┌─────────────────────────┐
                                                   │  Python: _calculate_wer │
                                                   │  WER = (S+D+I) / N      │
                                                   └─────────────────────────┘
```

The arithmetic is intentionally kept out of Claude's hands to eliminate arithmetic hallucinations.
Claude decides *what* the errors are; Python computes *the number*.

---

## The 5-Step Process

### Step 1 — NORMALIZE

Both the reference (ground truth) and hypothesis (STT output) are normalized before comparison.
The 13 normalization rules are:

| # | Rule | Example |
|---|------|---------|
| 1.1 | Lowercase | `Hello` → `hello` |
| 1.2 | Remove punctuation | `ready, set` → `ready set` |
| 1.3 | Expand contractions | `I'm` → `i am`, `don't` → `do not` |
| 1.4 | Normalize numbers | `3` = `three`, `$5` = `five dollars`, `1st` = `first` |
| 1.5 | Remove filler words (if asymmetric) | `um`, `uh`, `like`, `you know`, `well`, `so`, `actually`, `basically` |
| 1.6 | Expand abbreviations | `Dr.` = `doctor`, `Mr.` = `mister`, `St.` = `saint` or `street` |
| 1.7 | British/American spelling | `colour` = `color`, `favourite` = `favorite` |
| 1.8 | Hyphenation | `long-term` = `long term` = `longterm` |
| 1.9 | Spoken variants | `gonna` = `going to`, `yeah` = `yes`, `ok` = `okay` |
| 1.10 | Symbols to words | `&` = `and`, `@` = `at` |
| 1.11 | Possessives | `driver's` = `drivers` = `driver` (same referent) |
| 1.12 | Singular/plural | `license` = `licenses`, `ticket` = `tickets` (when concept is the same) |
| 1.13 | Minor grammar | `setting up` = `set up`; missing articles (`the`, `a`) that don't change meaning |

### Step 2 — ALIGN

After normalization, Claude performs a word-by-word alignment of the two texts using edit distance.
Each position is classified as a match, substitution, deletion, or insertion.

### Step 3 — SEMANTIC CHECK

This is the step that distinguishes Semantic WER from standard WER. For **every** difference found
in the alignment, Claude applies the key question:

> *Would an LLM agent respond differently to these two versions?*

Claude writes out the structured reasoning for each difference:

```
DIFFERENCE: "X" → "Y"
QUESTION: Would an LLM agent respond differently?
ANSWER: [YES/NO] because [reason]
COUNT AS ERROR: [YES/NO]
```

**Not errors (NO):**
- Singular/plural: `license` → `licenses` — same concept
- Possessives: `driver's` → `drivers` — same referent
- Missing articles: `the coastal areas` → `coastal areas`
- Hyphenation: `Wi-Fi` → `wi fi`

**Errors (YES):**
- Different content words: `card` → `car`, `trace` → `trade`
- Nonsense substitutions: `lentil` → `landon`, `Wi-Fi` → `wi fire`
- Subject/meaning changes: `I'm` → `When`

**Special counting rules:**
- Compound words count as **one** error, not multiple. `cross-country` → `koscanti` = S=1.
- Truncated text: when both texts are cut off at the same point, compare only the complete
  portions. Partial words at the truncation boundary are ignored.
- Trailing function words at truncation (`and`, `but`, `to`, `for`, `the`, …) omitted by the
  hypothesis are **not** counted as errors — they carry no standalone semantic content.

### Step 4 — COUNT

Only the differences where the semantic check answered YES are counted:

| Symbol | Meaning |
|--------|---------|
| S | Semantic substitutions — different word at the same position |
| D | Semantic deletions — word present in reference, absent in hypothesis |
| I | Semantic insertions — extra word in hypothesis not in reference |
| N | Total word count in the **normalized reference** |

### Step 5 — CALCULATE

Claude calls the `calculate_wer` tool with `(S, D, I, N)`. Python executes:

```
WER = (S + D + I) / N
```

Edge cases:
- Both texts empty → WER = 0.0
- Reference empty, hypothesis non-empty → WER = ∞
- Hypothesis empty, reference non-empty → WER = 1.0 (all words deleted), capped at 100%

---

## API Interaction Pattern

Semantic WER uses a two-turn Anthropic API conversation:

```
Turn 1  User  ──► "Here is reference / hypothesis. Compute WER."
        Claude ◄── Reasoning text + tool_use: calculate_wer(S, D, I, N, ...)

Turn 2  User  ──► tool_result: { wer: 0.034, ... }    ← Python computed this
        Claude ◄── Short summary of the result
```

The loop has a safety limit of 10 turns, but in practice exits after turn 1 + one final
summary call. Claude uses `claude-sonnet-4-5` at 4096 max tokens for the reasoning turn
and 1024 max tokens for the summary turn.

---

## What Gets Stored

Per sample, the following are written to the SQLite database (`stt_benchmark_data/results.db`):

**`wer_metrics` table**
- `wer` — final WER value (float)
- `substitutions`, `deletions`, `insertions`, `reference_words` — raw counts
- `normalized_reference`, `normalized_hypothesis` — the post-normalization texts Claude produced
- Per-error detail: type, reference word, hypothesis word, position

**`semantic_wer_traces` table**
- Full conversation trace as JSON (every assistant + tool message)
- Tool call inputs and outputs
- `num_turns`, `duration_ms`, `model_used`
- `session_id` (UUID) for cross-referencing

The normalized texts can be exported with `scripts/export_tsv.py --normalized` for use with
traditional WER tools such as `scripts/calc_wer.py`.

---

## Comparison to Standard WER

| | Standard WER | Semantic WER |
|---|---|---|
| Normalization | Whisper or English normalizer (rule-based) | Claude (LLM-based, context-aware) |
| Alignment | Exact word edit distance (`jiwer`) | Claude judgment after normalization |
| Error counting | All word differences | Only differences an LLM would act on differently |
| Plural/possessive | Counted as errors | Not counted |
| Missing articles | Counted as errors | Not counted |
| Arithmetic | `jiwer` | Python (`_calculate_wer`) |
| Reasoning trace | None | Stored in `semantic_wer_traces` |
| Cost | Free (local) | Claude API call per sample |

Standard WER can be computed offline using `scripts/calc_wer.py`; Semantic WER requires the
`uv run stt-benchmark wer` command and a valid `ANTHROPIC_API_KEY`.
