# Manual inspection of LLM-judge calibration verdicts

Tooling to hand out a subset of `llm_judge.py`'s calibration verdicts to
teammates for manual review, and to merge their reviews back into one table.

Three pieces:

- `export_judge_review.py` — you run this. Builds a review package
  (`judge_review_data.json` + a copy of `judge_review.html`) to send to
  teammates.
- `judge_review.html` — a standalone, offline review app. Teammates open it
  in a browser (no server, no install) next to the JSON file, review their
  assigned rows, and export their own results as a small JSON file.
- `merge_judge_reviews.py` — you run this. Combines everyone's exported
  review files into one table and prints agreement stats.

## Why this population specifically

Only rollouts that hit the token cap without reaching EOS
(`stop_reason == "length"`, the `"truncated"` stratum in
`llm_judge.select_calibration_sample`) are included. That cap is a necessary
(not sufficient) condition for real degeneration — see `onset_labels.py`'s
module docstring, which validated `stop_reason == "length"` at 99.6%
precision against LLM-judge ground truth. This is exactly the population
where a manual pass is worth the effort: checking the judge's `onset_quote`
position, and catching the rare misclassification (most often a false
negative — the judge reads genuine progress, e.g. brute-force enumeration,
into what is actually a non-progressing loop).

Only rows already judged (`status == "ok"`) by the chosen backend are
eligible — a failed/unjudged row has no verdict to review yet.

## 1. Export a review package

```bash
.venv/bin/python scripts/manual-inspection/export_judge_review.py \
  --config configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml \
  --reviewers "alice,bob,carlo" \
  --out-dir /path/to/handoff/dir
```

Key flags:

| Flag | Meaning |
|---|---|
| `--config` | Which dataset config to pull the calibration sample/results from. |
| `--backend` | Which judge backend's results to use (default `claude_agent_sdk`). |
| `--reviewers` | Comma-separated reviewer names. Rows are assigned round-robin. |
| `--max-samples` | Cap the total exported rows. Every `is_degenerating=False` row is always kept (rare, high-value for false negatives); remaining budget favors low/medium-confidence rows. Omit to export every eligible row. |
| `--seed` | Shuffle/assignment seed (default `0`). |
| `--skip-heuristics` | Skip attaching `repetition_score` / LRS heuristic context (faster export, "blinder" review). |
| `--out-dir` | Where to write `judge_review_data.json` and a copy of `judge_review.html`. |

This writes two files into `--out-dir`:

- `judge_review_data.json` — every eligible row: prompt, completion, the
  judge's verdict (`is_degenerating`, `confidence`, `reasoning`,
  `onset_quote` + its resolved character offset), optional heuristic scores,
  and which reviewer it's assigned to.
- `judge_review.html` — an unmodified copy of the review app.

Zip the `--out-dir` folder and send it to your teammates (email, Slack,
shared drive — whatever you normally use). **Both files must end up in the
same folder** on the teammate's machine (e.g. both saved into `Downloads/`)
— the app looks for the JSON next to itself.

## 2. Reviewers use the HTML app

Teammates just open `judge_review.html` in any modern browser (Chrome,
Firefox, Safari, Edge) — no install, no server needed.

1. **Load the data.** It tries to auto-load `judge_review_data.json` from
   the same folder. If the browser blocks that (common when opening a file
   directly from disk — `fetch()` on `file://` is blocked by CORS by
   default), use the manual file picker instead; it always works.
2. **Pick your name** from the dropdown. The app filters to only the rows
   assigned to you.
3. **Review each row**, in the "Le mie sequenze" ("My sequences") tab:
   - Read the prompt and the generated completion. The judge's claimed
     degeneration onset is highlighted in **orange**.
   - Confirm or correct the classification (degenerating / not
     degenerating) with the toggle buttons.
   - If degenerating, confirm the onset position, or click directly in the
     completion text at the point you think it actually starts — your pick
     is highlighted in **blue**.
   - Add a free-text comment, especially when you change something (e.g.
     "this is brute-force enumeration advancing on each iteration, not a
     loop").
   - Check "Ho completato la revisione" ("I've completed this review") when
     done with that row.
   - "Panoramica assegnazioni" ("Assignment overview") tab shows how many
     rows are assigned to each person.
4. **Progress autosaves** to the browser's `localStorage` — closing the tab
   or restarting the computer doesn't lose anything, as long as it's the
   same browser profile on the same machine. To move to another machine, use
   "Esporta le mie revisioni" to export progress-so-far, then the "import"
   file picker at the bottom of the export panel on the new machine to
   resume.
5. **Export and send back.** "Esporta le mie revisioni" downloads a JSON
   file (`judge_review_<name>_<dataset>.json`) containing **only** the rows
   marked completed. Send that file back to whoever is collecting reviews —
   it's small, since it excludes the prompt/completion text (those live in
   the shared `judge_review_data.json` already).

## 3. Merge everyone's reviews back together

Once you've collected everyone's exported JSON files:

```bash
.venv/bin/python scripts/manual-inspection/merge_judge_reviews.py \
  --inputs "handoff/judge_review_*.json" \
  --out handoff/merged_reviews.parquet
```

`--inputs` accepts explicit paths and/or glob patterns (quote glob patterns
so the shell doesn't expand them). `--out` accepts `.parquet`, `.csv`, or
`.json` — pick based on what you want to load next.

This prints a summary (counts per reviewer, how often reviewers overrode the
judge's classification and in which direction, onset-position agreement
among rows judged degenerating, how many rows have a comment) and writes the
combined table. If the same row was reviewed by more than one person, a
warning is printed but every copy is kept in the output — resolve those by
hand if it happens.

## File format reference

`judge_review_data.json`:

```jsonc
{
  "meta": {
    "schema_version": 1,
    "dataset_tag": "...", "backend": "...", "reviewers": [...],
    "total_eligible_rows": 397, "sampled_rows": 30, "reviewer_counts": {...}, ...
  },
  "records": [
    {
      "id": "<prompt_id>::<rollout_idx>",
      "prompt_id": "...", "rollout_idx": 0, "domain": "...", "stratum": "truncated",
      "assigned_to": "alice",
      "prompt_text": "...", "completion_text": "...",
      "heuristics": {"max_repetition_score": 0.93, "lrs_period_repeat_count": 4},
      "judge": {
        "backend": "...", "is_degenerating": true, "confidence": "high",
        "reasoning": "...", "onset_quote": "...",
        "onset_char_start": 481, "onset_char_end": 645
      }
    }
  ]
}
```

A reviewer's exported file (`judge_review_<name>_<dataset>.json`):

```jsonc
{
  "meta": {"reviewer": "alice", "dataset_tag": "...", "backend": "...", "n_reviewed": 10, "n_assigned": 15, ...},
  "reviews": [
    {
      "id": "<prompt_id>::<rollout_idx>", "prompt_id": "...", "rollout_idx": 0,
      "judge_is_degenerating": true,
      "classification_status": "confirmed",       // unset | confirmed | corrected
      "corrected_is_degenerating": true,
      "judge_onset_quote": "...", "judge_onset_char_start": 481,
      "position_status": "correct",                // unset | correct | corrected | ambiguous | na
      "corrected_onset_char_start": null,
      "corrected_onset_text": null,
      "comment": "", "reviewed_at": "2026-07-16T..."
    }
  ]
}
```
