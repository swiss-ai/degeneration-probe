"""Write the Hugging Face model card (README.md) of each released probe repository.

Reads the folders written by scripts/export_release.py and fills one card per
generator: what the probes are, the setting the thresholds were calibrated in,
a self-contained loading example, the thresholds, and the in-domain test results
of all three seeds with the released one marked.

    python scripts/write_model_cards.py [--release <store>/release] [--labels judge|human]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
HF_USER = "marcodene"
PAPER_TITLE = "Degeneration Detection Is Not Early Warning: False-Alarm Evaluation with PREWARN"
AUTHORS = ("De Negri, Marco and Sartori, Luca and Baggi, Lorenzo and Reihs, Moritz and "
           "Durech, Eduard Frank and El-Assady, Mennatallah and Pyatkin, Valentina and Hedström, Anna")
GENERATORS = {
    "apertus-8b-instruct": {"name": "Apertus-8B-Instruct", "license": "apache-2.0", "note": None},
    "llama-3.1-8b-instruct": {"name": "Llama-3.1-8B-Instruct", "license": "llama3.1",
                              "note": "Built with Llama. The adapters and probes are derived from Llama 3.1 and are "
                                      "released under the Llama 3.1 Community License."},
    "mistral-7b-instruct-v0.1": {"name": "Mistral-7B-Instruct-v0.1", "license": "apache-2.0", "note": None},
}


def fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def probe_rows(repo: Path):
    for folder in sorted((p for p in repo.iterdir() if (p / "config.json").is_file()),
                         key=lambda p: (p.name.split("-")[0] != "frozen", int(p.name.split("L")[-1]))):
        config = json.loads((folder / "config.json").read_text())
        thresholds = {f"{t['target_negative_fpr']:g}": t for t in json.loads((folder / "thresholds.json").read_text())["thresholds"]}
        seeds = json.loads((folder / "seeds.json").read_text())
        yield folder.name, config, thresholds, seeds


def card(repo: Path, labels: str) -> str:
    slug = repo.name.removeprefix("prewarn-probes-")
    info = GENERATORS[slug]
    probes = list(probe_rows(repo))
    base = probes[0][1]["base_model"]
    revision = probes[0][1]["base_model_revision"]
    example = next(name for name, config, _, _ in probes if not config["adapted"])

    probe_table = ["| Folder | Probe | Layer | Trained | Released seed |", "|---|---|---|---|---|"]
    for name, config, _, _ in probes:
        trained = "LoRA adapters (rank 16) on blocks 0–{} and the probe".format(config["layer"]) if config["adapted"] \
            else "the probe only, base model frozen"
        probe_table.append(f"| `{name}` | {config['regime']} | {config['layer']} | {trained} | {config['seed']} |")

    threshold_table = ["| Folder | τ at 1% | τ at 5% | τ at 10% |", "|---|---|---|---|"]
    for name, _, thresholds, _ in probes:
        threshold_table.append(f"| `{name}` | " + " | ".join(fmt(thresholds[b]["tau"], 4) for b in ["0.01", "0.05", "0.1"]) + " |")

    result_table = ["| Folder | Seed | False alarms | Post-onset coverage | Warn₂₅₆ | Early-warning rate | Median lead (tokens) |",
                    "|---|---|---|---|---|---|---|"]
    for name, config, _, seeds in probes:
        for entry in sorted(seeds, key=lambda e: e["seed"]):
            m = entry["metrics"]["test_indomain"]["0.05"]
            mark = f"**{entry['seed']}** (released)" if entry["seed"] == config["seed"] else str(entry["seed"])
            result_table.append(
                f"| `{name}` | {mark} | {100 * m['negative_fpr']:.1f}% | {fmt(m['in_pattern_recall'])} | "
                f"{fmt(m['warning_recall_256'])} | {fmt(m['fired_before_frontier'])} | {m['median_offset']:+.0f} |")

    label_text = ("onsets annotated by an LLM judge" if labels == "judge"
                  else "onsets annotated by an LLM judge and verified by hand")
    mistral_note = ("\n- On this model the probes rarely warn before the onset: their median first alarm comes after it, "
                    "so they are better suited to detecting a loop that has started than to warning of one.") \
        if slug.startswith("mistral") else ""
    license_note = f"\n\n{info['note']}" if info["note"] else ""

    return f"""---
license: {info['license']}
base_model: {base}
library_name: peft
tags:
- probe
- linear-probe
- degeneration
- repetition
- early-warning
- monitoring
---

# PREWARN degeneration probes for {info['name']}

Linear probes that read the residual stream of [{base}](https://huggingface.co/{base}) while it generates
and score every generated token for **repetitive degeneration**, a loop that keeps repeating until the
token limit is reached. Each probe comes with thresholds that make it raise an alarm on at most 1%, 5% or
10% of healthy responses. The probes are evaluated in the paper *{PAPER_TITLE}* (TODO: paper link).

## Probes

{chr(10).join(probe_table)}

Layer ℓ is the output of decoder block ℓ, with blocks numbered from 0 (`model.model.layers[ℓ]`,
`hidden_states[ℓ + 1]` in Transformers). A probe applies a layer norm and a linear map to that hidden
state and returns one logit per token. A frozen probe reads the activations the base model computes
during generation. An adapted probe reads them after its LoRA adapters, which must be active when it
scores. The released seed of each probe is the one of three with the highest warning coverage on
validation at the 1% budget.

Each folder holds `probe.safetensors` (layer norm and linear weights), `config.json` (base model,
pinned revision `{revision[:12]}`, layer), `thresholds.json` (thresholds per budget, calibrated on
validation), `seeds.json` (metrics of all three seeds) and, for adapted probes, the PEFT adapter files.

## Usage

The thresholds hold for responses generated and scored as in the paper: a single user turn formatted
with the model's chat template **without the beginning-of-sequence token** and tokenized without
special tokens, sampling with temperature 0.7 and top-p 0.9, and at most 4,096 new tokens.

```python
import json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

repo, probe = "{HF_USER}/{repo.name}", "{example}"
path = snapshot_download(repo, allow_patterns=[f"{{probe}}/*"]) + f"/{{probe}}"
config = json.load(open(f"{{path}}/config.json"))
tau = {{t["target_negative_fpr"]: t["tau"] for t in json.load(open(f"{{path}}/thresholds.json"))["thresholds"]}}[0.05]

tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_model_revision"])
model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["base_model_revision"],
                                             dtype=torch.bfloat16, device_map="cuda")
if config["adapted"]:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, path)

weights = load_file(f"{{path}}/probe.safetensors", device="cuda")
norm = torch.nn.LayerNorm(config["hidden_size"], device="cuda")
norm.load_state_dict({{"weight": weights["norm.weight"], "bias": weights["norm.bias"]}})
head = torch.nn.Linear(config["hidden_size"], 1, device="cuda")
head.load_state_dict({{"weight": weights["linear.weight"], "bias": weights["linear.bias"]}})

def prompt_ids(prompt):
    text = tokenizer.apply_chat_template([{{"role": "user", "content": prompt}}], tokenize=False,
                                         add_generation_prompt=True)
    return tokenizer(text.replace(tokenizer.bos_token, ""), add_special_tokens=False)["input_ids"]

@torch.no_grad()
def score(prompt, generated_ids):
    ids = prompt_ids(prompt)
    out = model(torch.tensor([ids + list(generated_ids)], device="cuda"), output_hidden_states=True)
    states = out.hidden_states[config["hidden_states_index"]][0, len(ids):]
    return torch.sigmoid(head(norm(states.float()))).squeeze(-1)   # one score per generated token

scores = score(prompt, generated_ids)
first_alarm = (scores >= tau).nonzero()   # the first row is the first alarm, if any
```

A frozen probe can also run during generation by reading the same hidden state at each new token, at no
extra cost. An adapted probe changes the model's outputs if its adapters stay active during generation, so
it needs a separate scoring pass over the generated tokens.

## Thresholds

An alarm is raised at the first token whose score reaches τ. τ is the lowest threshold at which at most the
given share of healthy validation responses raise an alarm.

{chr(10).join(threshold_table)}

## Results

In-domain test split at the 5% budget, with {label_text}. *False alarms*: share of healthy test responses
that raise an alarm. *Post-onset coverage*: share of tokens from the onset onward that are flagged.
*Warn₂₅₆*: share of the 256 tokens before the onset that are flagged. *Early-warning rate*: share of
alarmed degenerate responses whose first alarm comes before the onset. *Median lead*: first alarm minus
onset, negative when the alarm comes first. All metrics at 1%, 5% and 10% are in each folder's `seeds.json`.

{chr(10).join(result_table)}

## Data

Prompts come from seven sources covering mathematics, competitive programming, instruction following and
medical reasoning. Each prompt was answered 10 times. Splits are assigned per prompt. A response that emits
the end-of-sequence token is treated as healthy. A response that reaches the token limit is labelled by an
LLM judge, which reads the prompt and the response and quotes where the loop begins; that position is the
onset.

## Limitations

- The thresholds are calibrated for the prompt format and sampling settings above. Other settings change
  the score distribution, and the false-alarm rate is then no longer controlled.
- Long healthy responses are more likely to raise a false alarm than short ones.
- A response that loops and then emits the end-of-sequence token counts as healthy, so an alarm on it
  counts as a false alarm.
- The released seed is the best of three on validation, which does not make it the best on test. The
  results table lists all three seeds so the spread between them can be read.{mistral_note}

## Citation

```bibtex
@inproceedings{{TODO,
  title  = {{{PAPER_TITLE}}},
  author = {{{AUTHORS}}},
  year   = {{2026}},
}}
```{license_note}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=STORE / "release")
    parser.add_argument("--labels", choices=["judge", "human"], default="judge")
    args = parser.parse_args()
    for repo in sorted(args.release.glob("prewarn-probes-*")):
        (repo / "README.md").write_text(card(repo, args.labels))
        print(f"-> {repo / 'README.md'}")


if __name__ == "__main__":
    main()
