"""Label existing generation JSONL files with OpenAI models.

Reads generation outputs (from `python -m degeneration_probe generate`),
labels each as degenerate/not with one or more OpenAI models, and writes
a new JSONL with the labels added.

Example:
    uv run python scripts/label_generations.py \
      --input_dir outputs/generations/20260403_120000/data \
      --models gpt-5.4 gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABELING_PROMPT = """\
You are evaluating whether a language model's output shows signs of text degeneration.

Degeneration means the model got stuck in a repetitive loop, producing the same \
words, phrases, or patterns over and over. This is different from:
- Naturally listing items (not degeneration)
- Using some repeated words in normal prose (not degeneration)
- Repetition for rhetorical emphasis (not degeneration)

Only label as degenerate if the output clearly shows the model is stuck in an \
unintentional repetitive loop.

Prompt given to the model:
{prompt}

Model's response:
{response}

Is this response degenerate? Answer with exactly one word: "yes" or "no"."""


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_with_openai(
    client: OpenAI, prompt: str, response: str, model: str
) -> bool:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": LABELING_PROMPT.format(prompt=prompt, response=response),
            }
        ],
        max_tokens=5,
        temperature=0.0,
    )
    answer = completion.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label generation outputs with OpenAI models."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to generations JSONL file",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["gpt-5.4", "gpt-4o-mini"],
        help="OpenAI models to use for labeling",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output JSONL path (default: input with _labeled suffix)",
    )
    args = parser.parse_args()

    # OpenAI client
    openai_key_path = PROJECT_ROOT / "keys" / ".openai_key"
    api_key = openai_key_path.read_text(encoding="utf-8").strip()
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)

    input_path = Path(args.input_path)
    records = read_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    for model_name in args.models:
        label_key = f"llm_label_{model_name.replace('.', '_').replace('-', '_')}"
        print(f"\nLabeling with {model_name} -> field '{label_key}'")

        for idx, record in enumerate(records, start=1):
            if label_key in record:
                continue  # skip already labeled
            prompt = record["prompt"]
            response = record.get("generated_text", record.get("completion", ""))
            label = label_with_openai(client, prompt, response, model=model_name)
            record[label_key] = label
            if idx % 25 == 0 or idx == len(records):
                print(f"  [{idx}/{len(records)}]")

        n_pos = sum(1 for r in records if r.get(label_key))
        print(f"  {model_name}: {n_pos}/{len(records)} degenerate ({n_pos/len(records)*100:.1f}%)")

    # Save
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.with_stem(input_path.stem + "_labeled")
    write_jsonl(output_path, records)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
