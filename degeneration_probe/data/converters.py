"""Dataset converters for different HuggingFace dataset formats to probing format."""

import json
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

from datasets import Dataset

from degeneration_probe.types import AnnotatedSpan, ProbingItem


# Mapping from text labels to numeric values for probe training
_MAP_LABEL_TO_SCALAR = {
    'Not Supported': 1.0,
    'NS': 1.0,  # the probe should output 1.0 on text containing unsupported claims
    'Insufficient Information': 1.0,  # the probe should also output 1.0 if the label is 'Insufficient Information'
    'Supported': 0.0,
    'S': 0.0,
    'N/A': -100.0,
    None: -100.0
}

def prepare_longform_dataset(dataset: Dataset) -> List[ProbingItem]:
    """Prepare dataset from the one-shot pipeline labeling format."""
    probing_items: List[ProbingItem] = []

    for hf_item in dataset:
        prompt = hf_item['conversation'][-2]['content']
        completion = hf_item['conversation'][-1]['content']
        annotations: List[dict] = hf_item['annotations']

        annotated_spans: List[AnnotatedSpan] = []

        for entity in annotations:
            if entity is None or 'index' not in entity or not isinstance(entity['index'], int):
                continue

            entity_text = entity['span']
            label = entity['label']
            idx = entity['index']
            
            if idx is None:
                print(f"Entity {repr(entity_text)}'s idx set to None, discarding entity")
                continue
            elif not entity_text or entity_text not in completion:
                print(f"Entity {repr(entity_text)} not found in completion, discarding entity")
                continue

            annotated_spans.append(
                AnnotatedSpan(
                    span=entity_text,
                    label=_MAP_LABEL_TO_SCALAR[label],
                    index=idx
                )
            )

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_longform_dataset_old_format(dataset: Dataset) -> List[ProbingItem]:
    """Prepare dataset from the one-shot pipeline labeling format."""
    probing_items: List[ProbingItem] = []

    for hf_item in dataset:
        prompt = hf_item['conversation'][0]['content']
        completion = hf_item['completion'] if 'completion' in hf_item else hf_item['conversation'][-1]['content']
        annotations: List[dict] = hf_item['verified_entities']

        annotated_spans: List[AnnotatedSpan] = []

        for entity in annotations:
            if entity is None or 'idx' not in entity or not isinstance(entity['idx'], int):
                continue

            entity_text = entity['text']
            label = entity['label']
            idx = entity['idx']
            
            if idx is None:
                print(f"Entity {repr(entity_text)}'s idx set to None, discarding entity")
                continue
            elif not entity_text or entity_text not in completion:
                print(f"Entity {repr(entity_text)} not found in completion, discarding entity")
                continue

            annotated_spans.append(
                AnnotatedSpan(
                    span=entity_text,
                    label=_MAP_LABEL_TO_SCALAR[label],
                    index=idx
                )
            )

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_triviaqa(dataset: Dataset) -> List[ProbingItem]:
    """
    Pre-processes TriviaQA dataset.
    The greedy completion (labeled by an LLM) is at `gt_completion`
    The label is at `llm_judge_label` and it's a string containing `S`, `NS`, `N/A` or some undefined string
    The annotated spans will be the *whole completion*
    """
    assert 'question' in dataset[0] or 'conversation' in dataset[0]

    LABEL_FIELD: str = 'llm_judge_label' if 'llm_judge_label' in dataset.features else 'label'
    COMPLETION_FIELD: str = 'gt_completion'
    VALID_LABELS: List[str] = ['S', 'NS', 'N/A']
    EXACT_ANSWER_FIELD: str = 'exact_answer'

    probing_items = []
    for item in dataset:
        if item[LABEL_FIELD] not in VALID_LABELS:
            print(f"Invalid label {item[LABEL_FIELD]} for item, skipping")
            continue

        prompt = item['question'] if 'question' in item else item['conversation'][0]['content']
        completion = item[COMPLETION_FIELD]
        exact_answer = item[EXACT_ANSWER_FIELD] if EXACT_ANSWER_FIELD in item else ""

        if exact_answer is None or exact_answer not in completion:
            print(f"Exact answer {repr(exact_answer)} not found in completion {repr(completion)}")
            return None

        exact_answer_start_idx = completion.find(exact_answer)
        
        # The whole completion is labeled with the given label
        label_value: float = _MAP_LABEL_TO_SCALAR[item[LABEL_FIELD]]
        
        annotated_spans = [
            AnnotatedSpan(
                span=exact_answer,
                label=label_value,
                index=exact_answer_start_idx
            )
        ]

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_synthetic(dataset: Dataset) -> List[ProbingItem]:
    """Loads the synthetic dataset from the hub."""
    FIELD = 'probing_item_with_hallucinations'

    probing_items = []
    for i, item in enumerate(dataset):
        probing_item = item[FIELD]
        annotated_spans = [
            AnnotatedSpan(
                span=span['text'],
                label=span['label'],
                index=span['start_idx']
            )
            for span in probing_item['spans']
        ]

        # Sort spans by their index in the text
        completion = probing_item['completion']

        if len(completion) <= 500:
            print(f"For item {i} completion is too short ({len(completion)} characters): {repr(completion)}")
            continue

        annotated_spans = sorted(annotated_spans, key=lambda x: x.index)

        if not all(completion[span.index:span.index+len(span.span)] == span.span for span in annotated_spans):
            print(f"For item {i} spans are not aligned with the completion")
            for span in annotated_spans:
                if completion[span.index:span.index+len(span.span)] != span.span:
                    print(f"- Span: {span.span} | {span.index} | {span.label}")
            continue

        probing_items.append(ProbingItem(
            prompt=probing_item['prompt'],
            completion=probing_item['completion'],
            spans=annotated_spans
        ))
    return probing_items


def _first_present(item: dict, fields: Sequence[str]):
    """Return the first present field value from a HF row."""
    for field in fields:
        if field in item and item[field] is not None:
            return item[field], field
    return None, None


def _as_float_labels(values: Iterable) -> List[float]:
    """Normalize token-level label payloads to floats."""
    labels: List[float] = []

    for value in values:
        if value is None:
            labels.append(-100.0)
        elif isinstance(value, dict):
            score, _ = _first_present(
                value,
                [
                    "repetition_score",
                    "repetition",
                    "score",
                    "label",
                    "target",
                    "value",
                ],
            )
            labels.append(-100.0 if score is None else float(score))
        else:
            labels.append(float(value))

    return labels


def _as_token_strings(values: Optional[Iterable]) -> Optional[List[str]]:
    """Normalize optional token payloads to strings."""
    if values is None:
        return None

    tokens: List[str] = []
    for value in values:
        if isinstance(value, dict):
            token, _ = _first_present(value, ["token", "text", "value", "piece"])
            tokens.append("" if token is None else str(token))
        else:
            tokens.append(str(value))

    return tokens


def _apply_optional_token_mask(labels: List[float], item: dict) -> List[float]:
    """Apply common valid/ignore mask conventions to token labels."""
    mask, field = _first_present(
        item,
        [
            "label_mask",
            "valid_mask",
            "token_mask",
            "completion_mask",
            "loss_mask",
            "ignore_mask",
        ],
    )

    if mask is None:
        return labels

    mask = list(mask)
    masked_labels = list(labels)
    for i, keep_or_ignore in enumerate(mask[: len(masked_labels)]):
        if field == "ignore_mask":
            should_ignore = bool(keep_or_ignore)
        else:
            should_ignore = not bool(keep_or_ignore)

        if should_ignore:
            masked_labels[i] = -100.0

    return masked_labels


def _parse_chunk_summary(raw_chunk_summary) -> List[dict]:
    """Parse the chunk_summary column from a decoded value or JSON string."""
    if isinstance(raw_chunk_summary, str):
        try:
            raw_chunk_summary = json.loads(raw_chunk_summary)
        except json.JSONDecodeError as exc:
            raise ValueError("Could not parse chunk_summary as JSON") from exc

    if not isinstance(raw_chunk_summary, list):
        raise ValueError(
            f"Expected chunk_summary to be a list, got {type(raw_chunk_summary).__name__}"
        )

    if not all(isinstance(entry, dict) for entry in raw_chunk_summary):
        raise ValueError("Expected every chunk_summary entry to be a dictionary")

    return raw_chunk_summary


def _repetition_labels_from_chunk_summary(raw_chunk_summary) -> List[float]:
    """Extract repetition scores from chunk_summary, preserving token_index alignment."""
    chunk_summary = _parse_chunk_summary(raw_chunk_summary)

    indexed_entries = []
    for fallback_idx, entry in enumerate(chunk_summary):
        if "repetition" not in entry:
            raise ValueError(f"chunk_summary entry missing 'repetition': {entry}")

        token_index = entry.get("token_index", fallback_idx)
        if token_index is None:
            token_index = fallback_idx

        repetition = entry["repetition"]
        repetition_score = -100.0 if repetition is None else float(repetition)
        indexed_entries.append((int(token_index), repetition_score))

    if not indexed_entries:
        return []

    labels = [-100.0] * (max(idx for idx, _ in indexed_entries) + 1)
    for token_index, repetition_score in sorted(indexed_entries, key=lambda x: x[0]):
        if token_index < 0:
            raise ValueError(f"chunk_summary token_index must be non-negative, got {token_index}")
        labels[token_index] = repetition_score

    return labels


def prepare_repetition_instruct_token_level_dataset(dataset: Dataset) -> List[ProbingItem]:
    """Prepare token-level repetition-score data for regression probing."""
    probing_items: List[ProbingItem] = []

    prompt_fields = ["prompt", "instruction", "query", "question", "input"]
    completion_fields = ["completion", "response", "answer", "output", "generated_text"]
    score_fields = [
        "repetition_scores",
        "token_repetition_scores",
        "repetition_score",
        "scores",
        "token_scores",
        "degeneration_scores",
        "token_degenerating",
        "labels",
        "token_labels",
        "targets",
    ]
    token_fields = ["tokens", "completion_tokens", "generated_tokens", "response_tokens"]

    for hf_item in dataset:
        prompt, _ = _first_present(hf_item, prompt_fields)
        completion, _ = _first_present(hf_item, completion_fields)

        if (prompt is None or completion is None) and "conversation" in hf_item:
            conversation = hf_item["conversation"]
            if len(conversation) >= 2:
                prompt = conversation[-2].get("content", prompt)
                completion = conversation[-1].get("content", completion)

        scores, score_field = _first_present(hf_item, score_fields)
        tokens, _ = _first_present(hf_item, token_fields)
        chunk_summary = hf_item.get("chunk_summary")

        if prompt is None or completion is None or (scores is None and chunk_summary is None):
            available_fields = list(hf_item.keys())
            raise ValueError(
                "Could not parse repetition-score dataset row. "
                f"Need prompt/completion plus chunk_summary or scores fields. "
                f"Available fields: {available_fields}. Recognized score fields: {score_fields}."
            )

        if chunk_summary is not None:
            token_labels = _repetition_labels_from_chunk_summary(chunk_summary)
            token_label_tokens = None
        else:
            if isinstance(scores, (str, bytes)) or not isinstance(scores, Iterable):
                raise ValueError(
                    f"Expected token-level scores in field {score_field!r}, got {type(scores).__name__}."
                )
            token_labels = _as_float_labels(scores)
            token_label_tokens = _as_token_strings(tokens)

        token_labels = _apply_optional_token_mask(token_labels, hf_item)

        probing_items.append(
            ProbingItem(
                prompt=str(prompt),
                completion=str(completion),
                spans=[],
                token_labels=token_labels,
                token_label_tokens=token_label_tokens,
            )
        )

    return probing_items


def get_prepare_function(
    hf_repo: str,
    subset: Optional[str] = None
) -> Callable[[Dataset], List[ProbingItem]]:
    """Get the appropriate preparation function based on dataset name."""
    if 'degeneration-probe-instruct-token-level' in hf_repo:
        return prepare_repetition_instruct_token_level_dataset
    elif 'one_shot_pipeline' in str(subset) or 'hallucination-heads' in hf_repo:
        return prepare_longform_dataset_old_format
    elif 'modified' in str(subset) and 'synthetic-hallucinations' in hf_repo:
        return prepare_synthetic
    elif 'trivia_qa' in str(subset) or 'triviaqa' in hf_repo:
        return prepare_triviaqa
    else:
        # Default to one-shot pipeline format
        return prepare_longform_dataset
