"""Data types for probe training."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnnotatedSpan:
    """A text span with its hallucination label."""
    
    span: str  # The span text
    label: float  # 1.0 for hallucination, 0.0 for supported, -100.0 for ignored
    index: int  # Start index in the completion


@dataclass
class ProbingItem:
    """A single item containing prompt, completion and annotated spans."""

    prompt: str
    completion: str
    spans: List[AnnotatedSpan]
    token_labels: Optional[List[float]] = None
    token_label_tokens: Optional[List[str]] = None
    # Exact completion token ids (e.g. from a prior generation run), used in place of
    # retokenizing `completion` when set -- see `TokenizedProbingDataset
    # ._process_token_level_item`. `completion`'s own retokenization via the tokenizer
    # isn't guaranteed to round-trip losslessly (decode(encode(x)) != x in general), which
    # silently misaligns `token_labels` against the wrong tokens whenever it doesn't.
    # Providing the original ids sidesteps that risk entirely instead of requiring a
    # separate round-trip validation pass.
    completion_token_ids: Optional[List[int]] = None
