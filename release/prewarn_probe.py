"""Load a released degeneration probe and score or monitor a generation.

A probe reads the residual stream after one decoder block of its base model and
maps each token to the probability that the response is degenerating or about
to. It is a layer norm followed by a linear map, stored in ``probe.safetensors``
next to ``config.json`` (base model, revision and layer) and ``thresholds.json``
(the threshold for each false-alarm budget, calibrated on healthy validation
responses). An adapted probe also ships LoRA adapters, which change the
representation the probe reads.

The thresholds hold for responses generated and scored the way the probes were
trained: the prompt is the model's chat template with the beginning-of-sequence
token removed, tokenized without special tokens, followed by the generated
tokens. ``build_prompt_ids`` builds it.

    monitor = DegenerationMonitor.from_pretrained("marcodene/prewarn-probes-llama-3.1-8b-instruct",
                                                  subfolder="frozen-L15")
    prompt_ids = monitor.build_prompt_ids("Prove that ...")
    scores = monitor.score(prompt_ids, generated_ids)      # one probability per generated token
    first = monitor.first_alarm(scores, budget=0.05)        # index of the first alarm, or None

Only depends on torch, transformers, peft, safetensors and huggingface_hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from safetensors.torch import load_file
from torch import nn

CONFIG = "config.json"
THRESHOLDS = "thresholds.json"
WEIGHTS = "probe.safetensors"


def _resolve(path_or_repo: str, subfolder: Optional[str], revision: Optional[str]) -> Path:
    """A local folder as is, or a snapshot of the Hub repo restricted to ``subfolder``."""
    local = Path(path_or_repo) / (subfolder or "")
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    pattern = f"{subfolder}/*" if subfolder else "*"
    root = snapshot_download(path_or_repo, revision=revision, allow_patterns=[pattern])
    return Path(root) / (subfolder or "")


class ProbeHead(nn.Module):
    """Layer norm and a linear map from one hidden state to one logit."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(states.float())).squeeze(-1)


def build_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """The single-turn chat prompt the probes were trained and calibrated on."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    if tokenizer.bos_token:
        text = text.replace(tokenizer.bos_token, "")
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def decoder_layers(model) -> nn.ModuleList:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


class DegenerationMonitor:
    """A base model, optionally with the probe's adapters, and the probe head reading one of its blocks."""

    def __init__(self, model, tokenizer, head: ProbeHead, config: dict, thresholds: dict) -> None:
        self.model, self.tokenizer, self.head = model, tokenizer, head
        self.config, self.thresholds = config, thresholds
        self.layer = int(config["layer"])
        self._captured: Optional[torch.Tensor] = None
        self._hook = decoder_layers(model)[self.layer].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output[0] if isinstance(output, tuple) else output

    @classmethod
    def from_pretrained(cls, path_or_repo: str, subfolder: Optional[str] = None, *, revision: Optional[str] = None,
                        model=None, tokenizer=None, device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "DegenerationMonitor":
        folder = _resolve(path_or_repo, subfolder, revision)
        config = json.loads((folder / CONFIG).read_text())
        thresholds = json.loads((folder / THRESHOLDS).read_text())
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_model_revision"])
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["base_model_revision"],
                                                         torch_dtype=dtype).to(device)
        if config["adapted"]:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(folder))
        model.eval()
        head = ProbeHead(int(config["hidden_size"]))
        weights = load_file(str(folder / WEIGHTS))
        head.load_state_dict(weights)
        head.to(next(model.parameters()).device).eval()
        return cls(model, tokenizer, head, config, thresholds)

    def build_prompt_ids(self, prompt: str) -> list[int]:
        return build_prompt_ids(self.tokenizer, prompt)

    def close(self):
        """Detach the probe and its adapters and return the base model unchanged."""
        self._hook.remove()
        return self.model.unload() if self.config["adapted"] else self.model

    @torch.no_grad()
    def score(self, prompt_ids: Sequence[int], generated_ids: Sequence[int]) -> torch.Tensor:
        """Probability of degeneration at every generated token, from one forward pass over the response."""
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([list(prompt_ids) + list(generated_ids)], device=device)
        self.model(input_ids=input_ids, use_cache=False)
        logits = self.head(self._captured)[0, len(prompt_ids):]
        return torch.sigmoid(logits).float().cpu()

    def threshold(self, budget: float) -> float:
        for entry in self.thresholds["thresholds"]:
            if abs(float(entry["target_negative_fpr"]) - budget) < 1e-9:
                return float(entry["tau"])
        raise KeyError(f"no threshold for budget {budget}; available: "
                       f"{[e['target_negative_fpr'] for e in self.thresholds['thresholds']]}")

    def first_alarm(self, scores: torch.Tensor, budget: float = 0.05) -> Optional[int]:
        """Index of the first generated token whose score reaches the threshold, or None."""
        flagged = torch.nonzero(scores >= self.threshold(budget))
        return int(flagged[0]) if len(flagged) else None
