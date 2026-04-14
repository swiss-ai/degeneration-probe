"""Token-by-token generation engine with probe scoring and steering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from degeneration_probe.probe import SequenceProbe
from degeneration_probe.worker.steering import SteeringContext, SteeringStrategy


@dataclass
class TokenResult:
    """Result for a single generated token."""
    token_id: int
    token_text: str
    position: int
    probe_score: float
    was_steered: bool


class GenerationEngine:
    """Runs token-by-token generation with probe scoring and optional steering.

    The engine holds the model, tokenizer, and optionally a trained probe.
    For each generation request it:
    1. Encodes the prompt
    2. In a loop: runs one forward pass, hooks the probe layer, scores,
       optionally steers logits, samples the next token, yields the result.

    When a probe is provided the engine permanently registers the probe's
    forward hook on the target layer at construction time.  This means the
    hook fires during every ``model()`` call made by the engine, so we can
    read ``probe._hooked`` after each forward pass without re-running the
    model through ``probe.forward()``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        probe: Optional[SequenceProbe] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self._stop_requested = False
        self._hook_handle = None
        # Mutable view of the current generation's sampling + steering params.
        # The loop reads this every iteration, so external code can update the
        # sliders mid-stream.
        self._live_params: Optional[dict] = None

        if probe is not None:
            self._hook_handle = probe.target_module.register_forward_hook(probe._hook_fn)

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

    def request_stop(self):
        """Signal the generation loop to stop after the current token."""
        self._stop_requested = True

    def update_live_params(self, **updates) -> None:
        """Patch the running generation's params. No-op if nothing is running."""
        if self._live_params is None:
            return
        for key, value in updates.items():
            if key in self._live_params and value is not None:
                self._live_params[key] = value

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.01,
        top_p: float = 0.9,
        steering: Optional[SteeringStrategy] = None,
        steering_threshold: float = 0.8,
    ) -> Generator[TokenResult, None, None]:
        """Token-by-token generation generator. Yields one TokenResult per token.

        The worker streams each yielded result over WebSocket immediately,
        so the client sees tokens appear in real time.
        """
        self._stop_requested = False
        self._live_params = {
            "temperature": temperature,
            "top_p": top_p,
            "steering_threshold": steering_threshold,
        }
        device = getattr(self.model, "device", torch.device("cpu"))

        # Encode prompt. Base (non-instruct) models have no chat template; tokenize raw.
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        attention_mask = (
            encoded["attention_mask"]
            if isinstance(encoded, dict)
            else getattr(encoded, "attention_mask", torch.ones_like(input_ids))
        )

        generated_ids: list[int] = []
        past_key_values = None
        current_input_ids = input_ids

        for pos in range(max_new_tokens):
            if self._stop_requested:
                break

            # Clear hook capture before each forward pass
            if self.probe is not None:
                self.probe._hooked = None

            # Forward pass with KV cache: first iteration processes the full
            # prompt; subsequent iterations feed only the newly sampled token.
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            # Get logits for the last position
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Probe scoring
            probe_score = 0.0
            if self.probe is not None and self.probe._hooked is not None:
                hidden = self.probe._hooked[:, -1, :]  # [1, H]
                if hidden.dtype != self.probe.linear.weight.dtype:
                    hidden = hidden.to(self.probe.linear.weight.dtype)
                logit = self.probe.linear(hidden)  # [1, 1]
                probe_score = torch.sigmoid(logit).item()

            # Re-read live params every iteration so mid-stream UI slider
            # updates take effect on subsequent tokens.
            live_temperature = self._live_params["temperature"]
            live_top_p = self._live_params["top_p"]
            live_threshold = self._live_params["steering_threshold"]

            # Steering
            was_steered = False
            if steering is not None and steering.should_intervene(probe_score, live_threshold):
                ctx = SteeringContext(
                    recent_token_ids=generated_ids[-50:],
                    position=pos,
                )
                logits = steering.intervene(logits.squeeze(0), ctx).unsqueeze(0)
                was_steered = True

            # Sample
            if live_temperature < 0.02:
                next_token_id = logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits / live_temperature, dim=-1)
                if live_top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    mask = cumulative - sorted_probs > live_top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    idx = torch.multinomial(sorted_probs, 1)
                    next_token_id = sorted_indices.gather(-1, idx).item()
                else:
                    next_token_id = torch.multinomial(probs, 1).item()

            # EOS check
            if next_token_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            generated_ids.append(next_token_id)

            yield TokenResult(
                token_id=next_token_id,
                token_text=token_text,
                position=pos,
                probe_score=probe_score,
                was_steered=was_steered,
            )

            # Next iteration: feed only the new token; attention_mask grows
            # to cover the cached context plus the new position.
            current_input_ids = torch.tensor([[next_token_id]], device=device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

        self._live_params = None
