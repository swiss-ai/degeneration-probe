FROM nvcr.io/nvidia/pytorch:25.06-py3

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

ARG BRANCH=main

WORKDIR /workspace

RUN git clone --branch ${BRANCH} --single-branch https://github.com/Luca-Sartori/degeneration-probe.git
WORKDIR /workspace/degeneration-probe

# should I really do break-system-packages?
RUN uv pip install -e . --system --break-system-packages

# The base image ships a torchao older than peft accepts, and peft raises on
# it while attaching adapters rather than skipping the backend. A newer torchao
# cannot import against this image's torch, so the only workable state is not
# having it at all: absent, peft reports it unavailable and adapters attach
# through the ordinary path. Nothing in this project uses it.
RUN uv pip uninstall torchao --system --break-system-packages || true
RUN python -c "import importlib.util; assert importlib.util.find_spec('torchao') is None, 'torchao is still installed'"

# A dependency that disagrees with the base image only shows up on use, so
# check it here rather than in a job that has already queued. Attaching an
# adapter to a throwaway model exercises the dispatch path where a mismatched
# accelerator library actually surfaces; importing the trainer alone does not.
RUN python -c "\
import torch, transformers, peft, wandb; \
from transformers import Trainer, TrainingArguments, LlamaConfig, LlamaForCausalLM; \
from peft import LoraConfig, get_peft_model; \
get_peft_model(LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, vocab_size=128)), LoraConfig(target_modules=['q_proj'], r=4)); \
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| lora dispatch OK')"

ENV CUDA_CACHE_DISABLE=1

CMD ["/bin/bash"]
