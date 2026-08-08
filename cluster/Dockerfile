FROM nvcr.io/nvidia/pytorch:25.06-py3

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

ARG BRANCH=main

WORKDIR /workspace

RUN git clone --branch ${BRANCH} --single-branch https://github.com/Luca-Sartori/degeneration-probe.git
WORKDIR /workspace/degeneration-probe

# should I really do break-system-packages?
RUN uv pip install -e . --system --break-system-packages

# A dependency that disagrees with the base image's torch only shows up on
# import, so check it here rather than in a job that has already queued.
# Importing the trainer pulls in the quantizer backends, which is where a
# mismatched accelerator library surfaces first.
RUN python -c "\
import torch, transformers, peft, wandb; \
from transformers import Trainer, TrainingArguments; \
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

ENV CUDA_CACHE_DISABLE=1

CMD ["/bin/bash"]
