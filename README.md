# Degeneration probe

A research pipeline for measuring LLM text degeneration (e.g. repetition loops)
and training a probe to detect it from hidden activations.

---

Please note that this is still *work in progress*, some of the scripts and approaches might not be most efficient, docs might not be complete.

For a complete list of TODOs, please refer to the [issues page](https://github.com/Luca-Sartori/degeneration-probe/issues).

## Background

Probes can be used to detect a model's behaviour by using its hidden activations. By training a small probe (usually a single layer MLP), we can predict the model's behaviour — in this case, whether it is producing *degenerating* (e.g. repetitive) text.


## Models supported

While the code was tested for `Apertus_8B_Instruct_2509` and `Meta_Llama_3.1_8B_Instruct`, it should work for any standard language model from `transformers` library. 


## Installation

The basic installation setup for a local machine:
```

# 1. Clone and enter
git clone https://github.com/Luca-Sartori/degeneration-probe
cd degeneration-probe

# 2. Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create env and install (with CUDA torch on Linux)
uv sync

# 4. Store your API keys
mkdir -p ~/keys
echo "hf_..." > ~/keys/.hf_token
echo "..."    > ~/keys/.wandb_key

# 5. Run a training job
uv run python scripts/train_probe.py model=llama training=no_lora dataset=our_long_form
```
---

To set up the environment on the clariden cluster, please follow the [cluster guide](cluster/README.md).




## Acknowledgements

This project started as a fork of [`swiss-ai/feature-probes`](https://github.com/swiss-ai/feature-probes),
created by Tymoteusz Kwieciński and supervised by Anna Hedström and Imanol Schlag,
shared under the [Apache 2.0](./LICENSE.md) license.

That repo was itself developed initially as a project for Large Scale AI Engineering together
with Klejdi Sevdari, Michał Korniak and Jack Peck — see the original
[repo](https://github.com/sevdari/hallucination_probes).

The initial project was developed as an extension of the paper [*Real-Time Detection of Hallucinated Entities in Long-Form Generation* Obeso et. al.](https://arxiv.org/abs/2509.03531).
