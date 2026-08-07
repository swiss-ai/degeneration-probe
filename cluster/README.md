# Cluster jobs

Submit the default degeneration training job with:

```bash
sbatch cluster/train.sbatch
```

The job runs `scripts/train_probe.py`, whose Hydra defaults already select the
Apertus model, the `degeneration` training profile and the local Apertus dataset
build. Override a training field after the script command when needed, for
example `training.loss.name=mse`.

Dataset-generation jobs live under `cluster/utils/dataset/` and use
`configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml` by default.
