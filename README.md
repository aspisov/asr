# Automatic Speech Recognition Homework

## Overview

This repository implements the full pipeline required for the HSE DLA 2025 ASR homework. It builds on the official PyTorch template and extends it.

## Environment & Installation

> **Python**: 3.12 (required by Torch 2.7 which is used on Blackwell GPUs)
> **Package manager**: [uv](https://docs.astral.sh/uv/)

Clone the repository and install all dependencies:

```bash
uv sync
```

### Artifacts

The repository expects the following resources, all downloadable via notebook or CLI commands:

- Trained checkpoints (hosted on HuggingFace):
  - Final model: `https://huggingface.co/aspisov/asr/resolve/main/model_best-193347.pth`
- KenLM language model + vocabulary:
  - `https://openslr.org/resources/11/3-gram.arpa.gz`
  - `https://openslr.org/resources/11/librispeech-vocab.txt`

Place them under `saved/` or run the commands embedded in the demo notebook (see below).

## Training

The main schedule trains on LibriSpeech combined splits:

```bash
uv run python3 train.py -cn=train.yaml
```

## Evaluation & Inference

### LibriSpeech evaluation scripts


```bash
# test-clean
uv run python3 inference.py -cn=inference_test_clean.yaml inferencer.from_pretrained=path/to/checkpoint.pth

# test-other
uv run python3 inference.py -cn=inference_test_other.yaml inferencer.from_pretrained=path/to/checkpoint.pth
```

### Custom directory inference

To evaluate on an arbitrary folder structured as required by the homework (audio + optional transcriptions), run:

```bash
uv run python3 inference.py -cn=inference_custom.yaml inferencer.from_pretrained=path/to/checkpoint.pth custom_dataset.dataset_root=path/to/dataset
```

## Demo Notebook

`notebooks/demo_notebook.ipynb` is designed to run on Google Colab from a clean environment. It covers:

1. Cloning the repo and installing dependencies via `uv`
2. Downloading checkpoints, KenLM files, and vocab
3. Running inference on LibriSpeech custom dataset as well as LibriSpeech test-clean and test-other

## Experiments & Results

| Experiment | Checkpoint | CER (test-clean) | WER (test-clean) | CER (test-other) | WER (test-other) |
|------------|------------|------------|------------|------------|------------|
| Full training (20M parameters conformer) | model_best-193347.pth | 5.2 | 14.27 | 15.02 | 35.04 |

## Logging

Logging is done via CometML. You can find the report [here](https://www.comet.com/dmitriy-aspisov/pytorch-template-asr-example/view/new/panels).

## License

[MIT](./LICENSE)
