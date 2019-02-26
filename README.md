# PyTorch code2vec

This repo contains an implementation of [code2vec: Learning Distributed Representations of Code](https://arxiv.org/abs/1803.09473). 

## Requirements

- Python 3+
- PyTorch 1.0
- A CUDA compatible GPU

## Quickstart

1. `./download_preprocessed.sh` to get the pre-processed datasets from the code2vec and the [code2seq](https://arxiv.org/abs/1808.01400) papers.
  - Note: by default the 3 datasets from code2seq are commented out.
1. `python run.py` 

## To-Do

- Graph of results
- Inference code
- Embedding exploration
