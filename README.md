# Repository for "Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions"
### Authors: Clement Neo, Shay B. Cohen, and Fazl Barez

This repository contains the code and data for the paper "Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions". The paper explores the interplay between attention heads and specialized "next-token" neurons in the Multilayer Perceptron (MLP) layer of transformers, focusing on how these components interact to predict specific tokens.

## Usage Instructions

1. **Run the notebooks**. The main notebooks in `notebooks/` are notebooks 1-7, while 8 and 9 are additional experiments (like head ablation and neuron ablation). The main notebooks save data to `experiment_data/`.

### References

To cite this work, please use the following BibTeX entry:

```bibtex
@misc{neo2024interpreting,
  title={Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions}, 
  author={Clement Neo and Shay B. Cohen and Fazl Barez},
  year={2024},
  eprint={2402.15055},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
