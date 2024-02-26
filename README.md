# Repository for "Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions"
### Authors: Clement Neo, Shay B. Cohen, and Fazl Barez

This repository contains the code and data for the paper "Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions". The paper explores the interplay between attention heads and specialized "next-token" neurons in the Multilayer Perceptron (MLP) layer of transformers, focusing on how these components interact to predict specific tokens.

## Usage Instructions

1. **Download the Data**: The dataset used in this paper involves attention head activations and next-token neuron activations across various prompts. Use the script below to download the necessary data. Feel free to modify the script for different sets of prompts or tokens.

   ```
   bash
   ./src/data/download_data.sh

### Run the Analysis: 
To reproduce the findings of the paper, execute the following command. This script processes the downloaded data to identify attention heads and next-token neurons that show specialized activation patterns.

`python3 run.py`

### Data Analysis and Visualization: 

Explore the analysis of attention-MLP interactions through the provided Jupyter notebooks. These notebooks include visualizations and tables that detail the relationship between attention mechanisms and next-token prediction.

#### Notebook for General Analysis: 

`notebooks/analysis.ipynb`
#### Notebook for Attention Head Specialization:

`notebooks/attention_specialization.ipynb`

### Development Setup
Create a Python Virtual Environment and activate it:

`python3.10 -m venv venv`

`source venv/bin/activate`

Install Dependencies listed in `requirements.txt:`

### Explore Source Code: 
The `src` directory contains all the necessary scripts for data downloading, model analysis, and visualization.
Data Scripts: `src/data`
Model Analysis: `src/models`
Visualization: `src/visualization`

#### Modify Path Variables: 
Adjust path variables in `src/__init__.py` as necessary to point to your data, models, and outputs.

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