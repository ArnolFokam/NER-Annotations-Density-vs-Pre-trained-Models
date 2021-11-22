## Effects of NER annotations’ density on pre-trained models in the context oflow resourced languages

This repository contains code for the course project COMS4054A - NLP  2021. It contains codes and necessary notebooks to get the resutls from the report of the same project.

### Table of Contents
- [Folder structure](#folder-structure)
- [Environment setup](#environment-setup)
- [Training and Evaluation](#training-and-evaluation)

### Folder structure

```
Project
├── data
│   └── {lang} -> Dataset corpus in lang corpus
│           ├── train.txt -> train set
│           ├── dev.txt -> dev set
│           └── test.txt -> test set
├── images -> Images generated from various analysis
├── notebooks.
│   ├── cap_training_data.ipynb -> Jupyter notebook to create preprocessed corpus.
|   ├── results_analysis.ipynb -> Jupyter notebook to extract model evaluation files and generate plots.
│   └── train_ner.ipynb -> Jupyter notebook to train all the models and evaluate them.
└── README.md -> project description
```

### Environment setup

#### Requirements

##### Hardware
- RAM 16 GB or more
- GPU with CUDA support (for faster training)

##### Software
- [Python](https://www.python.org/) (>= 3.8) and equivalent [pip](https://pypi.org/project/pip/)


### Training and Evaluation

All the notebooks to train and evaluate all the models can be found in notebooks directory as outline in the [Folder structure](#folder-structure) section