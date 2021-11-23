## Effects of NER annotations’ density on pre-trained models in the context oflow resourced languages

This is repository for the course project COMS4054A - NLP  2021. It contains codes and necessary notebooks to get the resutls in the report of the same project.

### Table of Contents
- [Folder structure](#folder-structure)
- [Environment setup](#environment-setup)
    - [Hardware Requirements](#hardware-requirements)
    - [Software Requirements](#software-requirements)
- [Training and Evaluation](#training-and-evaluation)
    - [Reproducibility](#reproducibility)
- [Model Cards](#model-cards)
    - [BERT](#bert)
    - [RoBERTa](#roberta)
    - [Multilingual BERT (mBERT)](#bert)
    - [Multilingual BERT (mBERT) finetuned for NER](#multilingual-bert-(mbert)-finetuned-for-ner)
- [License](#license)

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

#### Hardware Requirements
- RAM 16 GB or more
- GPU with CUDA support (for faster training)

#### Software Requirements
- [Python](https://www.python.org/) (>= 3.8) and equivalent [pip](https://pypi.org/project/pip/)


### Training and Evaluation

All the notebooks to train and evaluate all the models can be found in notebooks directory as outline in the [Folder structure](#folder-structure) section.

#### Reproducibility

To reproduce the results of the report. Use the following hyperparameters:

- **Learning Rate** 5e-5
- **Batch Size** 32
- **Maximum Sequence Length** 164
- **Epochs** 30

### Model Cards

This section contains the description to various pre-trained models used as well as the link to model cards.

#### BERT
This is the [BERT base cased model](https://huggingface.co/bert-base-cased) trained on English text with 12 layers of transformers block with a hidden size of 768, 12 attention heads and 110 parameters. Here are the various model cards for different languages, kin, swa, pcm.

#### RoBERTa
This is the [RoBERTa base model](https://huggingface.co/roberta-base) with the same architecture as BERT. Here are the various model cards for different languages, [kin](), [swa](), [pcm]().

#### Multilingual BERT (mBERT)
This is the [BERT base multilingual cased model](https://huggingface.co/bert-base-multilingual-cased) is trained on 104 languages through masked-language modelling. Here are the various model cards for different languages, [kin](), [swa](), [pcm]().

#### Multilingual BERT (mBERT) finetuned for NER
This [model](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) is a fine-tuned version of the previous model on 10 high resourced languages for NER tasks. Here are the various model cards for different languages, [kin](), [swa](), [pcm]().

### License

All the code in this project is licensed under the [MIT license](https://www.apache.org/licenses/LICENSE-2.0).