# Personalized IR on SE-PQA

## Live Demo

Try the web application here: [Personalized IR App](https://personalized-ir.streamlit.app/)

## Downloading Files

Download all the files at the following link: [pir-notebooks-data.zip](https://drive.google.com/file/d/1f-R3uzit6zxAfESgPRusiXLo3qE0isf1/view?usp=sharing)

In this file, we have the following folders:

- **_cache_**: used for storing the output of a PyTerrier Transformer
- **_experiments_**: all the experiments that we executed
- **_index_sepqa_**: all the PyTerrier indexes that we have created
- **_models_**: Scikit-Learn trained models

## Notebooks

- **notebook1-data-analysis**: Look at the data
- **notebook2-baseline-retrieval**: BM25 and TF-IDF
- **notebook3-neural-reranking**: Reranking with a Bi-Encoder
- **notebook4-query-expansion**: Expand the query with an LLM
- **notebook5.1-personalize-ir**: Personalize the query with the Tags Score
- **notebook5.2-personalize-ir**: Other scores for personalization
- **notebook5.3-cold-start-problem**: Formulation of the Tags Score dependent on the number of questions written by the user
- **notebook6-ltr-personalized-ir**: Learn to rank on top of the personalized information retrieval pipeline

## Requirements

The file `environment.yaml` is the output of the command:

```bash
conda env export --no-builds > environment.yaml
```

## Acknowledgment

This project uses the SE-PQA dataset from:

Kasela, P., Braga, M., Pasi, G., & Perego, R. (2023). SE-PQA: a Resource for Personalized Community Question Answering [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.10679181>
