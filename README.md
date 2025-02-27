# Personalized IR on SE-PQA

### Downloading files

Download all the files at the following link: [pir-notebooks-data.zip](https://drive.google.com/file/d/1f-R3uzit6zxAfESgPRusiXLo3qE0isf1/view?usp=sharing)

In this file we have the following folders:

- _cache_: used for storing the output of a PyTerrier Transformer
- _experiments_: all the experiments that we executed
- _index_sepqa_: all the pyterrier indexes that we have created
- _models_: Scikit-Learn trained models

### Notebooks

- notebook1-data-analisys: look at the data
- notebook2-baseline-retrieval: BM25 and TF-IDF
- notebook3-neural-reranking: reranking with a Bi-Encoder
- notebook4-query-expansion: expand the query with an LLM
- notebook5.1-personalize-ir: personalize the query with the Tags Score
- notebook5.2-personalize-ir: other scores for personalization
- notebook5.3-cold-start-problem: formulation of the Tags Score which is dependent on the number of questions written by the user
- notebook6-ltr-personalized-ir: learn to rank on top of the personalized information retrieval pipeline

### Requirements

The file `environment.yaml` is the output of the command `conda env export --no-builds > environment.yaml` executed on the environment in which we have run all the notebooks of the project.
