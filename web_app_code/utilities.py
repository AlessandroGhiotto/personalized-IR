import pandas as pd
import re
import os
import torch
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import faiss
import joblib
from functools import partial
from streamlit import cache_data, cache_resource
import emoji
import nltk
from nltk.stem import PorterStemmer
from pyterrier.measures import *
import pyterrier as pt


@cache_resource
def download_nltk_stopwords():
    nltk.download("stopwords")
    return True


@cache_resource
def init_pyterrier():
    if not pt.java.started():
        pt.java.init()
    return True


# run them only once
download_nltk_stopwords()
init_pyterrier()
from nltk.corpus import stopwords


def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # remove emojis
    text = emoji.replace_emoji(text, "")
    # remove links
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # remove html tags
    # text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # lowercase verything
    text = text.lower()
    # remove backslashes
    text = re.sub(r"\\", "", text)
    # remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # remove whitespaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing whites
    text = text.strip()
    # apply spelling correction
    # text = TextBlob(text).correct()
    tokens = text.split()
    # remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # apply stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


@cache_data
def get_corpus(path="./web_app_data/subset_answers_with_users.json"):
    """
    Load the corpus of answers.
    """
    df = pd.read_json(path, orient="records", lines=True)
    df["doc_user_id"] = df["doc_user_id"].astype(str)
    return df


@cache_data
def get_list_of_tags(path="./web_app_data/list_tags.joblib"):
    """
    Get the list of tags.
    """
    return joblib.load(path)


def _get_tags_score(df, USER_TAGS):
    """
    get scores based on the tags of the user that asked the question
    and the user that have written the answer.

    used as argument of pyterrier.apply.doc_score()
        =>  the input is a ranked documents dataframe (batch), by query
            the output are the scores for each document in the batch
    """

    def compute_score(tags_uq, tags_ua):
        """
        tags_uq: set of tags of the user that asked the question
        tags_ua: set of tags of the user that wrote the answer
        """
        return len(tags_uq.intersection(tags_ua)) / (len(tags_uq) + 1)

    if not all(df["qid"] == df["qid"].iloc[0]):
        assert "Not all qids in the batch are equal"

    # get the tags of the user that asked the question
    # IS GIVEN AS INPUT DIRECTLY IN THE DATAFRAME
    tags_uq = df["user_tags"].iloc[0]

    # users that have written the answers
    uaS = df["doc_user_id"].values
    # get the tags of the users that have written the answers
    tags_uaS = [USER_TAGS.get(ua, {}) for ua in uaS]
    # compute the score for each answer
    scores = [compute_score(tags_uq, tags_ua) for tags_ua in tags_uaS]
    return scores


def _get_dense_scores(df, FAISS_INDEX, biencoder_model, text_field="query_unprocessed"):
    """
    get cosine similarity score with a biencoder model, with FAISS FlatIndex

    used as argument of pyterrier.apply.doc_score()
        =>  the input is a ranked documents dataframe (batch), by query
            the output are the scores for each document in the batch
    """
    if not all(df["qid"] == df["qid"].iloc[0]):
        assert "Not all qids in the batch are equal"
    # get the query unprocessed text
    query_text = df[text_field].iloc[0]
    # get the query embedding
    query_embedding = biencoder_model.encode(query_text).astype("float32")
    query_embedding = query_embedding / np.linalg.norm(
        query_embedding
    )  # normalize for cosine similarity

    # if we are reranking
    if "docid" in df.columns:
        # select the retrieved documents
        filter_ids = df["docid"].values
        id_selector = faiss.IDSelectorArray(np.array(filter_ids, dtype=np.int64))
        search_params = faiss.SearchParametersIVF(sel=id_selector)
        # rerank them
        k = len(filter_ids)
        distances, indices = FAISS_INDEX.search(
            np.array([query_embedding]), k, params=search_params
        )
    else:
        raise NotImplementedError("Not implemented for non reranking")

    # mapping {docid: score}
    score_mapping = {docid: score for docid, score in zip(indices[0], distances[0])}
    # get the scores in the original order (same as the input docids)
    scores_original_order = [score_mapping[docid] for docid in df["docid"]]
    return scores_original_order


###### MODELS ######


@cache_resource
def get_bm25(index_path="./web_app_data/index_bm25_users/data.properties"):
    """
    Get the BM25 retriever, with rank cut-off at 100.
    """
    bm25_index = pt.IndexFactory.of(index_path)
    bm25 = pt.terrier.Retriever(
        bm25_index,
        wmodel="BM25",
        controls={"c": 1.0, "bm25.k_1": 2.5},
        properties={
            "termpipelines": ""
        },  # stemmming and stompowrd removal is done manually with NLTK
        metadata=[
            "docno",
            "doc_user_id",
        ],  # ADD doc_user_id TO THE METADATA TO BE RETRIEVED (so we have it for the personalized search)
    )
    return bm25 % 100


@cache_resource
def get_biencoder(
    index_path="./web_app_data/MiniLM_faiss_IndexFlatIP.index",
    biencoder_model="sentence-transformers/all-MiniLM-L12-v2",
):
    """
    Get the biencoder model.
    """
    faiss_index = faiss.read_index(index_path)
    biencoder_model = SentenceTransformer(biencoder_model)
    get_dense_score = partial(
        _get_dense_scores, FAISS_INDEX=faiss_index, biencoder_model=biencoder_model
    )
    bi_enc = pt.apply.doc_score(get_dense_score, batch_size=64)
    return bi_enc


@cache_resource
def get_neural_reranker():
    """
    Get the dense retriever pipeline.
        - first stage retriever: BM25 with rank cut-off at 100
        - neural reranking: .9*bi_enc_norm + .1*bm25_norm
    """
    bm25 = get_bm25()
    bm25_norm = bm25 >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    bi_enc_norm = get_biencoder() >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    return bm25 >> 0.9 * bi_enc_norm + 0.1 * bm25_norm


@cache_resource
def get_personalized_pipeline():
    """
    Get the personalized pipeline.
        - first stage retriever: BM25 with rank cut-off at 100
        - personalized neural reranking: .7*bi_enc_norm + .1*bm25_norm + .2*tags_score_norm
    """
    bm25 = get_bm25()
    bm25_norm = bm25 >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    bi_enc_norm = get_biencoder() >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    USER_TAGS = joblib.load("./web_app_data/user_tags_full.joblib")
    tags_score = pt.apply.doc_score(
        partial(_get_tags_score, USER_TAGS=USER_TAGS), batch_size=64
    )
    tags_score_norm = tags_score >> pt.pipelines.PerQueryMaxMinScoreTransformer()
    return bm25 % 100 >> 0.7 * bi_enc_norm + 0.1 * bm25_norm + 0.2 * tags_score_norm
