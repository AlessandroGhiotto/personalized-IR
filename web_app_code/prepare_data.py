import pandas as pd
import re
import os
import torch
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import faiss
import joblib
import pyterrier as pt
from utilities import preprocess_text


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_corpus():
    df = pd.read_json("PIR_data/answer_retrieval/subset_answers.json", orient="index")
    df = df.reset_index()
    df.columns = ["docno", "text"]
    df = df.reset_index(drop=True)
    return df


def attach_user_id(
    corpus_df, save_path="./web_app_data/subset_answers_with_users.json"
):
    """Add the doc_user_id to the corppus_df"""

    ### WHOLE ANSWERS DATASET
    all_answers_df = pd.read_csv(
        "PIR_data/answers.csv", engine="python", on_bad_lines="skip"
    )
    docno_set = set(corpus_df["docno"].values)

    # answer -> user_id
    answer_to_user = {}
    for answer, user in zip(all_answers_df["Id"], all_answers_df["AccountId"]):
        if answer in docno_set and not pd.isnull(user):
            answer_to_user[answer] = int(user)

    del all_answers_df

    # Check if there are user_id that are equal to 0
    num_user_id_zero = (pd.Series(answer_to_user.values()) == 0).sum()
    if num_user_id_zero != 0:
        print("There are user_ids = 0, We can't use '0' for filling the missing values")

    # set to 0 user_ids that are not in the answer_to_user dictionary
    # store them as string for storing them in pyterrier index
    corpus_df["doc_user_id"] = (
        corpus_df["docno"].apply(lambda x: answer_to_user.get(x, "0")).astype(str)
    )

    ### SAVE
    corpus_df.to_json(save_path, orient="records", lines=True)
    # read it: pd.read_json(save_path, orient="records", lines=True)

    return corpus_df


def create_user_tags_index(
    corpus_df_with_users, path_USER_TAGS="./web_app_data/user_tags_full.joblib"
):
    """
    Get all the tags from the complete dataset

    USERS_TAGS: dictionary associating to each user a set of tags

    USERS_TAGS = {
        "user1_id": {"tag1", "tag2",...},
        "user2_id": {"tag1", ...},
    }

    In the notebooks we had the timestamps associated to the tags, but here we don't care
    """

    all_questions_df = pd.read_csv(
        "PIR_data/questions.csv", engine="python", on_bad_lines="skip"
    )[["Id", "AccountId", "Tags"]]
    all_questions_df = all_questions_df.rename(
        columns={
            "Id": "qid",
            "AccountId": "user_id",
            "Tags": "tags",
        }
    )

    # set the user_id to 0 if it is null, and convert to string
    all_questions_df["user_id"] = (
        all_questions_df["user_id"].fillna(0).astype(int).astype(str)
    )

    # convert tags to a list <tag1><tag2>...<tagN> -> [tag1, tag2, ..., tagN]
    all_questions_df["tags"] = all_questions_df["tags"].apply(
        lambda x: list(re.findall(r"<(.*?)>", x)) if not pd.isnull(x) else []
    )

    # users we care about (the ones in the corpus)
    useful_users = set(corpus_df_with_users["doc_user_id"].values)

    ##### Build USER_TAGS dictionary
    USER_TAGS = {}
    for user, group in all_questions_df.groupby("user_id"):
        if user in useful_users:
            # Convert the "tags" column to a flat set
            USER_TAGS[user] = set(tag for tags in group["tags"] for tag in tags)

    del all_questions_df

    # user_id = 0 was used for filling the missing values
    # we set it to an empty set so we get as score 0, since we don't have a profile of the user
    USER_TAGS["0"] = set()
    joblib.dump(USER_TAGS, path_USER_TAGS)

    return USER_TAGS


def create_list_of_tags(
    corpus_df_with_users,
    path_TAGS_LIST="./web_app_data/list_tags.joblib",
    number_of_tags=5000,
):
    """
    Get the list of tags to be shown in the multi-select widget.
    we keep the first 'number_of_tags' tags
    """
    all_questions_df = pd.read_csv(
        "PIR_data/questions.csv", engine="python", on_bad_lines="skip"
    )[["Id", "AccountId", "Tags"]]
    all_questions_df = all_questions_df.rename(
        columns={
            "Id": "qid",
            "AccountId": "user_id",
            "Tags": "tags",
        }
    )

    # set the user_id to 0 if it is null, and convert to string
    all_questions_df["user_id"] = (
        all_questions_df["user_id"].fillna(0).astype(int).astype(str)
    )

    # convert tags to a list <tag1><tag2>...<tagN> -> [tag1, tag2, ..., tagN]
    all_questions_df["tags"] = all_questions_df["tags"].apply(
        lambda x: list(re.findall(r"<(.*?)>", x)) if not pd.isnull(x) else []
    )

    # users we care about (the ones in the corpus)
    useful_users = set(corpus_df_with_users["doc_user_id"].values)

    filtered_questions_df = all_questions_df[
        all_questions_df["user_id"].isin(useful_users)
    ]

    # Explode the sets into individual rows
    df_exploded = filtered_questions_df.explode("tags")

    # Count occurrences
    tag_counts = df_exploded["tags"].value_counts()

    # Keep the first 'number_of_tags' tags
    tags_list = tag_counts.index[:number_of_tags].tolist()

    joblib.dump(tags_list, path_TAGS_LIST)


def create_sparse_index(
    corpus_df, path_index="./web_app_data/index_bm25_users/", RECREATE_INDEX=False
):
    """
    Create the sparse index for BM25.
    It as as metadata also the user_id of the document (for the personalized search)
    """

    if RECREATE_INDEX or not os.path.exists(path_index + "data.properties"):
        corpus_df_indexing = corpus_df.copy()
        # preprocess the text (lowercase, remove stopwords, stem, ...)
        corpus_df_indexing["text"] = corpus_df_indexing["text"].apply(
            lambda x: preprocess_text(x)
        )
        # put a placeholder for empty documents. so the number docid is consistent
        # If I delete them, I will get n docs in the dense index and n-(empty_docs) in the BM25 index
        # use 'the' as a placeholder so I'm sure to get 0 similarity (the other docs have no stopwords)
        corpus_df_indexing["text"] = corpus_df_indexing["text"].replace("", "the")
        max_text = corpus_df_indexing["text"].apply(len).max()
        max_docno = corpus_df_indexing["docno"].apply(len).max()
        max_user_id = corpus_df_indexing["doc_user_id"].apply(len).max()

        indexer = pt.IterDictIndexer(path_index, stemmer=None, stopwords=None)
        indexer.index(
            corpus_df_indexing.to_dict(orient="records"),
            fields={"text": max_text},
            meta={"docno": max_docno, "doc_user_id": max_user_id},
        )

        del corpus_df_indexing


def create_dense_index(
    corpus_df,
    path_index="./web_app_data/MiniLM_faiss_IndexFlatIP.index",
    biencoder_model="sentence-transformers/all-MiniLM-L12-v2",
):
    """
    Create the dense index for the biencoder model. We use the FAISS library for the cosine similarity.
    """
    # load the SentenceTransformer model (miniLM)
    biencoder_model = SentenceTransformer(biencoder_model)

    # compute all the embeddings
    corpus_embeddings = biencoder_model.encode(
        corpus_df["text"].tolist(), show_progress_bar=True
    )
    # corpus_embeddings.shape: (9398, 384) -> (num_docs, embedding_dim)

    # normalize the embeddings
    corpus_embeddings = np.array(corpus_embeddings, dtype="float32")
    corpus_embeddings = corpus_embeddings / np.linalg.norm(
        corpus_embeddings, axis=1, keepdims=True
    )

    # create a FAISS index for cosine similarity (IP = Inner Product)
    FAISS_INDEX = faiss.IndexFlatIP(corpus_embeddings.shape[1])

    # sdd vectors to the FAISS index
    FAISS_INDEX.add(corpus_embeddings)

    # save the index
    faiss.write_index(FAISS_INDEX, path_index)


def main():
    if not pt.java.started():
        pt.java.init()

    set_seed()
    print("Loaded corpus")
    corpus_df = load_corpus()
    print("Attaching user_id to the corpus")
    corpus_df = attach_user_id(corpus_df)
    print("creating 'USER_TAGS'")
    create_user_tags_index(corpus_df)
    print("creating 'TAGS_LIST'")
    create_list_of_tags(corpus_df)
    print("creating BM25 index")
    create_sparse_index(corpus_df)
    print("creating dense index")
    create_dense_index(corpus_df)


if __name__ == "__main__":
    main()
