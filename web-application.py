import streamlit as st
import pandas as pd


def main():
    from web_app_code import utilities

    st.title("Personalized IR", anchor=False)
    st.markdown(
        """Author: &nbsp; Alessandro Ghiotto &nbsp;
        [![Personal Profile](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AlessandroGhiotto)  
        Select a model, choose the tags, and enter your query to retrieve the most relevant documents from 
        a subset of Stack Exchange answers.
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Models details"):
        st.markdown(
            r"""
            - **Baseline**: BM25 model with $c = 1.0$ and $k_1 = 2.5$. The text is preprocessed by removing stopwords, stemming, and lowercasing.
            - **Neural Reranker**: BM25 % 100 >> .9 $\times$ BiEncoder + .1 $\times$BM25,  
                where BiEncoder is a dense retriever which
                uses [*"sentence-transformers/all-MiniLM-L12-v2"*](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 
                as embedding model. The score is given by the cosine similarity between the query and the document embeddings.
            - **Personalized**: BM25 % 100 >> .7 $\times$ BiEncoder + .1 $\times$ BM25 + .2 $\times$ TagsScore,  
                where TagsScore is a personalized
                score based on the tags of the user given as input and the tags of the user that have written the answer.

            **Notes**:
            - % k : is the rank cut-off at k. Only the top k documents are kept.  
            - \>\>  : is the "compose" operator (then), it allow to create a pipeline of models.  
            - The "linear combination" of the models creates a new score based on the weighted sum of the scores of the models. All the scores are normalized.
            - [FAISS](https://github.com/facebookresearch/faiss) is used for the dense retrieval and [PyTerrier](https://pyterrier.readthedocs.io/en/latest/) 
            is used for everything else.
            """
        )

    if st.session_state.get("tags") is None:
        st.session_state.search = False

    with st.form("search_form"):
        # MODEL
        model_str = st.segmented_control(
            "Retrieval Model",
            options=["Baseline", "Neural Reranker", "Personalized"],
            selection_mode="single",
        )

        # TAGS
        list_all_tags = utilities.get_list_of_tags()
        tags = st.multiselect(
            "Select tags",
            list_all_tags,
        )

        # QUERY
        query = st.text_input("Write you query here", value="")

        # N RESULTS
        n_results = st.number_input(
            "Number of results to show", min_value=1, max_value=30, value=5, step=1
        )

        # SUBMIT
        submit = st.form_submit_button("Search &nbsp; :mag:")
        if submit:
            if (model_str and tags != [] and query != "") or (
                model_str != "Personalized" and model_str and query != ""
            ):
                st.balloons()
                st.session_state.search = True
                # st.succes('sium')
            else:
                st.session_state.search = False
                st.warning("Please fill in all values")

    if submit and st.session_state.search:
        preprocess_query = utilities.preprocess_text(query)
        input_data = {
            "qid": 0,
            "query_unprocessed": query,
            "query": preprocess_query,
            "user_tags": set(tags),
        }
        input_df = pd.DataFrame([input_data])
        if model_str == "Baseline":
            model = utilities.get_bm25()
        if model_str == "Neural Reranker":
            model = utilities.get_neural_reranker()
        if model_str == "Personalized":
            model = utilities.get_personalized_pipeline()

        results = model.transform(input_df)
        if len(results) == 0:
            st.warning(
                "No results found. None of the words in the question are in the vocabulary."
            )
            st.stop()

        st.write("Top results:")
        results = results.sort_values("score", ascending=False)
        results = results.iloc[0:n_results]

        corpus_df = utilities.get_corpus()
        results = results.merge(corpus_df, on="docno", how="left")

        for i in range(len(results)):
            text = results["text"].iloc[i]
            if len(text) > 100:
                text = text[:100] + "..."
                with st.expander(f"**{i+1}.** {text}..."):  # First 100 chars or so
                    st.write(results["text"].iloc[i])  # Full text when expanded
            else:
                with st.container(border=True):
                    st.write(f"**{i+1}.** {text}")
        if len(results) < n_results:
            st.warning("Less results than requested. Try to write a few more words.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="PIR",
        page_icon=":mag:",
        menu_items={
            "Report a bug": "https://github.com/AlessandroGhiotto/personalized-IR/issues",
        },
        # layout="wide",
    )
    main()
