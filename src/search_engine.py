import os
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import our preprocessing function
from preprocessing import preprocess_text


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_FILE = (
    "data/processed/processed_documents.csv"
)

INVERTED_INDEX_FILE = (
    "index/inverted_index.json"
)

RESULTS_DIR = "results"

VECTORIZER_FILE = (
    "index/tfidf_vectorizer.pkl"
)

MATRIX_FILE = (
    "index/tfidf_matrix.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_documents():
    """
    Load processed documents.
    """

    if not os.path.exists(PROCESSED_FILE):

        raise FileNotFoundError(
            "Processed documents not found.\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(
        PROCESSED_FILE
    )

    if df.empty:

        raise ValueError(
            "Processed document dataset is empty."
        )

    return df


# ============================================================
# LOAD INVERTED INDEX
# ============================================================

def load_inverted_index():
    """
    Load the inverted index.
    """

    if not os.path.exists(
        INVERTED_INDEX_FILE
    ):

        raise FileNotFoundError(
            "Inverted index not found.\n"
            "Run indexing.py first."
        )

    with open(
        INVERTED_INDEX_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        index = json.load(file)

    return index


# ============================================================
# BUILD TF-IDF MODEL
# ============================================================

def build_tfidf_model(df):
    """
    Build TF-IDF matrix for the complete corpus.
    """

    print(
        "\nBuilding TF-IDF representation..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b"
    )

    matrix = vectorizer.fit_transform(
        df["processed_content"].fillna("")
    )

    # Save vectorizer
    with open(
        VECTORIZER_FILE,
        "wb"
    ) as file:

        pickle.dump(
            vectorizer,
            file
        )

    # Save matrix
    with open(
        MATRIX_FILE,
        "wb"
    ) as file:

        pickle.dump(
            matrix,
            file
        )

    print(
        f"TF-IDF matrix shape: "
        f"{matrix.shape}"
    )

    return (
        vectorizer,
        matrix
    )


# ============================================================
# LOAD OR BUILD TF-IDF MODEL
# ============================================================

def load_tfidf_model(df):
    """
    Load previously saved TF-IDF model.
    If it doesn't exist, build it.
    """

    if (
        os.path.exists(VECTORIZER_FILE)
        and
        os.path.exists(MATRIX_FILE)
    ):

        print(
            "\nLoading existing TF-IDF model..."
        )

        with open(
            VECTORIZER_FILE,
            "rb"
        ) as file:

            vectorizer = pickle.load(
                file
            )

        with open(
            MATRIX_FILE,
            "rb"
        ) as file:

            matrix = pickle.load(
                file
            )

        return (
            vectorizer,
            matrix
        )

    return build_tfidf_model(
        df
    )


# ============================================================
# GET CANDIDATE DOCUMENTS
# ============================================================

def get_candidate_documents(
    query_tokens,
    inverted_index
):
    """
    Use the inverted index to identify candidate documents.

    Only documents containing at least one query term
    are considered.
    """

    candidate_documents = set()

    for token in query_tokens:

        if token in inverted_index:

            postings = (
                inverted_index[token]
            )

            candidate_documents.update(
                postings.keys()
            )

    return candidate_documents


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    df,
    inverted_index,
    vectorizer,
    tfidf_matrix,
    top_k=10
):
    """
    Search the document collection.

    Steps:

    1. Preprocess query
    2. Find candidate documents using inverted index
    3. Convert query to TF-IDF
    4. Calculate cosine similarity
    5. Rank documents
    6. Return Top-K
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        return pd.DataFrame()

    # --------------------------------------------------------
    # Preprocess query
    # --------------------------------------------------------

    processed_query = preprocess_text(
        query
    )

    if not processed_query:

        return pd.DataFrame()

    query_tokens = (
        processed_query.split()
    )

    print(
        f"\nOriginal query: {query}"
    )

    print(
        f"Processed query: "
        f"{processed_query}"
    )

    # --------------------------------------------------------
    # Candidate retrieval
    # --------------------------------------------------------

    candidate_documents = (
        get_candidate_documents(
            query_tokens,
            inverted_index
        )
    )

    print(
        f"Candidate documents: "
        f"{len(candidate_documents)}"
    )

    # No matching documents
    if not candidate_documents:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Convert query into TF-IDF vector
    # --------------------------------------------------------

    query_vector = (
        vectorizer.transform(
            [processed_query]
        )
    )

    # --------------------------------------------------------
    # Create document ID mapping
    # --------------------------------------------------------

    doc_id_to_index = {
        str(doc_id): index
        for index, doc_id
        in enumerate(
            df["doc_id"]
        )
    }

    candidate_indices = []

    candidate_doc_ids = []

    for doc_id in candidate_documents:

        if doc_id in doc_id_to_index:

            candidate_indices.append(
                doc_id_to_index[doc_id]
            )

            candidate_doc_ids.append(
                doc_id
            )

    if not candidate_indices:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Select candidate vectors
    # --------------------------------------------------------

    candidate_matrix = (
        tfidf_matrix[
            candidate_indices
        ]
    )

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = (
        cosine_similarity(
            query_vector,
            candidate_matrix
        )
        .flatten()
    )

    # --------------------------------------------------------
    # Create results
    # --------------------------------------------------------

    results = []

    for i, score in enumerate(
        similarities
    ):

        original_index = (
            candidate_indices[i]
        )

        row = df.iloc[
            original_index
        ]

        results.append(
            {
                "doc_id":
                    row["doc_id"],

                "title":
                    row["title"],

                "category":
                    row["category"],

                "url":
                    row["url"],

                "score":
                    float(score),

                "word_count":
                    row["word_count"],

                "processed_content":
                    row["processed_content"]
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="score",
        ascending=False
    )

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    results_df = (
        results_df
        .head(top_k)
        .reset_index(drop=True)
    )

    results_df.insert(
        0,
        "rank",
        range(
            1,
            len(results_df) + 1
        )
    )

    return results_df


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def display_results(
    query,
    results
):
    """
    Print search results in a readable format.
    """

    print("\n")
    print("=" * 80)

    print(
        f"SEARCH RESULTS FOR: {query}"
    )

    print("=" * 80)

    if results.empty:

        print(
            "\nNo relevant documents found."
        )

        return

    for _, row in results.iterrows():

        print("\n" + "-" * 80)

        print(
            f"Rank       : "
            f"{row['rank']}"
        )

        print(
            f"Document ID: "
            f"{row['doc_id']}"
        )

        print(
            f"Title      : "
            f"{row['title']}"
        )

        print(
            f"Category   : "
            f"{row['category']}"
        )

        print(
            f"Score      : "
            f"{row['score']:.6f}"
        )

        print(
            f"URL        : "
            f"{row['url']}"
        )

        print(
            f"Word Count : "
            f"{row['word_count']}"
        )

    print("\n" + "=" * 80)


# ============================================================
# SAVE SEARCH RESULTS
# ============================================================

def save_results(
    query,
    results
):
    """
    Save search results to CSV.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    safe_query = (
        query
        .lower()
        .replace(" ", "_")
    )

    safe_query = (
        "".join(
            character
            for character in safe_query
            if character.isalnum()
            or character == "_"
        )
    )

    filename = os.path.join(
        RESULTS_DIR,
        f"search_{safe_query}.csv"
    )

    results.to_csv(
        filename,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nResults saved to: "
        f"{filename}"
    )


# ============================================================
# INTERACTIVE SEARCH
# ============================================================

def interactive_search(
    df,
    inverted_index,
    vectorizer,
    tfidf_matrix
):
    """
    Run an interactive command-line search.
    """

    print("\n")
    print("=" * 80)
    print("INTERACTIVE INFORMATION RETRIEVAL SEARCH")
    print("=" * 80)

    print(
        "\nType 'exit' to stop."
    )

    while True:

        query = input(
            "\nEnter search query: "
        ).strip()

        if query.lower() == "exit":

            break

        if not query:

            print(
                "Please enter a query."
            )

            continue

        top_k_input = input(
            "Enter number of results (default 5): "
        ).strip()

        if top_k_input:

            try:

                top_k = int(
                    top_k_input
                )

                if top_k <= 0:

                    top_k = 5

            except ValueError:

                top_k = 5

        else:

            top_k = 5

        results = search(
            query=query,
            df=df,
            inverted_index=inverted_index,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_k=top_k
        )

        display_results(
            query,
            results
        )

        if not results.empty:

            save_results(
                query,
                results
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "TF-IDF INFORMATION RETRIEVAL SEARCH ENGINE"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    print(
        "\nLoading documents..."
    )

    df = load_documents()

    print(
        f"Documents loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Load inverted index
    # --------------------------------------------------------

    print(
        "\nLoading inverted index..."
    )

    inverted_index = (
        load_inverted_index()
    )

    print(
        f"Indexed terms: "
        f"{len(inverted_index)}"
    )

    # --------------------------------------------------------
    # Load/build TF-IDF
    # --------------------------------------------------------

    (
        vectorizer,
        tfidf_matrix
    ) = load_tfidf_model(
        df
    )

    # --------------------------------------------------------
    # Start interactive search
    # --------------------------------------------------------

    interactive_search(
        df,
        inverted_index,
        vectorizer,
        tfidf_matrix
    )

    print(
        "\nSearch engine closed."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()