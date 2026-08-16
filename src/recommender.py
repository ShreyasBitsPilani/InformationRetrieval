import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_FILE = (
    "data/processed/processed_documents.csv"
)

VECTORIZER_FILE = (
    "index/tfidf_vectorizer.pkl"
)

MATRIX_FILE = (
    "index/tfidf_matrix.pkl"
)

RESULTS_DIR = "results"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    if not os.path.exists(
        PROCESSED_FILE
    ):

        raise FileNotFoundError(
            "Processed documents not found.\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(
        PROCESSED_FILE
    )

    if df.empty:

        raise ValueError(
            "Document dataset is empty."
        )

    return df


# ============================================================
# LOAD TF-IDF MODEL
# ============================================================

def load_tfidf_model():

    if not os.path.exists(
        VECTORIZER_FILE
    ):

        raise FileNotFoundError(
            "TF-IDF vectorizer not found.\n"
            "Run search_engine.py first."
        )

    if not os.path.exists(
        MATRIX_FILE
    ):

        raise FileNotFoundError(
            "TF-IDF matrix not found.\n"
            "Run search_engine.py first."
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


# ============================================================
# FIND DOCUMENT
# ============================================================

def find_document(
    df,
    doc_id
):

    matches = df[
        df["doc_id"].astype(str)
        == str(doc_id)
    ]

    if matches.empty:

        return None

    return matches.index[0]


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def recommend(
    doc_id,
    df,
    tfidf_matrix,
    top_k=5
):
    """
    Generate content-based recommendations.

    Parameters:
        doc_id        : selected document
        df            : document dataframe
        tfidf_matrix  : TF-IDF document matrix
        top_k         : number of recommendations

    Returns:
        DataFrame containing recommendations
    """

    # --------------------------------------------------------
    # Find selected document
    # --------------------------------------------------------

    document_index = find_document(
        df,
        doc_id
    )

    if document_index is None:

        raise ValueError(
            f"Document {doc_id} not found."
        )

    # --------------------------------------------------------
    # Get selected document vector
    # --------------------------------------------------------

    selected_vector = (
        tfidf_matrix[
            document_index
        ]
    )

    # --------------------------------------------------------
    # Calculate similarity against all documents
    # --------------------------------------------------------

    similarity_scores = (
        cosine_similarity(
            selected_vector,
            tfidf_matrix
        )
        .flatten()
    )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    recommendations = []

    for index, score in enumerate(
        similarity_scores
    ):

        # Do not recommend the same document
        if index == document_index:
            continue

        row = df.iloc[
            index
        ]

        recommendations.append(
            {
                "doc_id":
                    row["doc_id"],

                "title":
                    row["title"],

                "category":
                    row["category"],

                "url":
                    row["url"],

                "similarity":
                    float(score)
            }
        )

    recommendations_df = pd.DataFrame(
        recommendations
    )

    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    recommendations_df = (
        recommendations_df
        .sort_values(
            by="similarity",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add recommendation rank
    # --------------------------------------------------------

    recommendations_df.insert(
        0,
        "rank",
        range(
            1,
            len(recommendations_df) + 1
        )
    )

    return recommendations_df


# ============================================================
# DISPLAY DOCUMENT
# ============================================================

def display_selected_document(
    df,
    doc_id
):

    document_index = find_document(
        df,
        doc_id
    )

    if document_index is None:

        return

    row = df.iloc[
        document_index
    ]

    print("\n")
    print("=" * 80)
    print("SELECTED DOCUMENT")
    print("=" * 80)

    print(
        f"Document ID : {row['doc_id']}"
    )

    print(
        f"Title       : {row['title']}"
    )

    print(
        f"Category    : {row['category']}"
    )

    print(
        f"URL         : {row['url']}"
    )

    print("=" * 80)


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

def display_recommendations(
    recommendations
):

    print("\n")
    print("=" * 80)
    print("CONTENT-BASED RECOMMENDATIONS")
    print("=" * 80)

    if recommendations.empty:

        print(
            "\nNo recommendations found."
        )

        return

    for _, row in (
        recommendations.iterrows()
    ):

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
            f"Similarity : "
            f"{row['similarity']:.6f}"
        )

        print(
            f"URL        : "
            f"{row['url']}"
        )

    print("\n" + "=" * 80)


# ============================================================
# SAVE RECOMMENDATIONS
# ============================================================

def save_recommendations(
    doc_id,
    recommendations
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    filename = os.path.join(
        RESULTS_DIR,
        f"recommendations_{doc_id}.csv"
    )

    recommendations.to_csv(
        filename,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nRecommendations saved to: "
        f"{filename}"
    )


# ============================================================
# INTERACTIVE RECOMMENDATION
# ============================================================

def interactive_recommendation(
    df,
    tfidf_matrix
):

    print("\n")
    print("=" * 80)
    print("DOCUMENT RECOMMENDATION SYSTEM")
    print("=" * 80)

    print(
        "\nAvailable documents:"
    )

    # Display first 30 documents
    display_df = df[
        [
            "doc_id",
            "title",
            "category"
        ]
    ].head(30)

    print(
        display_df.to_string(
            index=False
        )
    )

    print(
        "\nType 'exit' to stop."
    )

    while True:

        doc_id = input(
            "\nEnter document ID: "
        ).strip()

        if doc_id.lower() == "exit":

            break

        if not doc_id:

            print(
                "Please enter a document ID."
            )

            continue

        # Check document
        if find_document(
            df,
            doc_id
        ) is None:

            print(
                f"Document {doc_id} "
                f"does not exist."
            )

            continue

        top_k_input = input(
            "Number of recommendations "
            "(default 5): "
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

        display_selected_document(
            df,
            doc_id
        )

        recommendations = recommend(
            doc_id,
            df,
            tfidf_matrix,
            top_k
        )

        display_recommendations(
            recommendations
        )

        save_recommendations(
            doc_id,
            recommendations
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONTENT-BASED DOCUMENT RECOMMENDER")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_documents()

    print(
        f"\nDocuments loaded: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Load TF-IDF
    # --------------------------------------------------------

    (
        vectorizer,
        tfidf_matrix
    ) = load_tfidf_model()

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    # --------------------------------------------------------
    # Start interactive recommendation
    # --------------------------------------------------------

    interactive_recommendation(
        df,
        tfidf_matrix
    )

    print(
        "\nRecommendation system closed."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()