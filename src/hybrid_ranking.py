import os

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PAGERANK_FILE = (
    "results/pagerank_scores.csv"
)

RESULTS_DIR = "results"


# Default weights
TFIDF_WEIGHT = 0.70
PAGERANK_WEIGHT = 0.30


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def normalize_scores(
    scores
):
    """
    Normalize scores to the range 0-1.
    """

    scores = np.array(
        scores,
        dtype=float
    )

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:

        return np.ones_like(
            scores
        )

    return (
        (scores - min_score)
        /
        (max_score - min_score)
    )


# ============================================================
# LOAD PAGERANK
# ============================================================

def load_pagerank():

    if not os.path.exists(
        PAGERANK_FILE
    ):

        raise FileNotFoundError(
            "pagerank_scores.csv not found.\n"
            "Run pagerank.py first."
        )

    return pd.read_csv(
        PAGERANK_FILE
    )


# ============================================================
# LOAD TF-IDF SEARCH RESULTS
# ============================================================

def load_search_results():

    files = []

    if not os.path.exists(
        RESULTS_DIR
    ):

        return files

    for filename in os.listdir(
        RESULTS_DIR
    ):

        if (
            filename.startswith(
                "search_"
            )
            and
            filename.endswith(
                ".csv"
            )
        ):

            files.append(
                os.path.join(
                    RESULTS_DIR,
                    filename
                )
            )

    return files


# ============================================================
# HYBRID RANKING
# ============================================================

def apply_hybrid_ranking(
    search_results,
    pagerank_df,
    tfidf_weight=0.70,
    pagerank_weight=0.30
):
    """
    Combine TF-IDF and PageRank.

    Final Score =
        TF-IDF Weight × TF-IDF
        +
        PageRank Weight × normalized PageRank
    """

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = pd.merge(
        search_results,
        pagerank_df[
            [
                "doc_id",
                "pagerank"
            ]
        ],
        on="doc_id",
        how="left"
    )

    # Missing PageRank becomes zero
    merged["pagerank"] = (
        merged["pagerank"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Normalize PageRank
    # --------------------------------------------------------

    merged[
        "pagerank_normalized"
    ] = normalize_scores(
        merged["pagerank"]
    )

    # --------------------------------------------------------
    # Calculate hybrid score
    # --------------------------------------------------------

    merged[
        "hybrid_score"
    ] = (
        tfidf_weight
        * merged["score"]
        +
        pagerank_weight
        * merged["pagerank_normalized"]
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    merged = merged.sort_values(
        by="hybrid_score",
        ascending=False
    )

    merged = merged.reset_index(
        drop=True
    )

    merged["hybrid_rank"] = (
        merged.index + 1
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return merged


# ============================================================
# DISPLAY
# ============================================================

def display_comparison(
    results
):

    print("\n")
    print("=" * 100)

    print(
        "TF-IDF vs PAGERANK vs HYBRID RANKING"
    )

    print("=" * 100)

    columns = [
        "hybrid_rank",
        "doc_id",
        "title",
        "score",
        "pagerank",
        "pagerank_normalized",
        "hybrid_score"
    ]

    display_df = results[
        columns
    ].copy()

    display_df = display_df.head(
        10
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print("=" * 100)


# ============================================================
# PROCESS ONE SEARCH RESULT
# ============================================================

def process_search_result(
    search_file,
    pagerank_df
):

    print(
        f"\nProcessing: "
        f"{search_file}"
    )

    search_results = pd.read_csv(
        search_file
    )

    if search_results.empty:

        print(
            "No search results."
        )

        return

    hybrid_results = (
        apply_hybrid_ranking(
            search_results,
            pagerank_df,
            TFIDF_WEIGHT,
            PAGERANK_WEIGHT
        )
    )

    display_comparison(
        hybrid_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = os.path.basename(
        search_file
    )

    output_filename = (
        filename.replace(
            "search_",
            "hybrid_"
        )
    )

    output_path = os.path.join(
        RESULTS_DIR,
        output_filename
    )

    hybrid_results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("HYBRID RANKING")
    print("=" * 100)

    pagerank_df = (
        load_pagerank()
    )

    search_files = (
        load_search_results()
    )

    if not search_files:

        print(
            "\nNo TF-IDF search result files found."
        )

        print(
            "Run search_engine.py first."
        )

        return

    print(
        f"\nTF-IDF result files found: "
        f"{len(search_files)}"
    )

    print(
        f"TF-IDF weight: "
        f"{TFIDF_WEIGHT}"
    )

    print(
        f"PageRank weight: "
        f"{PAGERANK_WEIGHT}"
    )

    for search_file in search_files:

        process_search_result(
            search_file,
            pagerank_df
        )

    print("\n")
    print("=" * 100)
    print("HYBRID RANKING COMPLETE")
    print("=" * 100)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()