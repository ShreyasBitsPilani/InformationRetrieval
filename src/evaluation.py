import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

QUERY_FILE = (
    "data/processed/evaluation_queries.csv"
)

JUDGMENT_FILE = (
    "data/processed/relevance_judgments.csv"
)

RESULTS_DIR = "results"

OUTPUT_PER_QUERY = (
    "results/evaluation_per_query.csv"
)

OUTPUT_SUMMARY = (
    "results/evaluation_summary.csv"
)


# ============================================================
# BASIC METRICS
# ============================================================

def precision(
    retrieved,
    relevant
):
    """
    Precision = relevant retrieved / retrieved
    """

    if len(retrieved) == 0:
        return 0.0

    relevant_retrieved = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant
    )

    return (
        relevant_retrieved
        /
        len(retrieved)
    )


def recall(
    retrieved,
    relevant
):
    """
    Recall = relevant retrieved / all relevant
    """

    if len(relevant) == 0:
        return 0.0

    relevant_retrieved = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant
    )

    return (
        relevant_retrieved
        /
        len(relevant)
    )


def f1_score(
    precision_value,
    recall_value
):
    """
    Harmonic mean of precision and recall.
    """

    if (
        precision_value
        +
        recall_value
        == 0
    ):
        return 0.0

    return (
        2
        *
        precision_value
        *
        recall_value
        /
        (
            precision_value
            +
            recall_value
        )
    )


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k
):
    """
    Precision among the first K results.
    """

    top_k = retrieved[:k]

    return precision(
        top_k,
        relevant
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    retrieved,
    relevant,
    k
):
    """
    Recall among the first K results.
    """

    top_k = retrieved[:k]

    return recall(
        top_k,
        relevant
    )


# ============================================================
# AVERAGE PRECISION
# ============================================================

def average_precision(
    retrieved,
    relevant
):
    """
    Average Precision for one query.
    """

    if len(relevant) == 0:
        return 0.0

    score = 0.0

    relevant_found = 0

    for rank, doc_id in enumerate(
        retrieved,
        start=1
    ):

        if doc_id in relevant:

            relevant_found += 1

            score += (
                relevant_found
                /
                rank
            )

    return (
        score
        /
        len(relevant)
    )


# ============================================================
# MEAN AVERAGE PRECISION
# ============================================================

def mean_average_precision(
    query_results
):
    """
    MAP across all queries.
    """

    if not query_results:
        return 0.0

    scores = []

    for result in query_results:

        scores.append(
            average_precision(
                result["retrieved"],
                result["relevant"]
            )
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# MEAN RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
    retrieved,
    relevant
):
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, doc_id in enumerate(
        retrieved,
        start=1
    ):

        if doc_id in relevant:

            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    query_results
):
    """
    MRR across all queries.
    """

    if not query_results:
        return 0.0

    scores = []

    for result in query_results:

        scores.append(
            reciprocal_rank(
                result["retrieved"],
                result["relevant"]
            )
        )

    return float(
        np.mean(scores)
    )


# ============================================================
# NDCG
# ============================================================

def dcg_at_k(
    retrieved,
    relevant,
    k
):
    """
    Discounted Cumulative Gain.
    """

    dcg = 0.0

    for rank, doc_id in enumerate(
        retrieved[:k],
        start=1
    ):

        relevance = (
            1
            if doc_id in relevant
            else 0
        )

        dcg += (
            relevance
            /
            np.log2(rank + 1)
        )

    return dcg


def ndcg_at_k(
    retrieved,
    relevant,
    k
):
    """
    Normalized Discounted Cumulative Gain.
    """

    actual_dcg = dcg_at_k(
        retrieved,
        relevant,
        k
    )

    ideal_retrieved = (
        [None]
        *
        min(
            len(relevant),
            k
        )
    )

    ideal_dcg = dcg_at_k(
        ideal_retrieved,
        set(ideal_retrieved),
        k
    )

    # The generic ideal calculation above gives zero,
    # so calculate ideal DCG directly.

    ideal_dcg = sum(
        1
        /
        np.log2(rank + 1)
        for rank in range(
            1,
            min(
                len(relevant),
                k
            )
            + 1
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return (
        actual_dcg
        /
        ideal_dcg
    )


# ============================================================
# LOAD SEARCH RESULTS
# ============================================================

def load_search_result(
    query_id,
    query
):
    """
    Find the corresponding TF-IDF result file.
    """

    safe_query = (
        query
        .lower()
        .replace(" ", "_")
    )

    safe_query = "".join(
        character
        for character in safe_query
        if character.isalnum()
        or character == "_"
    )

    filename = (
        f"search_{safe_query}.csv"
    )

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(path):

        return None

    return pd.read_csv(
        path
    )


# ============================================================
# LOAD HYBRID RESULTS
# ============================================================

def load_hybrid_result(
    query
):
    """
    Find corresponding hybrid result.
    """

    safe_query = (
        query
        .lower()
        .replace(" ", "_")
    )

    safe_query = "".join(
        character
        for character in safe_query
        if character.isalnum()
        or character == "_"
    )

    filename = (
        f"hybrid_{safe_query}.csv"
    )

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(path):

        return None

    return pd.read_csv(
        path
    )


# ============================================================
# BUILD QUERY RESULT
# ============================================================

def evaluate_one_query(
    query_id,
    retrieved,
    relevant,
    k=10
):

    p = precision(
        retrieved,
        relevant
    )

    r = recall(
        retrieved,
        relevant
    )

    f1 = f1_score(
        p,
        r
    )

    p5 = precision_at_k(
        retrieved,
        relevant,
        5
    )

    r5 = recall_at_k(
        retrieved,
        relevant,
        5
    )

    p10 = precision_at_k(
        retrieved,
        relevant,
        10
    )

    r10 = recall_at_k(
        retrieved,
        relevant,
        10
    )

    ap = average_precision(
        retrieved,
        relevant
    )

    rr = reciprocal_rank(
        retrieved,
        relevant
    )

    ndcg = ndcg_at_k(
        retrieved,
        relevant,
        k
    )

    return {
        "query_id": query_id,
        "precision": p,
        "recall": r,
        "f1": f1,
        "precision_at_5": p5,
        "recall_at_5": r5,
        "precision_at_10": p10,
        "recall_at_10": r10,
        "average_precision": ap,
        "reciprocal_rank": rr,
        "ndcg_at_10": ndcg
    }


# ============================================================
# EVALUATE METHOD
# ============================================================

def evaluate_method(
    queries,
    judgments,
    method
):

    per_query = []

    query_result_objects = []

    for _, query_row in queries.iterrows():

        query_id = query_row[
            "query_id"
        ]

        query = query_row[
            "query"
        ]

        # ----------------------------------------------------
        # Get relevance judgments
        # ----------------------------------------------------

        relevant_docs = set(
            judgments[
                (
                    judgments["query_id"]
                    == query_id
                )
                &
                (
                    judgments["relevant"]
                    == 1
                )
            ]["doc_id"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # Get retrieved documents
        # ----------------------------------------------------

        if method == "TF-IDF":

            result_df = (
                load_search_result(
                    query_id,
                    query
                )
            )

            if result_df is None:
                continue

            retrieved = (
                result_df[
                    "doc_id"
                ]
                .astype(str)
                .tolist()
            )

        elif method == "Hybrid":

            result_df = (
                load_hybrid_result(
                    query
                )
            )

            if result_df is None:
                continue

            result_df = (
                result_df.sort_values(
                    "hybrid_score",
                    ascending=False
                )
            )

            retrieved = (
                result_df[
                    "doc_id"
                ]
                .astype(str)
                .tolist()
            )

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = evaluate_one_query(
            query_id,
            retrieved,
            relevant_docs,
            k=10
        )

        metrics["method"] = method

        per_query.append(
            metrics
        )

        query_result_objects.append(
            {
                "retrieved": retrieved,
                "relevant": relevant_docs
            }
        )

    # --------------------------------------------------------
    # MAP and MRR
    # --------------------------------------------------------

    map_score = (
        mean_average_precision(
            query_result_objects
        )
    )

    mrr_score = (
        mean_reciprocal_rank(
            query_result_objects
        )
    )

    return (
        per_query,
        map_score,
        mrr_score
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INFORMATION RETRIEVAL EVALUATION")
    print("=" * 80)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    queries = pd.read_csv(
        QUERY_FILE
    )

    judgments = pd.read_csv(
        JUDGMENT_FILE
    )

    print(
        f"\nQueries: {len(queries)}"
    )

    print(
        f"Judgments: {len(judgments)}"
    )

    # --------------------------------------------------------
    # Evaluate TF-IDF
    # --------------------------------------------------------

    print("\nEvaluating TF-IDF...")

    (
        tfidf_results,
        tfidf_map,
        tfidf_mrr
    ) = evaluate_method(
        queries,
        judgments,
        "TF-IDF"
    )

    # --------------------------------------------------------
    # Evaluate Hybrid
    # --------------------------------------------------------

    print("\nEvaluating Hybrid ranking...")

    (
        hybrid_results,
        hybrid_map,
        hybrid_mrr
    ) = evaluate_method(
        queries,
        judgments,
        "Hybrid"
    )

    # --------------------------------------------------------
    # Combine per-query results
    # --------------------------------------------------------

    all_results = (
        tfidf_results
        +
        hybrid_results
    )

    per_query_df = pd.DataFrame(
        all_results
    )

    per_query_df.to_csv(
        OUTPUT_PER_QUERY,
        index=False
    )

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary_rows = []

    for method, map_score, mrr_score in [
        (
            "TF-IDF",
            tfidf_map,
            tfidf_mrr
        ),
        (
            "Hybrid",
            hybrid_map,
            hybrid_mrr
        )
    ]:

        method_df = (
            per_query_df[
                per_query_df["method"]
                == method
            ]
        )

        summary_rows.append(
            {
                "method": method,

                "precision":
                    method_df[
                        "precision"
                    ].mean(),

                "recall":
                    method_df[
                        "recall"
                    ].mean(),

                "f1":
                    method_df[
                        "f1"
                    ].mean(),

                "precision_at_5":
                    method_df[
                        "precision_at_5"
                    ].mean(),

                "recall_at_5":
                    method_df[
                        "recall_at_5"
                    ].mean(),

                "precision_at_10":
                    method_df[
                        "precision_at_10"
                    ].mean(),

                "recall_at_10":
                    method_df[
                        "recall_at_10"
                    ].mean(),

                "map":
                    map_score,

                "mrr":
                    mrr_score,

                "ndcg_at_10":
                    method_df[
                        "ndcg_at_10"
                    ].mean()
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # Round for readability
    metric_columns = [
        "precision",
        "recall",
        "f1",
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10",
        "map",
        "mrr",
        "ndcg_at_10"
    ]

    summary_df[
        metric_columns
    ] = summary_df[
        metric_columns
    ].round(4)

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("=" * 100)

    print(
        f"\nPer-query results: "
        f"{OUTPUT_PER_QUERY}"
    )

    print(
        f"Summary: "
        f"{OUTPUT_SUMMARY}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()