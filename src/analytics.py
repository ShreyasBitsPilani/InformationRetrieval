import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_FILE = (
    "data/processed/processed_documents.csv"
)

CORPUS_STATS_FILE = (
    "data/processed/corpus_statistics.csv"
)

PAGERANK_FILE = (
    "results/pagerank_scores.csv"
)

EVALUATION_FILE = (
    "results/evaluation_summary.csv"
)

OUTPUT_DIR = (
    "results/visualizations"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def create_output_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_processed_documents():

    return pd.read_csv(
        PROCESSED_FILE
    )


def load_pagerank():

    return pd.read_csv(
        PAGERANK_FILE
    )


def load_evaluation():

    return pd.read_csv(
        EVALUATION_FILE
    )


# ============================================================
# 1. DOCUMENT LENGTH DISTRIBUTION
# ============================================================

def plot_document_length(
    df
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        df[
            "processed_word_count"
        ],
        bins=20
    )

    plt.xlabel(
        "Document Length (Words)"
    )

    plt.ylabel(
        "Number of Documents"
    )

    plt.title(
        "Document Length Distribution"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "document_length_distribution.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {output}"
    )


# ============================================================
# 2. CATEGORY DISTRIBUTION
# ============================================================

def plot_category_distribution(
    df
):

    category_counts = (
        df[
            "category"
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    category_counts.plot(
        kind="bar"
    )

    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Number of Documents"
    )

    plt.title(
        "Document Distribution by Category"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "category_distribution.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {output}"
    )


# ============================================================
# 3. TOP TERMS
# ============================================================

def plot_top_terms():

    term_file = (
        "index/term_statistics.csv"
    )

    if not os.path.exists(
        term_file
    ):

        print(
            "term_statistics.csv not found."
        )

        return

    terms = pd.read_csv(
        term_file
    )

    top_terms = (
        terms
        .sort_values(
            "total_frequency",
            ascending=False
        )
        .head(20)
        .sort_values(
            "total_frequency"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_terms["term"],
        top_terms[
            "total_frequency"
        ]
    )

    plt.xlabel(
        "Term Frequency"
    )

    plt.ylabel(
        "Term"
    )

    plt.title(
        "Top 20 Corpus Terms"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "top_terms.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {output}"
    )


# ============================================================
# 4. PAGERANK DISTRIBUTION
# ============================================================

def plot_pagerank_distribution(
    pagerank
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        pagerank[
            "pagerank"
        ],
        bins=20
    )

    plt.xlabel(
        "PageRank Score"
    )

    plt.ylabel(
        "Number of Documents"
    )

    plt.title(
        "PageRank Score Distribution"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "pagerank_distribution.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {output}"
    )


# ============================================================
# 5. TOP PAGERANK DOCUMENTS
# ============================================================

def plot_top_pagerank(
    pagerank
):

    top_docs = (
        pagerank
        .sort_values(
            "pagerank",
            ascending=True
        )
        .tail(10)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_docs["title"].astype(str),
        top_docs["pagerank"]
    )

    plt.xlabel(
        "PageRank Score"
    )

    plt.ylabel(
        "Document"
    )

    plt.title(
        "Top 10 Documents by PageRank"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "top_pagerank_documents.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {output}"
    )


# ============================================================
# 6. EVALUATION COMPARISON
# ============================================================

def plot_evaluation_comparison(
    evaluation
):

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "precision_at_5",
        "recall_at_5",
        "map",
        "mrr",
        "ndcg_at_10"
    ]

    available_metrics = [
        metric
        for metric in metric_columns
        if metric in evaluation.columns
    ]

    comparison = (
        evaluation[
            [
                "method"
            ]
            +
            available_metrics
        ]
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    comparison = comparison.set_index(
        "method"
    )

    ax = comparison[
        available_metrics
    ].T.plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_xlabel(
        "Evaluation Metric"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "TF-IDF vs Hybrid Ranking Performance"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend(
        title="Ranking Method"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "tfidf_vs_hybrid.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Created: {output}"
    )


# ============================================================
# 7. SAVE ANALYTICS TABLE
# ============================================================

def create_corpus_summary(
    df,
    pagerank
):

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "Total Documents",

                "value":
                    len(df)
            },

            {
                "metric":
                    "Average Document Length",

                "value":
                    round(
                        df[
                            "processed_word_count"
                        ].mean(),
                        2
                    )
            },

            {
                "metric":
                    "Median Document Length",

                "value":
                    round(
                        df[
                            "processed_word_count"
                        ].median(),
                        2
                    )
            },

            {
                "metric":
                    "Minimum Document Length",

                "value":
                    int(
                        df[
                            "processed_word_count"
                        ].min()
                    )
            },

            {
                "metric":
                    "Maximum Document Length",

                "value":
                    int(
                        df[
                            "processed_word_count"
                        ].max()
                    )
            },

            {
                "metric":
                    "Unique Categories",

                "value":
                    df[
                        "category"
                    ].nunique()
            },

            {
                "metric":
                    "Average PageRank",

                "value":
                    round(
                        pagerank[
                            "pagerank"
                        ].mean(),
                        6
                    )
            }
        ]
    )

    output = os.path.join(
        OUTPUT_DIR,
        "corpus_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(
        f"Created: {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("IR SYSTEM PERFORMANCE ANALYTICS")
    print("=" * 80)

    create_output_directory()

    print(
        "\nLoading processed documents..."
    )

    df = load_processed_documents()

    print(
        f"Documents: {len(df)}"
    )

    print(
        "\nLoading PageRank..."
    )

    pagerank = load_pagerank()

    print(
        f"PageRank records: "
        f"{len(pagerank)}"
    )

    print(
        "\nLoading evaluation..."
    )

    evaluation = load_evaluation()

    print(
        f"Evaluation methods: "
        f"{len(evaluation)}"
    )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    print(
        "\nGenerating visualizations..."
    )

    plot_document_length(
        df
    )

    plot_category_distribution(
        df
    )

    plot_top_terms()

    plot_pagerank_distribution(
        pagerank
    )

    plot_top_pagerank(
        pagerank
    )

    plot_evaluation_comparison(
        evaluation
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    create_corpus_summary(
        df,
        pagerank
    )

    print("\n")
    print("=" * 80)
    print("ANALYTICS COMPLETE")
    print("=" * 80)

    print(
        f"\nVisualization directory:"
        f"\n{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()