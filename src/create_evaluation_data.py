import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FILE = (
    "data/processed/documents.csv"
)

QUERY_FILE = (
    "data/processed/evaluation_queries.csv"
)

JUDGMENT_FILE = (
    "data/processed/relevance_judgments.csv"
)


# ============================================================
# CREATE EVALUATION QUERIES
# ============================================================

def create_queries():

    queries = [
        {
            "query_id": "Q001",
            "query": (
                "information retrieval "
                "search ranking"
            ),
            "category": "Information Retrieval"
        },

        {
            "query_id": "Q002",
            "query": (
                "machine learning "
                "classification"
            ),
            "category": "Machine Learning"
        },

        {
            "query_id": "Q003",
            "query": (
                "artificial intelligence "
                "systems"
            ),
            "category": "Artificial Intelligence"
        },

        {
            "query_id": "Q004",
            "query": (
                "natural language "
                "processing text"
            ),
            "category": "Natural Language Processing"
        },

        {
            "query_id": "Q005",
            "query": (
                "data mining algorithms"
            ),
            "category": "Data Mining"
        }
    ]

    return pd.DataFrame(
        queries
    )


# ============================================================
# CREATE RELEVANCE JUDGMENTS
# ============================================================

def create_judgments(
    documents,
    queries
):

    rows = []

    for _, query_row in queries.iterrows():

        query_id = query_row[
            "query_id"
        ]

        query_category = query_row[
            "category"
        ]

        for _, document_row in (
            documents.iterrows()
        ):

            doc_id = str(
                document_row["doc_id"]
            )

            document_category = str(
                document_row["category"]
            )

            # ------------------------------------------------
            # Category-based relevance
            # ------------------------------------------------

            relevant = int(
                document_category
                .strip()
                .lower()
                ==
                query_category
                .strip()
                .lower()
            )

            rows.append(
                {
                    "query_id": query_id,

                    "doc_id": doc_id,

                    "relevant": relevant
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING EVALUATION DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    if not os.path.exists(
        DOCUMENT_FILE
    ):

        raise FileNotFoundError(
            f"Missing {DOCUMENT_FILE}. "
            "Run dataset_builder.py first."
        )

    documents = pd.read_csv(
        DOCUMENT_FILE
    )

    print(
        f"\nDocuments: {len(documents)}"
    )

    # --------------------------------------------------------
    # Create queries
    # --------------------------------------------------------

    queries = create_queries()

    # --------------------------------------------------------
    # Create relevance judgments
    # --------------------------------------------------------

    judgments = create_judgments(
        documents,
        queries
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    queries.to_csv(
        QUERY_FILE,
        index=False,
        encoding="utf-8"
    )

    judgments.to_csv(
        JUDGMENT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nEvaluation queries: "
        f"{len(queries)}"
    )

    print(
        f"Relevance judgments: "
        f"{len(judgments)}"
    )

    print("\nRelevant documents per query:")

    summary = (
        judgments[
            judgments["relevant"] == 1
        ]
        .groupby("query_id")
        .size()
    )

    print(summary)

    print("\nFiles created:")

    print(
        f"1. {QUERY_FILE}"
    )

    print(
        f"2. {JUDGMENT_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()