import os
import json
import math
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/processed_documents.csv"
)

INDEX_DIR = "index"

INVERTED_INDEX_FILE = (
    "index/inverted_index.json"
)

DOCUMENT_STATS_FILE = (
    "index/document_stats.json"
)

INDEX_METADATA_FILE = (
    "index/index_metadata.json"
)

TERM_STATS_FILE = (
    "index/term_statistics.csv"
)


# ============================================================
# CREATE INDEX DIRECTORY
# ============================================================

def create_index_directory():
    """
    Create directory for storing index files.
    """

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )


# ============================================================
# LOAD PROCESSED DOCUMENTS
# ============================================================

def load_documents():
    """
    Load the processed document collection.
    """

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise ValueError(
            "The processed document collection is empty."
        )

    return df


# ============================================================
# BUILD INVERTED INDEX
# ============================================================

def build_inverted_index(df):
    """
    Build an inverted index.

    Structure:

    {
        "term": {
            "doc_id": term_frequency,
            ...
        }
    }
    """

    inverted_index = defaultdict(dict)

    document_stats = {}

    # --------------------------------------------------------
    # Process every document
    # --------------------------------------------------------

    for _, row in df.iterrows():

        doc_id = str(
            row["doc_id"]
        )

        processed_content = str(
            row["processed_content"]
        )

        # Split processed text into tokens
        tokens = processed_content.split()

        # Term frequency for current document
        term_frequency = Counter(
            tokens
        )

        # ----------------------------------------------------
        # Store postings
        # ----------------------------------------------------

        for term, frequency in (
            term_frequency.items()
        ):

            inverted_index[
                term
            ][doc_id] = int(frequency)

        # ----------------------------------------------------
        # Document statistics
        # ----------------------------------------------------

        document_stats[doc_id] = {

            "title": str(
                row["title"]
            ),

            "category": str(
                row["category"]
            ),

            "url": str(
                row["url"]
            ),

            "word_count": int(
                len(tokens)
            ),

            "unique_terms": int(
                len(term_frequency)
            ),

            "max_term_frequency": int(
                max(
                    term_frequency.values()
                )
                if term_frequency
                else 0
            )
        }

    return (
        dict(inverted_index),
        document_stats
    )


# ============================================================
# CALCULATE TERM STATISTICS
# ============================================================

def calculate_term_statistics(
    inverted_index,
    total_documents
):
    """
    Calculate:

    TF
    DF
    IDF

    for every term.
    """

    rows = []

    for term, postings in (
        inverted_index.items()
    ):

        document_frequency = len(
            postings
        )

        total_frequency = sum(
            postings.values()
        )

        # Smooth IDF
        idf = math.log(
            (
                total_documents + 1
            )
            /
            (
                document_frequency + 1
            )
        ) + 1

        rows.append(
            {
                "term": term,

                "document_frequency":
                    document_frequency,

                "total_frequency":
                    total_frequency,

                "idf":
                    round(
                        idf,
                        6
                    )
            }
        )

    return rows


# ============================================================
# SAVE INVERTED INDEX
# ============================================================

def save_inverted_index(
    inverted_index
):
    """
    Save inverted index as JSON.
    """

    with open(
        INVERTED_INDEX_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inverted_index,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# SAVE DOCUMENT STATISTICS
# ============================================================

def save_document_stats(
    document_stats
):
    """
    Save document statistics as JSON.
    """

    with open(
        DOCUMENT_STATS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            document_stats,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# SAVE INDEX METADATA
# ============================================================

def save_index_metadata(
    df,
    inverted_index
):
    """
    Save overall information about the index.
    """

    total_documents = len(
        df
    )

    total_terms = len(
        inverted_index
    )

    total_postings = sum(
        len(postings)
        for postings
        in inverted_index.values()
    )

    metadata = {

        "total_documents":
            total_documents,

        "unique_terms":
            total_terms,

        "total_postings":
            total_postings,

        "average_postings_per_term":
            round(
                total_postings / total_terms,
                4
            )
            if total_terms > 0
            else 0
    }

    with open(
        INDEX_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


# ============================================================
# SAVE TERM STATISTICS
# ============================================================

def save_term_statistics(
    term_statistics
):
    """
    Save term-level statistics to CSV.
    """

    df = pd.DataFrame(
        term_statistics
    )

    df = df.sort_values(
        by="document_frequency",
        ascending=False
    )

    df.to_csv(
        TERM_STATS_FILE,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# DISPLAY SAMPLE INDEX
# ============================================================

def display_sample_index(
    inverted_index,
    number_of_terms=20
):
    """
    Display a sample of the inverted index.
    """

    print("\n")
    print("=" * 70)
    print("SAMPLE INVERTED INDEX")
    print("=" * 70)

    terms = sorted(
        inverted_index.keys()
    )[
        :number_of_terms
    ]

    for term in terms:

        postings = (
            inverted_index[term]
        )

        print(
            f"\n{term}"
        )

        for doc_id, frequency in (
            list(postings.items())[:10]
        ):

            print(
                f"    {doc_id} "
                f"(TF={frequency})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "INVERTED INDEX BUILDER"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    create_index_directory()

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    print(
        "\nLoading processed documents..."
    )

    df = load_documents()

    print(
        f"Documents loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Build inverted index
    # --------------------------------------------------------

    print(
        "\nBuilding inverted index..."
    )

    (
        inverted_index,
        document_stats
    ) = build_inverted_index(
        df
    )

    # --------------------------------------------------------
    # Calculate term statistics
    # --------------------------------------------------------

    print(
        "Calculating term statistics..."
    )

    term_statistics = (
        calculate_term_statistics(
            inverted_index,
            len(df)
        )
    )

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    print(
        "Saving inverted index..."
    )

    save_inverted_index(
        inverted_index
    )

    save_document_stats(
        document_stats
    )

    save_index_metadata(
        df,
        inverted_index
    )

    save_term_statistics(
        term_statistics
    )

    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    display_sample_index(
        inverted_index
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_documents = len(
        df
    )

    total_terms = len(
        inverted_index
    )

    total_postings = sum(
        len(postings)
        for postings
        in inverted_index.values()
    )

    print("\n")
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Documents indexed       : "
        f"{total_documents}"
    )

    print(
        f"Unique terms            : "
        f"{total_terms}"
    )

    print(
        f"Total postings          : "
        f"{total_postings}"
    )

    print(
        f"Average postings/term   : "
        f"{total_postings / total_terms:.2f}"
        if total_terms > 0
        else "Average postings/term   : 0"
    )

    print("\nIndex files:")

    print(
        f"1. {INVERTED_INDEX_FILE}"
    )

    print(
        f"2. {DOCUMENT_STATS_FILE}"
    )

    print(
        f"3. {INDEX_METADATA_FILE}"
    )

    print(
        f"4. {TERM_STATS_FILE}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()