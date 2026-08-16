import os
import re
import pandas as pd
import numpy as np

from collections import Counter

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer
)

from nltk.stem import PorterStemmer


# ============================================================
# CONFIGURATION
# ============================================================

CONTENT_FILE = "data/processed/content.csv"
METADATA_FILE = "data/processed/documents.csv"

OUTPUT_PROCESSED = (
    "data/processed/processed_documents.csv"
)

OUTPUT_KEYWORDS = (
    "data/processed/keywords.csv"
)

OUTPUT_STATISTICS = (
    "data/processed/corpus_statistics.csv"
)


# ============================================================
# INITIALIZE STEMMER
# ============================================================

stemmer = PorterStemmer()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean and normalize document text.

    Processing steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove non-alphabetic characters
    4. Tokenize
    5. Remove stopwords
    6. Remove very short words
    7. Apply stemming
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # Convert to lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Keep alphabetic characters only
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = text.split()

    # --------------------------------------------------------
    # Stopword removal
    # --------------------------------------------------------

    tokens = [
        token
        for token in tokens
        if token not in ENGLISH_STOP_WORDS
    ]

    # --------------------------------------------------------
    # Remove very short tokens
    # --------------------------------------------------------

    tokens = [
        token
        for token in tokens
        if len(token) >= 3
    ]

    # --------------------------------------------------------
    # Stemming
    # --------------------------------------------------------

    stemmed_tokens = [
        stemmer.stem(token)
        for token in tokens
    ]

    return " ".join(stemmed_tokens)


# ============================================================
# TOKENIZE CLEAN TEXT
# ============================================================

def get_tokens(clean_text):
    """
    Convert cleaned text into a list of tokens.
    """

    if not clean_text:
        return []

    return clean_text.split()


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

def calculate_document_statistics(
    doc_id,
    original_text,
    clean_text
):
    """
    Calculate document-level statistics.
    """

    original_tokens = str(
        original_text
    ).split()

    clean_tokens = get_tokens(
        clean_text
    )

    unique_terms = set(
        clean_tokens
    )

    return {

        "doc_id": doc_id,

        "original_word_count":
            len(original_tokens),

        "processed_word_count":
            len(clean_tokens),

        "unique_terms":
            len(unique_terms),

        "average_word_length":
            (
                np.mean(
                    [
                        len(word)
                        for word in clean_tokens
                    ]
                )
                if clean_tokens
                else 0
            )
    }


# ============================================================
# EXTRACT TOP KEYWORDS USING TF-IDF
# ============================================================

def extract_keywords(
    doc_ids,
    processed_texts,
    top_n=10
):
    """
    Extract top TF-IDF keywords for each document.
    """

    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(
        processed_texts
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    keyword_rows = []

    for row_index, doc_id in enumerate(doc_ids):

        row = tfidf_matrix[
            row_index
        ].toarray().flatten()

        top_indices = np.argsort(
            row
        )[::-1][:top_n]

        rank = 1

        for index in top_indices:

            score = row[index]

            if score <= 0:
                continue

            keyword_rows.append(
                {
                    "doc_id": doc_id,
                    "rank": rank,
                    "keyword": feature_names[index],
                    "tfidf_score": round(
                        float(score),
                        6
                    )
                }
            )

            rank += 1

    return keyword_rows


# ============================================================
# CORPUS-WIDE TERM FREQUENCY
# ============================================================

def calculate_term_frequency(
    processed_texts
):
    """
    Calculate corpus-wide term frequency.
    """

    counter = Counter()

    for text in processed_texts:

        tokens = get_tokens(
            text
        )

        counter.update(
            tokens
        )

    return counter


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print("=" * 70)
    print("TEXT PREPROCESSING AND FEATURE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    content_df = pd.read_csv(
        CONTENT_FILE
    )

    metadata_df = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Documents loaded: "
        f"{len(content_df)}"
    )

    # --------------------------------------------------------
    # Merge metadata and content
    # --------------------------------------------------------

    df = pd.merge(
        metadata_df,
        content_df,
        on="doc_id",
        how="inner"
    )

    print(
        f"Documents after merge: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Preprocess documents
    # --------------------------------------------------------

    print("\nPreprocessing documents...")

    df["processed_content"] = (
        df["content"]
        .fillna("")
        .apply(preprocess_text)
    )

    # --------------------------------------------------------
    # Document statistics
    # --------------------------------------------------------

    print(
        "\nCalculating document statistics..."
    )

    statistics = []

    for _, row in df.iterrows():

        stats = calculate_document_statistics(
            row["doc_id"],
            row["content"],
            row["processed_content"]
        )

        statistics.append(
            stats
        )

    statistics_df = pd.DataFrame(
        statistics
    )

    # --------------------------------------------------------
    # Merge statistics
    # --------------------------------------------------------

    df = pd.merge(
        df,
        statistics_df,
        on="doc_id",
        how="left"
    )

    # --------------------------------------------------------
    # Extract keywords
    # --------------------------------------------------------

    print(
        "\nExtracting TF-IDF keywords..."
    )

    processed_texts = (
        df["processed_content"]
        .tolist()
    )

    doc_ids = (
        df["doc_id"]
        .tolist()
    )

    # Check whether corpus contains text
    valid_texts = [
        text
        for text in processed_texts
        if text.strip()
    ]

    if not valid_texts:

        print(
            "ERROR: No usable text found."
        )

        return

    keyword_rows = extract_keywords(
        doc_ids,
        processed_texts,
        top_n=10
    )

    keywords_df = pd.DataFrame(
        keyword_rows
    )

    # --------------------------------------------------------
    # Corpus term frequencies
    # --------------------------------------------------------

    print(
        "\nCalculating corpus term frequencies..."
    )

    term_counter = calculate_term_frequency(
        processed_texts
    )

    top_terms = (
        term_counter
        .most_common(30)
    )

    # --------------------------------------------------------
    # Corpus statistics
    # --------------------------------------------------------

    total_documents = len(df)

    total_words = (
        statistics_df[
            "processed_word_count"
        ].sum()
    )

    total_unique_terms = len(
        term_counter
    )

    average_document_length = (
        statistics_df[
            "processed_word_count"
        ].mean()
    )

    median_document_length = (
        statistics_df[
            "processed_word_count"
        ].median()
    )

    min_document_length = (
        statistics_df[
            "processed_word_count"
        ].min()
    )

    max_document_length = (
        statistics_df[
            "processed_word_count"
        ].max()
    )

    corpus_statistics = pd.DataFrame(
        [
            {
                "metric": "total_documents",
                "value": total_documents
            },
            {
                "metric": "total_processed_words",
                "value": int(total_words)
            },
            {
                "metric": "unique_terms",
                "value": total_unique_terms
            },
            {
                "metric": "average_document_length",
                "value": round(
                    float(
                        average_document_length
                    ),
                    2
                )
            },
            {
                "metric": "median_document_length",
                "value": round(
                    float(
                        median_document_length
                    ),
                    2
                )
            },
            {
                "metric": "minimum_document_length",
                "value": int(
                    min_document_length
                )
            },
            {
                "metric": "maximum_document_length",
                "value": int(
                    max_document_length
                )
            }
        ]
    )

    # --------------------------------------------------------
    # Save processed documents
    # --------------------------------------------------------

    processed_columns = [
        "doc_id",
        "title",
        "category",
        "url",
        "source",
        "crawl_depth",
        "word_count",
        "content_hash",
        "content",
        "processed_content",
        "original_word_count",
        "processed_word_count",
        "unique_terms",
        "average_word_length"
    ]

    processed_df = df[
        processed_columns
    ]

    processed_df.to_csv(
        OUTPUT_PROCESSED,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save keywords
    # --------------------------------------------------------

    keywords_df.to_csv(
        OUTPUT_KEYWORDS,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save corpus statistics
    # --------------------------------------------------------

    corpus_statistics.to_csv(
        OUTPUT_STATISTICS,
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Documents              : "
        f"{total_documents}"
    )

    print(
        f"Processed words        : "
        f"{int(total_words)}"
    )

    print(
        f"Unique terms           : "
        f"{total_unique_terms}"
    )

    print(
        f"Average document size  : "
        f"{average_document_length:.2f}"
    )

    print(
        f"Minimum document size  : "
        f"{min_document_length}"
    )

    print(
        f"Maximum document size  : "
        f"{max_document_length}"
    )

    print("\nTop 30 corpus terms:")

    for rank, (term, count) in enumerate(
        top_terms,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{term:<20} "
            f"{count}"
        )

    print("\nOutput files:")

    print(
        f"Processed documents: "
        f"{OUTPUT_PROCESSED}"
    )

    print(
        f"Keywords: "
        f"{OUTPUT_KEYWORDS}"
    )

    print(
        f"Statistics: "
        f"{OUTPUT_STATISTICS}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()