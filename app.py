import os
import sys
import json
import pickle
import time
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# OPTIONAL PREPROCESSING IMPORT
# ============================================================

try:
    from preprocessing import preprocess_text
except Exception:
    preprocess_text = None


# ============================================================
# FILE PATHS
# ============================================================

PROCESSED_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "processed_documents.csv"
)

CONTENT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "content.csv"
)

KEYWORDS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "keywords.csv"
)

CORPUS_STATS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "corpus_statistics.csv"
)

DOCUMENTS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "documents.csv"
)

LINKS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "links.csv"
)

DOC_LINKS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "doc_links.csv"
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

INVERTED_INDEX_FILE = os.path.join(
    PROJECT_ROOT,
    "index",
    "inverted_index.json"
)

INDEX_METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "index",
    "index_metadata.json"
)

TFIDF_VECTORIZER_FILE = os.path.join(
    PROJECT_ROOT,
    "index",
    "tfidf_vectorizer.pkl"
)

TFIDF_MATRIX_FILE = os.path.join(
    PROJECT_ROOT,
    "index",
    "tfidf_matrix.pkl"
)

PAGERANK_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "pagerank_scores.csv"
)

GRAPH_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "document_graph.json"
)

EVALUATION_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "evaluation_summary.csv"
)

PER_QUERY_EVALUATION_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "evaluation_per_query.csv"
)

VISUALIZATION_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "visualizations"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def exists(path):
    return os.path.exists(path)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def normalize_text(text):
    """
    Prepare query text for TF-IDF.
    Uses the project's preprocessing function when available.
    """

    text = str(text)

    if preprocess_text is not None:

        try:

            processed = preprocess_text(
                text
            )

            if isinstance(
                processed,
                list
            ):

                return " ".join(
                    str(x)
                    for x in processed
                )

            return str(
                processed
            )

        except Exception:
            pass

    return text.lower()


def get_document_text_column(df):
    """
    Identify the most appropriate text column.
    """

    candidates = [
        "processed_text",
        "content",
        "text",
        "body",
        "document"
    ]

    for column in candidates:

        if column in df.columns:

            return column

    return None


def get_title(row):
    return str(
        row.get(
            "title",
            "Untitled Document"
        )
    )


def get_doc_id(row):
    return str(
        row.get(
            "doc_id",
            ""
        )
    )


def get_category(row):
    return str(
        row.get(
            "category",
            "Unknown"
        )
    )


def get_url(row):
    return str(
        row.get(
            "url",
            ""
        )
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    if not exists(
        PROCESSED_FILE
    ):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            PROCESSED_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD KEYWORDS
# ============================================================

@st.cache_data
def load_keywords():

    if not exists(
        KEYWORDS_FILE
    ):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            KEYWORDS_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD CORPUS STATISTICS
# ============================================================

@st.cache_data
def load_corpus_statistics():

    if not exists(
        CORPUS_STATS_FILE
    ):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            CORPUS_STATS_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD INVERTED INDEX
# ============================================================

@st.cache_data
def load_inverted_index():

    if not exists(
        INVERTED_INDEX_FILE
    ):

        return {}

    try:

        with open(
            INVERTED_INDEX_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {}


# ============================================================
# LOAD INDEX METADATA
# ============================================================

@st.cache_data
def load_index_metadata():

    if not exists(
        INDEX_METADATA_FILE
    ):

        return {}

    try:

        with open(
            INDEX_METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {}


# ============================================================
# LOAD TF-IDF MODEL
# ============================================================

@st.cache_resource
def load_tfidf_model():

    if not exists(
        TFIDF_VECTORIZER_FILE
    ):

        return None, None

    if not exists(
        TFIDF_MATRIX_FILE
    ):

        return None, None

    try:

        with open(
            TFIDF_VECTORIZER_FILE,
            "rb"
        ) as file:

            vectorizer = pickle.load(
                file
            )

        with open(
            TFIDF_MATRIX_FILE,
            "rb"
        ) as file:

            matrix = pickle.load(
                file
            )

        return (
            vectorizer,
            matrix
        )

    except Exception:

        return None, None


# ============================================================
# LOAD PAGERANK
# ============================================================

@st.cache_data
def load_pagerank():

    if not exists(
        PAGERANK_FILE
    ):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            PAGERANK_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD GRAPH
# ============================================================

@st.cache_data
def load_graph():

    if not exists(
        GRAPH_FILE
    ):

        return {}

    try:

        with open(
            GRAPH_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {}


# ============================================================
# LOAD EVALUATION
# ============================================================

@st.cache_data
def load_evaluation():

    if not exists(
        EVALUATION_FILE
    ):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            EVALUATION_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD PER-QUERY EVALUATION
# ============================================================

@st.cache_data
def load_per_query_evaluation():

    if not exists(
        PER_QUERY_EVALUATION_FILE
    ):

        return pd.DataFrame()

    try:

        return pd.read_csv(
            PER_QUERY_EVALUATION_FILE
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# SEARCH
# ============================================================

def perform_tfidf_search(
    query,
    documents,
    vectorizer,
    tfidf_matrix,
    top_k
):

    if not query.strip():

        return pd.DataFrame()

    if vectorizer is None:

        return pd.DataFrame()

    if tfidf_matrix is None:

        return pd.DataFrame()

    processed_query = normalize_text(
        query
    )

    try:

        query_vector = (
            vectorizer.transform(
                [processed_query]
            )
        )

    except Exception:

        try:

            query_vector = (
                vectorizer.transform(
                    [query]
                )
            )

        except Exception:

            return pd.DataFrame()

    scores = (
        cosine_similarity(
            query_vector,
            tfidf_matrix
        )
        .flatten()
    )

    results = documents.copy()

    if len(scores) != len(results):

        return pd.DataFrame()

    results[
        "score"
    ] = scores

    results = results[
        results[
            "score"
        ] > 0
    ]

    results = (
        results
        .sort_values(
            "score",
            ascending=False
        )
        .head(
            top_k
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "rank"
    ] = (
        results.index + 1
    )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def perform_hybrid_search(
    query,
    documents,
    vectorizer,
    tfidf_matrix,
    pagerank,
    top_k,
    tfidf_weight=0.70,
    pagerank_weight=0.30
):

    results = perform_tfidf_search(
        query,
        documents,
        vectorizer,
        tfidf_matrix,
        len(documents)
    )

    if results.empty:

        return results

    if pagerank.empty:

        results[
            "pagerank"
        ] = 0.0

        results[
            "pagerank_normalized"
        ] = 0.0

        results[
            "hybrid_score"
        ] = results[
            "score"
        ]

        return (
            results
            .head(top_k)
            .reset_index(
                drop=True
            )
        )

    pagerank_temp = pagerank[
        [
            "doc_id",
            "pagerank"
        ]
    ].copy()

    results[
        "doc_id"
    ] = results[
        "doc_id"
    ].astype(str)

    pagerank_temp[
        "doc_id"
    ] = pagerank_temp[
        "doc_id"
    ].astype(str)

    results = pd.merge(
        results,
        pagerank_temp,
        on="doc_id",
        how="left"
    )

    results[
        "pagerank"
    ] = (
        results[
            "pagerank"
        ]
        .fillna(0.0)
    )

    pr_min = results[
        "pagerank"
    ].min()

    pr_max = results[
        "pagerank"
    ].max()

    if pr_max > pr_min:

        results[
            "pagerank_normalized"
        ] = (
            results[
                "pagerank"
            ] - pr_min
        ) / (
            pr_max - pr_min
        )

    else:

        results[
            "pagerank_normalized"
        ] = 0.0

    results[
        "hybrid_score"
    ] = (
        tfidf_weight
        *
        results[
            "score"
        ]
        +
        pagerank_weight
        *
        results[
            "pagerank_normalized"
        ]
    )

    results = (
        results
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        .head(
            top_k
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "rank"
    ] = (
        results.index + 1
    )

    return results


# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendations(
    doc_id,
    documents,
    tfidf_matrix,
    top_k
):

    if tfidf_matrix is None:

        return pd.DataFrame()

    document_ids = (
        documents[
            "doc_id"
        ]
        .astype(str)
        .tolist()
    )

    doc_id = str(
        doc_id
    )

    if doc_id not in document_ids:

        return pd.DataFrame()

    document_index = (
        document_ids.index(
            doc_id
        )
    )

    selected_vector = (
        tfidf_matrix[
            document_index
        ]
    )

    scores = (
        cosine_similarity(
            selected_vector,
            tfidf_matrix
        )
        .flatten()
    )

    recommendations = []

    for index, score in enumerate(
        scores
    ):

        if index == document_index:
            continue

        row = documents.iloc[
            index
        ]

        recommendations.append(
            {
                "doc_id":
                    row["doc_id"],

                "title":
                    row["title"],

                "category":
                    row.get(
                        "category",
                        "Unknown"
                    ),

                "url":
                    row.get(
                        "url",
                        ""
                    ),

                "similarity":
                    float(score)
            }
        )

    if not recommendations:

        return pd.DataFrame()

    result = (
        pd.DataFrame(
            recommendations
        )
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(
            top_k
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "rank"
    ] = (
        result.index + 1
    )

    return result


# ============================================================
# TEXT MINING / CLASSIFICATION
# ============================================================

def get_classification_dataframe(
    documents
):

    if documents.empty:

        return pd.DataFrame()

    text_column = (
        get_document_text_column(
            documents
        )
    )

    if text_column is None:

        return pd.DataFrame()

    if "category" not in documents.columns:

        return pd.DataFrame()

    df = documents[
        [
            text_column,
            "category"
        ]
    ].copy()

    df[
        text_column
    ] = (
        df[
            text_column
        ]
        .fillna("")
        .astype(str)
    )

    df[
        "category"
    ] = (
        df[
            "category"
        ]
        .fillna("")
        .astype(str)
    )

    df = df[
        (
            df[
                text_column
            ].str.strip() != ""
        )
        &
        (
            df[
                "category"
            ].str.strip() != ""
        )
    ]

    return df


def evaluate_feature_strategies(
    documents
):

    df = get_classification_dataframe(
        documents
    )

    if df.empty:

        return (
            pd.DataFrame(),
            {}
        )

    text_column = (
        get_document_text_column(
            documents
        )
    )

    X_text = (
        df[
            text_column
        ]
        .tolist()
    )

    y = (
        df[
            "category"
        ]
        .tolist()
    )

    if len(
        set(y)
    ) < 2:

        return (
            pd.DataFrame(),
            {}
        )

    # --------------------------------------------------------
    # Use stratification only when possible
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X_text,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y
            )
        )

    except Exception:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X_text,
                y,
                test_size=0.25,
                random_state=42
            )
        )

    strategies = [
        (
            "Count Vectorizer",
            CountVectorizer(
                stop_words="english",
                ngram_range=(1, 1)
            )
        ),
        (
            "TF-IDF Unigram",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 1)
            )
        ),
        (
            "TF-IDF Unigram + Bigram",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "TF-IDF No Stopword Removal",
            TfidfVectorizer(
                stop_words=None,
                ngram_range=(1, 1)
            )
        )
    ]

    rows = []

    detailed_results = {}

    for strategy_name, vectorizer_obj in strategies:

        try:

            X_train_features = (
                vectorizer_obj.fit_transform(
                    X_train
                )
            )

            X_test_features = (
                vectorizer_obj.transform(
                    X_test
                )
            )

            classifier = LogisticRegression(
                max_iter=2000
            )

            classifier.fit(
                X_train_features,
                y_train
            )

            predictions = (
                classifier.predict(
                    X_test_features
                )
            )

            accuracy = (
                accuracy_score(
                    y_test,
                    predictions
                )
            )

            feature_count = (
                X_train_features.shape[1]
            )

            rows.append(
                {
                    "Strategy":
                        strategy_name,

                    "# Features":
                        feature_count,

                    "Test Accuracy":
                        round(
                            float(
                                accuracy
                            ),
                            4
                        )
                }
            )

            detailed_results[
                strategy_name
            ] = {
                "vectorizer":
                    vectorizer_obj,

                "classifier":
                    classifier,

                "y_test":
                    y_test,

                "predictions":
                    predictions,

                "accuracy":
                    accuracy,

                "feature_count":
                    feature_count
            }

        except Exception as exc:

            rows.append(
                {
                    "Strategy":
                        strategy_name,

                    "# Features":
                        0,

                    "Test Accuracy":
                        0.0
                }
            )

            detailed_results[
                strategy_name
            ] = {
                "error":
                    str(exc)
            }

    return (
        pd.DataFrame(rows),
        detailed_results
    )


# ============================================================
# RUN EXISTING PYTHON MODULE
# ============================================================

def run_python_module(
    module_name
):

    module_path = os.path.join(
        SRC_DIR,
        module_name
    )

    if not module_path.endswith(
        ".py"
    ):

        module_path += ".py"

    if not exists(
        module_path
    ):

        return (
            False,
            f"Module not found: {module_path}"
        )

    try:

        result = subprocess.run(
            [
                sys.executable,
                module_path
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:

            return (
                True,
                result.stdout
            )

        return (
            False,
            result.stderr
        )

    except subprocess.TimeoutExpired:

        return (
            False,
            "Process timed out after 300 seconds."
        )

    except Exception as exc:

        return (
            False,
            str(exc)
        )


# ============================================================
# LOAD DATA
# ============================================================

documents = load_documents()

keywords = load_keywords()

corpus_statistics = (
    load_corpus_statistics()
)

inverted_index = (
    load_inverted_index()
)

index_metadata = (
    load_index_metadata()
)

vectorizer, tfidf_matrix = (
    load_tfidf_model()
)

pagerank = load_pagerank()

graph = load_graph()

evaluation = load_evaluation()

per_query_evaluation = (
    load_per_query_evaluation()
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔎 Information Retrieval System"
)

st.caption(
    "End-to-End Information Retrieval • "
    "Crawling • Text Mining • TF-IDF • "
    "PageRank • Hybrid Ranking • Recommendations • Evaluation"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🕷️ Crawling",
        "🧪 Text Mining",
        "📚 Index Management",
        "🔎 Search",
        "💡 Recommendations",
        "📄 Document Profile",
        "🔗 PageRank",
        "📊 Analytics",
        "📈 Evaluation"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "System Dashboard"
    )

    st.write(
        """
        This application demonstrates an end-to-end
        Information Retrieval system. The complete workflow
        covers document acquisition, preprocessing, text mining,
        indexing, search, PageRank, hybrid ranking,
        recommendation, evaluation, and analytics.
        """
    )

    st.divider()

    total_documents = len(
        documents
    )

    total_terms = (
        index_metadata.get(
            "unique_terms",
            len(inverted_index)
        )
    )

    total_postings = (
        index_metadata.get(
            "total_postings",
            0
        )
    )

    if (
        "category"
        in documents.columns
    ):

        total_categories = (
            documents[
                "category"
            ].nunique()
        )

    else:

        total_categories = 0

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Documents",
        total_documents
    )

    col2.metric(
        "Unique Terms",
        total_terms
    )

    col3.metric(
        "Index Postings",
        total_postings
    )

    col4.metric(
        "Categories",
        total_categories
    )

    st.divider()

    st.subheader(
        "Complete IR Lifecycle"
    )

    st.code(
        """
Web Sources
    ↓
Web Crawling
    ↓
Duplicate URL / Document Handling
    ↓
Text Preprocessing
    ↓
Text Mining
 ┌───────────────┬──────────────────┐
 ↓               ↓                  ↓
Keywords    Document Profile   Classification
 └───────────────┴──────────────────┘
                    ↓
             Index Management
                    ↓
              TF-IDF Retrieval
                    ↓
                PageRank
                    ↓
             Hybrid Ranking
                    ↓
          Content Recommendation
                    ↓
               Evaluation
                    ↓
          Performance Analytics
        """,
        language="text"
    )

    st.success(
        "Use the sidebar to access every major IR component."
    )


# ============================================================
# CRAWLING
# ============================================================

elif page == "🕷️ Crawling":

    st.header(
        "🕷️ Web Crawling Interface"
    )

    st.write(
        """
        Configure the document acquisition process.
        The current repository contains the collected corpus
        under data/raw/ and the processed metadata under
        data/processed/.
        """
    )

    st.subheader(
        "Crawling Configuration"
    )

    seed_urls_text = st.text_area(
        "Seed URL(s)",
        placeholder=(
            "Enter one URL per line\n"
            "https://example.com\n"
            "https://example.org"
        ),
        height=120
    )

    col1, col2 = st.columns(2)

    with col1:

        crawl_depth = st.number_input(
            "Crawling Depth",
            min_value=0,
            max_value=5,
            value=1,
            step=1
        )

    with col2:

        max_documents = st.number_input(
            "Maximum Documents",
            min_value=1,
            max_value=500,
            value=50,
            step=1
        )

    col1, col2 = st.columns(2)

    with col1:

        remove_duplicate_urls = st.checkbox(
            "Remove Duplicate URLs",
            value=True
        )

    with col2:

        remove_duplicate_documents = st.checkbox(
            "Remove Duplicate Documents",
            value=True
        )

    st.divider()

    st.subheader(
        "Current Corpus"
    )

    raw_document_count = 0

    if exists(
        RAW_DIR
    ):

        raw_document_count = len(
            [
                file
                for file in os.listdir(
                    RAW_DIR
                )
                if file.lower().endswith(
                    ".html"
                )
            ]
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Raw Documents",
        raw_document_count
    )

    col2.metric(
        "Processed Documents",
        len(documents)
    )

    col3.metric(
        "Configured Maximum",
        max_documents
    )

    st.divider()

    st.info(
        """
        The existing dataset is already collected and stored in
        data/raw/. The controls above record the crawling
        configuration required for the end-to-end workflow.
        """
    )

    if st.button(
        "🚀 Start Crawling / Refresh Corpus",
        type="primary"
    ):

        if not seed_urls_text.strip():

            st.warning(
                "Please enter at least one seed URL."
            )

        else:

            seed_urls = [
                url.strip()
                for url in seed_urls_text.splitlines()
                if url.strip()
            ]

            st.write(
                "**Configured Seeds:**"
            )

            for url in seed_urls:

                st.write(
                    f"- {url}"
                )

            st.write(
                f"**Crawling depth:** {crawl_depth}"
            )

            st.write(
                f"**Maximum documents:** {max_documents}"
            )

            st.write(
                f"**Duplicate URL removal:** "
                f"{'Enabled' if remove_duplicate_urls else 'Disabled'}"
            )

            st.write(
                f"**Duplicate document removal:** "
                f"{'Enabled' if remove_duplicate_documents else 'Disabled'}"
            )

            st.warning(
                """
                The currently stored corpus is used by the application.
                To perform a new web crawl, the crawler implementation
                in src/dataset_builder.py must support the supplied
                crawling parameters.
                """
            )

    st.divider()

    st.subheader(
        "Acquisition Artifacts"
    )

    artifact_data = {
        "Artifact": [
            "Raw HTML Documents",
            "Document Metadata",
            "Document Content",
            "Document Links",
            "Processed Documents"
        ],
        "Path": [
            "data/raw/",
            "data/processed/documents.csv",
            "data/processed/content.csv",
            "data/processed/doc_links.csv",
            "data/processed/processed_documents.csv"
        ],
        "Available": [
            exists(RAW_DIR),
            exists(DOCUMENTS_FILE),
            exists(CONTENT_FILE),
            exists(DOC_LINKS_FILE),
            exists(PROCESSED_FILE)
        ]
    }

    st.dataframe(
        pd.DataFrame(
            artifact_data
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TEXT MINING
# ============================================================

elif page == "🧪 Text Mining":

    st.header(
        "🧪 Text Preprocessing and Mining"
    )

    st.write(
        """
        This module demonstrates document profiling,
        keyword analysis, document classification, and
        comparative feature extraction strategies.
        """
    )

    # --------------------------------------------------------
    # Corpus statistics
    # --------------------------------------------------------

    st.subheader(
        "Corpus Statistics"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Documents",
        len(documents)
    )

    if "category" in documents.columns:

        col2.metric(
            "Categories",
            documents[
                "category"
            ].nunique()
        )

    else:

        col2.metric(
            "Categories",
            0
        )

    if "processed_word_count" in documents.columns:

        col3.metric(
            "Average Words",
            f"{documents['processed_word_count'].mean():.1f}"
        )

    else:

        col3.metric(
            "Average Words",
            "N/A"
        )

    if "unique_terms" in documents.columns:

        col4.metric(
            "Average Unique Terms",
            f"{documents['unique_terms'].mean():.1f}"
        )

    else:

        col4.metric(
            "Average Unique Terms",
            "N/A"
        )

    st.divider()

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    st.subheader(
        "🔑 Keyword Extraction"
    )

    if keywords.empty:

        st.info(
            "Keyword data is not available."
        )

    else:

        st.dataframe(
            keywords.head(100),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # Document profile summary
    # --------------------------------------------------------

    st.subheader(
        "📄 Document Profiling"
    )

    profile_columns = [
        column
        for column in [
            "doc_id",
            "title",
            "category",
            "processed_word_count",
            "unique_terms"
        ]
        if column in documents.columns
    ]

    if profile_columns:

        st.dataframe(
            documents[
                profile_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    st.subheader(
        "🏷️ Document Classification"
    )

    classification_df = (
        get_classification_dataframe(
            documents
        )
    )

    if classification_df.empty:

        st.warning(
            """
            Document classification cannot be performed because
            a suitable text column and category column were not found.
            """
        )

    else:

        st.write(
            f"Classification dataset contains "
            f"**{len(classification_df)} documents**."
        )

        st.write(
            "Category distribution:"
        )

        st.bar_chart(
            classification_df[
                "category"
            ].value_counts()
        )

        st.divider()

        if st.button(
            "🧪 Run Feature Comparison",
            type="primary"
        ):

            with st.spinner(
                "Training classification models..."
            ):

                comparison_df, detailed_results = (
                    evaluate_feature_strategies(
                        documents
                    )
                )

            if comparison_df.empty:

                st.error(
                    "Feature comparison could not be completed."
                )

            else:

                st.session_state[
                    "feature_comparison"
                ] = comparison_df

                st.session_state[
                    "classification_results"
                ] = detailed_results

        if (
            "feature_comparison"
            in st.session_state
        ):

            comparison_df = (
                st.session_state[
                    "feature_comparison"
                ]
            )

            st.subheader(
                "Comparative Feature Analysis"
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Feature Strategy Accuracy"
            )

            chart_df = comparison_df.set_index(
                "Strategy"
            )[
                "Test Accuracy"
            ]

            st.bar_chart(
                chart_df
            )

            st.subheader(
                "Feature Counts"
            )

            feature_df = comparison_df.set_index(
                "Strategy"
            )[
                "# Features"
            ]

            st.bar_chart(
                feature_df
            )

            st.divider()

            st.subheader(
                "Classification Details"
            )

            detailed_results = st.session_state.get(
                "classification_results",
                {}
            )

            strategy_names = list(
                detailed_results.keys()
            )

            if strategy_names:

                selected_strategy = st.selectbox(
                    "Select Feature Strategy",
                    strategy_names
                )

                result = detailed_results[
                    selected_strategy
                ]

                if "error" in result:

                    st.error(
                        result["error"]
                    )

                else:

                    st.metric(
                        "Test Accuracy",
                        f"{result['accuracy']:.4f}"
                    )

                    cm = confusion_matrix(
                        result["y_test"],
                        result["predictions"]
                    )

                    st.write(
                        "Confusion Matrix"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            cm
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    report = classification_report(
                        result["y_test"],
                        result["predictions"],
                        output_dict=True,
                        zero_division=0
                    )

                    report_df = (
                        pd.DataFrame(
                            report
                        )
                        .transpose()
                    )

                    st.write(
                        "Classification Report"
                    )

                    st.dataframe(
                        report_df,
                        use_container_width=True
                    )


# ============================================================
# INDEX MANAGEMENT
# ============================================================

elif page == "📚 Index Management":

    st.header(
        "📚 Index Management"
    )

    st.write(
        """
        Inspect the inverted index and TF-IDF indexing artifacts
        used by the Information Retrieval system.
        """
    )

    index_exists = exists(
        INVERTED_INDEX_FILE
    )

    metadata_exists = exists(
        INDEX_METADATA_FILE
    )

    vectorizer_exists = exists(
        TFIDF_VECTORIZER_FILE
    )

    matrix_exists = exists(
        TFIDF_MATRIX_FILE
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Inverted Index",
        "Ready" if index_exists else "Missing"
    )

    col2.metric(
        "Metadata",
        "Ready" if metadata_exists else "Missing"
    )

    col3.metric(
        "TF-IDF Vectorizer",
        "Ready" if vectorizer_exists else "Missing"
    )

    col4.metric(
        "TF-IDF Matrix",
        "Ready" if matrix_exists else "Missing"
    )

    st.divider()

    st.subheader(
        "Index Statistics"
    )

    total_documents = len(
        documents
    )

    unique_terms = index_metadata.get(
        "unique_terms",
        len(inverted_index)
    )

    total_postings = index_metadata.get(
        "total_postings",
        0
    )

    matrix_rows = 0
    matrix_columns = 0

    if tfidf_matrix is not None:

        try:

            matrix_rows = (
                tfidf_matrix.shape[0]
            )

            matrix_columns = (
                tfidf_matrix.shape[1]
            )

        except Exception:

            pass

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Indexed Documents",
        total_documents
    )

    col2.metric(
        "Unique Terms",
        unique_terms
    )

    col3.metric(
        "Index Postings",
        total_postings
    )

    col4.metric(
        "TF-IDF Features",
        matrix_columns
    )

    st.divider()

    st.subheader(
        "TF-IDF Matrix"
    )

    st.write(
        f"Rows (documents): **{matrix_rows}**"
    )

    st.write(
        f"Columns (features): **{matrix_columns}**"
    )

    st.divider()

    st.subheader(
        "Index Metadata"
    )

    if index_metadata:

        metadata_rows = []

        for key, value in index_metadata.items():

            metadata_rows.append(
                {
                    "Property": key,
                    "Value": value
                }
            )

        st.dataframe(
            pd.DataFrame(
                metadata_rows
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Index metadata is not available."
        )

    st.divider()

    st.subheader(
        "Index Files"
    )

    index_files = [
        INVERTED_INDEX_FILE,
        INDEX_METADATA_FILE,
        TFIDF_VECTORIZER_FILE,
        TFIDF_MATRIX_FILE
    ]

    file_rows = []

    for path in index_files:

        file_rows.append(
            {
                "File":
                    os.path.relpath(
                        path,
                        PROJECT_ROOT
                    ),

                "Exists":
                    exists(path),

                "Size (KB)":
                    round(
                        os.path.getsize(path) / 1024,
                        2
                    )
                    if exists(path)
                    else 0
            }
        )

    st.dataframe(
        pd.DataFrame(
            file_rows
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Index Operations"
    )

    if st.button(
        "🔄 Rebuild Index",
        type="primary"
    ):

        with st.spinner(
            "Running indexing.py..."
        ):

            success, output = (
                run_python_module(
                    "indexing.py"
                )
            )

        if success:

            st.success(
                "Indexing completed successfully."
            )

            with st.expander(
                "Indexing Output"
            ):

                st.text(
                    output
                )

            st.cache_data.clear()
            st.cache_resource.clear()

            st.info(
                "Reload the page to load the newly generated index."
            )

        else:

            st.error(
                "Indexing failed."
            )

            st.code(
                output
            )


# ============================================================
# SEARCH
# ============================================================

elif page == "🔎 Search":

    st.header(
        "🔎 Search Engine"
    )

    st.write(
        """
        Enter a query and retrieve documents using TF-IDF
        or Hybrid TF-IDF + PageRank ranking.
        """
    )

    query = st.text_input(
        "Enter Search Query",
        placeholder=(
            "Example: machine learning classification"
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        ranking_method = st.selectbox(
            "Ranking Method",
            [
                "TF-IDF",
                "Hybrid"
            ]
        )

    with col2:

        top_k = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

    search_button = st.button(
        "🔍 Search",
        type="primary"
    )

    if search_button:

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        elif documents.empty:

            st.error(
                "Document dataset is empty."
            )

        elif vectorizer is None:

            st.error(
                "TF-IDF model was not found."
            )

        else:

            start_time = time.perf_counter()

            with st.spinner(
                "Searching documents..."
            ):

                if ranking_method == "TF-IDF":

                    results = (
                        perform_tfidf_search(
                            query,
                            documents,
                            vectorizer,
                            tfidf_matrix,
                            top_k
                        )
                    )

                else:

                    results = (
                        perform_hybrid_search(
                            query,
                            documents,
                            vectorizer,
                            tfidf_matrix,
                            pagerank,
                            top_k
                        )
                    )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            st.caption(
                f"Query execution time: "
                f"{elapsed:.6f} seconds"
            )

            if results.empty:

                st.info(
                    "No matching documents were found."
                )

            else:

                st.success(
                    f"{len(results)} results found."
                )

                for _, row in (
                    results.iterrows()
                ):

                    title = get_title(
                        row
                    )

                    doc_id = get_doc_id(
                        row
                    )

                    category = get_category(
                        row
                    )

                    url = get_url(
                        row
                    )

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            f"{int(row['rank'])}. {title}"
                        )

                        st.write(
                            f"**Document ID:** {doc_id}"
                        )

                        st.write(
                            f"**Category:** {category}"
                        )

                        st.write(
                            f"**TF-IDF Score:** "
                            f"{row['score']:.4f}"
                        )

                        if ranking_method == "Hybrid":

                            st.write(
                                f"**PageRank:** "
                                f"{row.get('pagerank', 0):.6f}"
                            )

                            st.write(
                                f"**Hybrid Score:** "
                                f"{row.get('hybrid_score', 0):.4f}"
                            )

                        if url.startswith(
                            "http"
                        ):

                            st.link_button(
                                "Open Source",
                                url
                            )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.header(
        "💡 Content-Based Recommendations"
    )

    st.write(
        """
        The recommendation system uses TF-IDF document
        representations and cosine similarity to find
        similar documents.
        """
    )

    if documents.empty:

        st.error(
            "Documents are not available."
        )

    elif tfidf_matrix is None:

        st.error(
            "TF-IDF matrix is not available."
        )

    else:

        document_options = {}

        for _, row in (
            documents.iterrows()
        ):

            doc_id = str(
                row["doc_id"]
            )

            title = str(
                row["title"]
            )

            document_options[
                f"{doc_id} — {title}"
            ] = doc_id

        selected_label = st.selectbox(
            "Select Document",
            list(
                document_options.keys()
            )
        )

        selected_doc_id = (
            document_options[
                selected_label
            ]
        )

        top_k = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5
        )

        if st.button(
            "Generate Recommendations",
            type="primary"
        ):

            recommendations = (
                get_recommendations(
                    selected_doc_id,
                    documents,
                    tfidf_matrix,
                    top_k
                )
            )

            if recommendations.empty:

                st.info(
                    "No recommendations found."
                )

            else:

                st.subheader(
                    "Recommended Documents"
                )

                for _, row in (
                    recommendations.iterrows()
                ):

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            f"{int(row['rank'])}. "
                            f"{row['title']}"
                        )

                        st.write(
                            f"**Document ID:** "
                            f"{row['doc_id']}"
                        )

                        st.write(
                            f"**Category:** "
                            f"{row['category']}"
                        )

                        st.write(
                            f"**Cosine Similarity:** "
                            f"{row['similarity']:.4f}"
                        )

                        if str(
                            row["url"]
                        ).startswith(
                            "http"
                        ):

                            st.link_button(
                                "Open Source",
                                row["url"]
                            )


# ============================================================
# DOCUMENT PROFILE
# ============================================================

elif page == "📄 Document Profile":

    st.header(
        "📄 Document Profile"
    )

    if documents.empty:

        st.error(
            "Documents are not available."
        )

    else:

        document_options = {}

        for _, row in (
            documents.iterrows()
        ):

            doc_id = str(
                row["doc_id"]
            )

            document_options[
                f"{doc_id} — {row['title']}"
            ] = doc_id

        selected_label = st.selectbox(
            "Select Document",
            list(
                document_options.keys()
            )
        )

        selected_doc_id = (
            document_options[
                selected_label
            ]
        )

        selected = documents[
            documents[
                "doc_id"
            ].astype(str)
            == selected_doc_id
        ]

        if not selected.empty:

            row = selected.iloc[0]

            st.subheader(
                str(
                    row["title"]
                )
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Document ID",
                str(
                    row["doc_id"]
                )
            )

            if (
                "processed_word_count"
                in row
            ):

                col2.metric(
                    "Word Count",
                    safe_int(
                        row[
                            "processed_word_count"
                        ]
                    )
                )

            else:

                col2.metric(
                    "Word Count",
                    "N/A"
                )

            if (
                "unique_terms"
                in row
            ):

                col3.metric(
                    "Unique Terms",
                    safe_int(
                        row[
                            "unique_terms"
                        ]
                    )
                )

            else:

                col3.metric(
                    "Unique Terms",
                    "N/A"
                )

            st.divider()

            st.write(
                f"**Category:** "
                f"{row.get('category', 'N/A')}"
            )

            st.write(
                f"**Source:** "
                f"{row.get('source', 'N/A')}"
            )

            url = str(
                row.get(
                    "url",
                    ""
                )
            )

            if url.startswith(
                "http"
            ):

                st.link_button(
                    "Open Original Source",
                    url
                )

            st.subheader(
                "🔑 Top Keywords"
            )

            if not keywords.empty:

                document_keywords = (
                    keywords[
                        keywords[
                            "doc_id"
                        ].astype(str)
                        == selected_doc_id
                    ]
                )

                if not document_keywords.empty:

                    available_columns = [
                        column
                        for column in [
                            "rank",
                            "keyword",
                            "tfidf_score"
                        ]
                        if column in
                        document_keywords.columns
                    ]

                    st.dataframe(
                        document_keywords[
                            available_columns
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No keyword information available."
                    )

            st.subheader(
                "📄 Document Content"
            )

            content = str(
                row.get(
                    "content",
                    ""
                )
            )

            if len(content) > 10000:

                content = (
                    content[:10000]
                    +
                    "\n\n[Content truncated]"
                )

            st.text_area(
                "Content",
                content,
                height=350
            )


# ============================================================
# PAGERANK
# ============================================================

elif page == "🔗 PageRank":

    st.header(
        "🔗 PageRank Analysis"
    )

    if pagerank.empty:

        st.error(
            "PageRank results are not available."
        )

    else:

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Documents",
            len(pagerank)
        )

        col2.metric(
            "Average PageRank",
            f"{pagerank['pagerank'].mean():.6f}"
        )

        col3.metric(
            "Maximum PageRank",
            f"{pagerank['pagerank'].max():.6f}"
        )

        st.divider()

        st.subheader(
            "Top Documents by PageRank"
        )

        max_documents = min(
            30,
            len(pagerank)
        )

        default_documents = min(
            10,
            max_documents
        )

        top_n = st.slider(
            "Number of Documents",
            min_value=1,
            max_value=max_documents,
            value=default_documents
        )

        top_pagerank = (
            pagerank
            .sort_values(
                "pagerank",
                ascending=False
            )
            .head(
                top_n
            )
        )

        st.dataframe(
            top_pagerank,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "PageRank Distribution"
        )

        st.bar_chart(
            top_pagerank.set_index(
                "doc_id"
            )[
                "pagerank"
            ]
        )

        if graph:

            st.divider()

            st.subheader(
                "Document Graph Statistics"
            )

            nodes = len(
                graph.get(
                    "nodes",
                    []
                )
            )

            edges = len(
                graph.get(
                    "edges",
                    []
                )
            )

            col1, col2 = (
                st.columns(2)
            )

            col1.metric(
                "Graph Nodes",
                nodes
            )

            col2.metric(
                "Graph Edges",
                edges
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.header(
        "📊 Corpus and Performance Analytics"
    )

    summary_file = os.path.join(
        VISUALIZATION_DIR,
        "corpus_summary.csv"
    )

    if exists(
        summary_file
    ):

        summary = pd.read_csv(
            summary_file
        )

        st.subheader(
            "Corpus Summary"
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # Performance information
    # --------------------------------------------------------

    st.subheader(
        "⚡ System Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Documents",
        len(documents)
    )

    col2.metric(
        "Index Terms",
        len(inverted_index)
    )

    col3.metric(
        "TF-IDF Features",
        tfidf_matrix.shape[1]
        if tfidf_matrix is not None
        else 0
    )

    col4.metric(
        "Graph Nodes",
        len(
            graph.get(
                "nodes",
                []
            )
        )
        if graph
        else 0
    )

    st.divider()

    visualization_files = [
        (
            "Document Length Distribution",
            "document_length_distribution.png"
        ),
        (
            "Category Distribution",
            "category_distribution.png"
        ),
        (
            "Top 20 Corpus Terms",
            "top_terms.png"
        ),
        (
            "PageRank Distribution",
            "pagerank_distribution.png"
        ),
        (
            "Top PageRank Documents",
            "top_pagerank_documents.png"
        ),
        (
            "TF-IDF vs Hybrid Ranking",
            "tfidf_vs_hybrid.png"
        )
    ]

    for title, filename in visualization_files:

        image_path = os.path.join(
            VISUALIZATION_DIR,
            filename
        )

        if exists(
            image_path
        ):

            st.subheader(
                title
            )

            st.image(
                image_path,
                use_container_width=True
            )


# ============================================================
# EVALUATION
# ============================================================

elif page == "📈 Evaluation":

    st.header(
        "📈 Retrieval Evaluation"
    )

    if evaluation.empty:

        st.error(
            "Evaluation results are not available."
        )

    else:

        st.subheader(
            "Overall Evaluation Results"
        )

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        if (
            "method"
            in evaluation.columns
        ):

            methods = (
                evaluation[
                    "method"
                ]
                .astype(str)
                .tolist()
            )

            selected_method = st.selectbox(
                "Select Ranking Method",
                methods
            )

            selected = evaluation[
                evaluation[
                    "method"
                ].astype(str)
                == selected_method
            ]

            if not selected.empty:

                row = selected.iloc[0]

                required_metrics = [
                    (
                        "Precision",
                        "precision"
                    ),
                    (
                        "Recall",
                        "recall"
                    ),
                    (
                        "F1",
                        "f1"
                    ),
                    (
                        "Precision@K",
                        "precision_at_k"
                    ),
                    (
                        "Recall@K",
                        "recall_at_k"
                    ),
                    (
                        "MAP",
                        "map"
                    ),
                    (
                        "MRR",
                        "mrr"
                    ),
                    (
                        "NDCG@10",
                        "ndcg_at_10"
                    )
                ]

                available_metrics = [
                    item
                    for item in required_metrics
                    if item[1] in row.index
                ]

                if available_metrics:

                    columns = st.columns(
                        min(
                            len(
                                available_metrics
                            ),
                            4
                        )
                    )

                    for index, (
                        label,
                        key
                    ) in enumerate(
                        available_metrics
                    ):

                        columns[
                            index % len(columns)
                        ].metric(
                            label,
                            f"{safe_float(row[key]):.4f}"
                        )

        st.divider()

        if not per_query_evaluation.empty:

            st.subheader(
                "Per-Query Evaluation"
            )

            st.dataframe(
                per_query_evaluation,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        image_path = os.path.join(
            VISUALIZATION_DIR,
            "tfidf_vs_hybrid.png"
        )

        if exists(
            image_path
        ):

            st.subheader(
                "TF-IDF vs Hybrid Comparison"
            )

            st.image(
                image_path,
                use_container_width=True
            )

        st.divider()

        st.subheader(
            "Metric Definitions"
        )

        st.markdown(
            """
**Precision**  
Proportion of retrieved documents that are relevant.

**Recall**  
Proportion of relevant documents that were retrieved.

**F1-score**  
Harmonic mean of precision and recall.

**Precision@K**  
Precision among the top K retrieved documents.

**Recall@K**  
Recall among the top K retrieved documents.

**MAP**  
Mean Average Precision across evaluation queries.

**MRR**  
Mean Reciprocal Rank of the first relevant result.

**NDCG**  
Normalized Discounted Cumulative Gain, which evaluates
ranking quality while considering the position of relevant
documents.
            """
        )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Information Retrieval Assignment - 2"
)

st.sidebar.caption(
    "TF-IDF • PageRank • Hybrid Ranking • "
    "Content-Based Recommendation"
)