import os
import sys
import json
import pickle

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity


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
# IMPORT PREPROCESSING
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

            # If preprocessing returns a list
            if isinstance(
                processed,
                list
            ):

                return " ".join(
                    str(x)
                    for x in processed
                )

            # If preprocessing returns a string
            return str(
                processed
            )

        except Exception:
            pass

    return text.lower()


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    if not exists(
        PROCESSED_FILE
    ):

        return pd.DataFrame()

    return pd.read_csv(
        PROCESSED_FILE
    )


# ============================================================
# LOAD KEYWORDS
# ============================================================

@st.cache_data
def load_keywords():

    if not exists(
        KEYWORDS_FILE
    ):

        return pd.DataFrame()

    return pd.read_csv(
        KEYWORDS_FILE
    )


# ============================================================
# LOAD CORPUS STATISTICS
# ============================================================

@st.cache_data
def load_corpus_statistics():

    if not exists(
        CORPUS_STATS_FILE
    ):

        return pd.DataFrame()

    return pd.read_csv(
        CORPUS_STATS_FILE
    )


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
# SEARCH FUNCTION
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

    # --------------------------------------------------------
    # Preprocess query
    # --------------------------------------------------------

    processed_query = normalize_text(
        query
    )

    # --------------------------------------------------------
    # Convert query into TF-IDF vector
    # --------------------------------------------------------

    try:

        query_vector = (
            vectorizer.transform(
                [processed_query]
            )
        )

    except Exception:

        # Fallback to raw query
        try:

            query_vector = (
                vectorizer.transform(
                    [query]
                )
            )

        except Exception:

            return pd.DataFrame()

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    scores = (
        cosine_similarity(
            query_vector,
            tfidf_matrix
        )
        .flatten()
    )

    # --------------------------------------------------------
    # Create result table
    # --------------------------------------------------------

    results = documents.copy()

    results[
        "score"
    ] = scores

    # --------------------------------------------------------
    # Remove zero-score documents
    # --------------------------------------------------------

    results = results[
        results[
            "score"
        ] > 0
    ]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    results[
        "rank"
    ] = (
        results.index
        + 1
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

    # --------------------------------------------------------
    # Merge PageRank
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize PageRank
    # --------------------------------------------------------

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
            ]
            - pr_min
        ) / (
            pr_max
            - pr_min
        )

    else:

        results[
            "pagerank_normalized"
        ] = 0.0

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

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
        results.index
        + 1
    )

    return results


# ============================================================
# CONTENT-BASED RECOMMENDATIONS
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
                    row["category"],

                "url":
                    row["url"],

                "similarity":
                    float(score)
            }
        )

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

    if not result.empty:

        result[
            "rank"
        ] = (
            result.index
            + 1
        )

    return result


# ============================================================
# LOAD ALL DATA
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
# APPLICATION HEADER
# ============================================================

st.title(
    "🔎 Information Retrieval System"
)

st.caption(
    "Web Search • TF-IDF • PageRank • Hybrid Ranking • Recommendations"
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
        Information Retrieval system developed for the
        assignment.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # System pipeline
    # --------------------------------------------------------

    st.subheader(
        "IR System Pipeline"
    )

    st.code(
        """
Web Sources
    ↓
Dataset Construction
    ↓
Text Preprocessing
    ↓
Keyword Extraction
    ↓
Document Profiling
    ↓
Inverted Index + TF-IDF
    ↓
Search
    ↓
PageRank
    ↓
Hybrid Ranking
    ↓
Recommendation
    ↓
Evaluation
    ↓
Analytics
        """,
        language="text"
    )

    st.success(
        "All major IR modules are available through the sidebar."
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
        Enter a query and retrieve documents using
        TF-IDF or Hybrid TF-IDF + PageRank ranking.
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
                "TF-IDF model was not found. "
                "Please run the indexing/search preparation first."
            )

        else:

            with st.spinner(
                "Searching documents..."
            ):

                if (
                    ranking_method
                    == "TF-IDF"
                ):

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

                    title = str(
                        row.get(
                            "title",
                            "Untitled"
                        )
                    )

                    doc_id = str(
                        row.get(
                            "doc_id",
                            ""
                        )
                    )

                    category = str(
                        row.get(
                            "category",
                            "Unknown"
                        )
                    )

                    url = str(
                        row.get(
                            "url",
                            ""
                        )
                    )

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            f"{int(row['rank'])}. "
                            f"{title}"
                        )

                        st.write(
                            f"**Document ID:** {doc_id}"
                        )

                        st.write(
                            f"**Category:** {category}"
                        )

                        if (
                            ranking_method
                            == "TF-IDF"
                        ):

                            st.write(
                                f"**TF-IDF Score:** "
                                f"{row['score']:.4f}"
                            )

                        else:

                            st.write(
                                f"**TF-IDF Score:** "
                                f"{row['score']:.4f}"
                            )

                            st.write(
                                f"**PageRank:** "
                                f"{row['pagerank']:.6f}"
                            )

                            st.write(
                                f"**Hybrid Score:** "
                                f"{row['hybrid_score']:.4f}"
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
        Select a document to find other documents
        with similar TF-IDF representations.
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
                str(row["doc_id"])
            )

            if (
                "processed_word_count"
                in row
            ):

                col2.metric(
                    "Word Count",
                    int(
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
                    int(
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

            # ------------------------------------------------
            # Keywords
            # ------------------------------------------------

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
                        if column
                        in document_keywords.columns
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

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

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

        # ----------------------------------------------------
        # PageRank chart
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Graph statistics
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Corpus Summary
    # --------------------------------------------------------

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

    else:

        st.warning(
            "Corpus summary not found."
        )

    st.divider()

    # --------------------------------------------------------
    # Document Length
    # --------------------------------------------------------

    image_path = os.path.join(
        VISUALIZATION_DIR,
        "document_length_distribution.png"
    )

    if exists(
        image_path
    ):

        st.subheader(
            "Document Length Distribution"
        )

        st.image(
            image_path,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Category Distribution
    # --------------------------------------------------------

    image_path = os.path.join(
        VISUALIZATION_DIR,
        "category_distribution.png"
    )

    if exists(
        image_path
    ):

        st.subheader(
            "Category Distribution"
        )

        st.image(
            image_path,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Top Terms
    # --------------------------------------------------------

    image_path = os.path.join(
        VISUALIZATION_DIR,
        "top_terms.png"
    )

    if exists(
        image_path
    ):

        st.subheader(
            "Top 20 Corpus Terms"
        )

        st.image(
            image_path,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PageRank Distribution
    # --------------------------------------------------------

    image_path = os.path.join(
        VISUALIZATION_DIR,
        "pagerank_distribution.png"
    )

    if exists(
        image_path
    ):

        st.subheader(
            "PageRank Distribution"
        )

        st.image(
            image_path,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Top PageRank Documents
    # --------------------------------------------------------

    image_path = os.path.join(
        VISUALIZATION_DIR,
        "top_pagerank_documents.png"
    )

    if exists(
        image_path
    ):

        st.subheader(
            "Top PageRank Documents"
        )

        st.image(
            image_path,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TF-IDF vs Hybrid
    # --------------------------------------------------------

    image_path = os.path.join(
        VISUALIZATION_DIR,
        "tfidf_vs_hybrid.png"
    )

    if exists(
        image_path
    ):

        st.subheader(
            "TF-IDF vs Hybrid Ranking"
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

        # ----------------------------------------------------
        # Metric cards
        # ----------------------------------------------------

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

                metric_names = [
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

                columns = st.columns(
                    len(metric_names)
                )

                for column, (
                    label,
                    key
                ) in zip(
                    columns,
                    metric_names
                ):

                    if key in row:

                        column.metric(
                            label,
                            f"{safe_float(row[key]):.4f}"
                        )

        st.divider()

        # ----------------------------------------------------
        # Per query
        # ----------------------------------------------------

        if not per_query_evaluation.empty:

            st.subheader(
                "Per-Query Evaluation"
            )

            st.dataframe(
                per_query_evaluation,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # Evaluation chart
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Metric explanation
        # ----------------------------------------------------

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

            **Precision@5**  
            Precision among the top 5 retrieved documents.

            **Recall@5**  
            Recall among the top 5 retrieved documents.

            **MAP**  
            Mean Average Precision across the evaluation queries.

            **MRR**  
            Mean Reciprocal Rank of the first relevant result.

            **NDCG@10**  
            Measures ranking quality while considering the position
            of relevant documents in the top 10 results.
            """
        )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Information Retrieval Assignment"
)

st.sidebar.caption(
    "TF-IDF • PageRank • Hybrid Ranking • Recommendations"
)