import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FILE = (
    "data/processed/documents.csv"
)

LINK_FILE = (
    "data/processed/links.csv"
)

OUTPUT_FILE = (
    "data/processed/doc_links.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    documents = pd.read_csv(
        DOCUMENT_FILE
    )

    links = pd.read_csv(
        LINK_FILE
    )

    return documents, links


# ============================================================
# BUILD URL → DOCUMENT ID MAPPING
# ============================================================

def build_url_mapping(documents):

    url_to_doc = {}

    for _, row in documents.iterrows():

        url = str(
            row["url"]
        )

        doc_id = str(
            row["doc_id"]
        )

        url_to_doc[url] = doc_id

    return url_to_doc


# ============================================================
# BUILD DOCUMENT GRAPH
# ============================================================

def build_document_links(
    documents,
    links
):

    url_to_doc = build_url_mapping(
        documents
    )

    document_links = []

    for _, row in links.iterrows():

        source_url = str(
            row["source_url"]
        )

        target_url = str(
            row["target_url"]
        )

        # Source must exist in our corpus
        if source_url not in url_to_doc:
            continue

        # Target must also exist in our corpus
        if target_url not in url_to_doc:
            continue

        source_doc = url_to_doc[
            source_url
        ]

        target_doc = url_to_doc[
            target_url
        ]

        # Avoid self-links
        if source_doc == target_doc:
            continue

        document_links.append(
            {
                "source_doc": source_doc,
                "target_doc": target_doc
            }
        )

    result = pd.DataFrame(
        document_links
    )

    if result.empty:

        return result

    # Remove duplicate edges
    result = result.drop_duplicates()

    return result


# ============================================================
# SAVE GRAPH
# ============================================================

def save_graph(document_links):

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    document_links.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DOCUMENT GRAPH BUILDER")
    print("=" * 70)

    print("\nLoading documents and links...")

    documents, links = load_data()

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"URL links: {len(links)}"
    )

    print(
        "\nConverting URLs to document IDs..."
    )

    document_links = build_document_links(
        documents,
        links
    )

    if document_links.empty:

        print(
            "\nWARNING:"
        )

        print(
            "No links between documents "
            "were found."
        )

        print(
            "PageRank cannot be calculated "
            "from an empty graph."
        )

        return

    save_graph(
        document_links
    )

    print("\n")
    print("=" * 70)
    print("DOCUMENT GRAPH CREATED")
    print("=" * 70)

    print(
        f"Documents in corpus : "
        f"{len(documents)}"
    )

    print(
        f"Document links      : "
        f"{len(document_links)}"
    )

    print(
        f"Output              : "
        f"{OUTPUT_FILE}"
    )

    print("\nSample links:")

    print(
        document_links.head(20)
        .to_string(index=False)
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()