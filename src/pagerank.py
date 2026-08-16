import os
import json

import pandas as pd
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FILE = (
    "data/processed/documents.csv"
)

GRAPH_FILE = (
    "data/processed/doc_links.csv"
)

OUTPUT_FILE = (
    "results/pagerank_scores.csv"
)

GRAPH_JSON_FILE = (
    "results/document_graph.json"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_documents():

    return pd.read_csv(
        DOCUMENT_FILE
    )


def load_graph_links():

    return pd.read_csv(
        GRAPH_FILE
    )


# ============================================================
# BUILD NETWORKX GRAPH
# ============================================================

def build_graph(
    documents,
    links
):

    graph = nx.DiGraph()

    # --------------------------------------------------------
    # Add every document as a node
    # --------------------------------------------------------

    for _, row in documents.iterrows():

        doc_id = str(
            row["doc_id"]
        )

        graph.add_node(
            doc_id,
            title=str(row["title"]),
            category=str(row["category"])
        )

    # --------------------------------------------------------
    # Add document relationships
    # --------------------------------------------------------

    for _, row in links.iterrows():

        source = str(
            row["source_doc"]
        )

        target = str(
            row["target_doc"]
        )

        if (
            source in graph
            and
            target in graph
        ):

            graph.add_edge(
                source,
                target
            )

    return graph


# ============================================================
# CALCULATE PAGERANK
# ============================================================

def calculate_pagerank(
    graph
):

    print(
        "\nCalculating PageRank..."
    )

    scores = nx.pagerank(
        graph,
        alpha=0.85
    )

    return scores


# ============================================================
# SAVE SCORES
# ============================================================

def save_scores(
    documents,
    scores
):

    rows = []

    for _, row in documents.iterrows():

        doc_id = str(
            row["doc_id"]
        )

        rows.append(
            {
                "doc_id": doc_id,

                "title": str(
                    row["title"]
                ),

                "category": str(
                    row["category"]
                ),

                "pagerank": float(
                    scores.get(
                        doc_id,
                        0.0
                    )
                )
            }
        )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        by="pagerank",
        ascending=False
    )

    results = results.reset_index(
        drop=True
    )

    results["rank"] = (
        results.index + 1
    )

    results = results[
        [
            "rank",
            "doc_id",
            "title",
            "category",
            "pagerank"
        ]
    ]

    os.makedirs(
        "results",
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return results


# ============================================================
# SAVE GRAPH
# ============================================================

def save_graph(graph):

    graph_data = {
        "nodes": [],
        "edges": []
    }

    for node, attributes in (
        graph.nodes(data=True)
    ):

        graph_data["nodes"].append(
            {
                "id": node,
                "title": attributes.get(
                    "title",
                    ""
                ),
                "category": attributes.get(
                    "category",
                    ""
                )
            }
        )

    for source, target in (
        graph.edges()
    ):

        graph_data["edges"].append(
            {
                "source": source,
                "target": target
            }
        )

    with open(
        GRAPH_JSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            graph_data,
            file,
            indent=2
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    graph,
    results
):

    print("\n")
    print("=" * 80)
    print("PAGERANK RESULTS")
    print("=" * 80)

    print(
        f"\nNumber of nodes : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Number of edges : "
        f"{graph.number_of_edges()}"
    )

    print("\nTop 10 documents:\n")

    print(
        results.head(10).to_string(
            index=False
        )
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PAGERANK DOCUMENT RANKING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    documents = load_documents()

    links = load_graph_links()

    print(
        f"\nDocuments loaded: "
        f"{len(documents)}"
    )

    print(
        f"Document links loaded: "
        f"{len(links)}"
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    graph = build_graph(
        documents,
        links
    )

    print(
        f"\nGraph nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges()}"
    )

    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    scores = calculate_pagerank(
        graph
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results = save_scores(
        documents,
        scores
    )

    save_graph(
        graph
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        graph,
        results
    )

    print("\nOutput files:")

    print(
        f"PageRank scores: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Graph: "
        f"{GRAPH_JSON_FILE}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()