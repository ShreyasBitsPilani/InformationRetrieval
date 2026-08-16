import os
import re
import csv
import time
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

MAX_DEPTH = 1
MAX_PAGES = 50

REQUEST_TIMEOUT = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


# Multiple seed sources
SEED_URLS = {
    "Information Retrieval":
        "https://en.wikipedia.org/wiki/Information_retrieval",

    "Machine Learning":
        "https://en.wikipedia.org/wiki/Machine_learning",

    "Artificial Intelligence":
        "https://en.wikipedia.org/wiki/Artificial_intelligence",

    "Natural Language Processing":
        "https://en.wikipedia.org/wiki/Natural_language_processing",

    "Data Mining":
        "https://en.wikipedia.org/wiki/Data_mining",
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """
    Create directories required for storing the dataset.
    """

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Directories ready.")


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):
    """
    Normalize a URL so that small URL variations
    do not create duplicate documents.
    """

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove fragments
    path = parsed.path.rstrip("/")

    if not path:
        path = "/"

    normalized = f"{scheme}://{netloc}{path}"

    return normalized


# ============================================================
# URL VALIDATION
# ============================================================

def is_valid_url(url):
    """
    Check whether the URL is an HTTP/HTTPS URL.
    """

    try:
        parsed = urlparse(url)

        return parsed.scheme in ["http", "https"] and bool(parsed.netloc)

    except Exception:
        return False


# ============================================================
# DOWNLOAD WEB PAGE
# ============================================================

def fetch_page(url):
    """
    Download a web page and return its HTML.
    """

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:

            content_type = response.headers.get(
                "Content-Type",
                ""
            )

            if "text/html" in content_type:

                return response.text

        print(
            f"Could not fetch: {url} "
            f"(status={response.status_code})"
        )

        return None

    except requests.RequestException as e:

        print(f"Request error for {url}: {e}")

        return None


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text(html):
    """
    Extract readable text from HTML.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove unnecessary elements
    for element in soup(
        ["script", "style", "noscript", "svg"]
    ):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT TITLE
# ============================================================

def extract_title(html):
    """
    Extract the title of the web page.
    """

    soup = BeautifulSoup(html, "html.parser")

    if soup.title:

        title = soup.title.get_text(
            strip=True
        )

        return title

    return "Untitled Document"


# ============================================================
# EXTRACT LINKS
# ============================================================

def extract_links(html, base_url):
    """
    Extract HTTP/HTTPS links from a page.
    """

    soup = BeautifulSoup(html, "html.parser")

    links = set()

    for anchor in soup.find_all("a", href=True):

        href = anchor.get("href")

        absolute_url = urljoin(
            base_url,
            href
        )

        normalized = normalize_url(
            absolute_url
        )

        if is_valid_url(normalized):

            links.add(normalized)

    return links


# ============================================================
# DOCUMENT HASH
# ============================================================

def generate_document_hash(text):
    """
    Generate a SHA-256 hash for document content.

    This helps identify duplicate documents.
    """

    normalized_text = re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


# ============================================================
# SAFE FILE NAME
# ============================================================

def create_filename(doc_id):
    """
    Create a safe filename for storing raw HTML.
    """

    return os.path.join(
        RAW_DIR,
        f"{doc_id}.html"
    )


# ============================================================
# CRAWLER
# ============================================================

def crawl(seed_urls, max_depth=1, max_pages=50):
    """
    Crawl pages starting from multiple seed URLs.

    Parameters:
        seed_urls  : dictionary {category: url}
        max_depth  : maximum crawling depth
        max_pages   : maximum number of pages

    Returns:
        documents
        links
        crawl_statistics
    """

    # Queue stores:
    # (URL, depth, category)

    queue = deque()

    for category, url in seed_urls.items():

        normalized_url = normalize_url(url)

        queue.append(
            (
                normalized_url,
                0,
                category
            )
        )

    # Track URLs that have already been processed
    visited_urls = set()

    # Track document hashes
    document_hashes = set()

    documents = []

    links = []

    duplicate_urls = 0
    duplicate_documents = 0

    doc_counter = 1

    while queue and len(documents) < max_pages:

        current_url, depth, category = queue.popleft()

        current_url = normalize_url(
            current_url
        )

        # ----------------------------------------------------
        # Duplicate URL check
        # ----------------------------------------------------

        if current_url in visited_urls:

            duplicate_urls += 1

            continue

        visited_urls.add(
            current_url
        )

        print(
            f"\n[{len(documents) + 1}] "
            f"Crawling: {current_url}"
        )

        print(
            f"Depth: {depth}"
        )

        # ----------------------------------------------------
        # Fetch page
        # ----------------------------------------------------

        html = fetch_page(
            current_url
        )

        if html is None:

            continue

        # ----------------------------------------------------
        # Extract information
        # ----------------------------------------------------

        title = extract_title(
            html
        )

        text = extract_text(
            html
        )

        if not text:

            continue

        # ----------------------------------------------------
        # Duplicate document check
        # ----------------------------------------------------

        document_hash = generate_document_hash(
            text
        )

        if document_hash in document_hashes:

            duplicate_documents += 1

            print(
                "Duplicate document detected."
            )

            continue

        document_hashes.add(
            document_hash
        )

        # ----------------------------------------------------
        # Generate document ID
        # ----------------------------------------------------

        doc_id = f"DOC{doc_counter:04d}"

        doc_counter += 1

        # ----------------------------------------------------
        # Store raw HTML
        # ----------------------------------------------------

        raw_filename = create_filename(
            doc_id
        )

        with open(
            raw_filename,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(html)

        # ----------------------------------------------------
        # Document metadata
        # ----------------------------------------------------

        word_count = len(
            text.split()
        )

        document = {

            "doc_id": doc_id,

            "title": title,

            "category": category,

            "url": current_url,

            "source": urlparse(
                current_url
            ).netloc,

            "crawl_depth": depth,

            "word_count": word_count,

            "content_hash": document_hash,

            "content": text,
        }

        documents.append(
            document
        )

        print(
            f"Stored {doc_id}: {title}"
        )

        # ----------------------------------------------------
        # Extract links
        # ----------------------------------------------------

        page_links = extract_links(
            html,
            current_url
        )

        # ----------------------------------------------------
        # Add links to graph
        # ----------------------------------------------------

        for link in page_links:

            links.append(
                {
                    "source_url": current_url,
                    "target_url": link
                }
            )

        # ----------------------------------------------------
        # Continue crawling
        # ----------------------------------------------------

        if depth < max_depth:

            for link in page_links:

                if link not in visited_urls:

                    queue.append(
                        (
                            link,
                            depth + 1,
                            category
                        )
                    )

        # Small delay between requests
        time.sleep(0.5)

    # ========================================================
    # STATISTICS
    # ========================================================

    statistics = {

        "documents": len(documents),

        "unique_urls": len(visited_urls),

        "duplicate_urls": duplicate_urls,

        "duplicate_documents": duplicate_documents,

        "links": len(links),

    }

    return (
        documents,
        links,
        statistics
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(documents, links):
    """
    Save metadata, content and link information
    into separate CSV files.
    """

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_file = os.path.join(
        PROCESSED_DIR,
        "documents.csv"
    )

    metadata_fields = [
        "doc_id",
        "title",
        "category",
        "url",
        "source",
        "crawl_depth",
        "word_count",
        "content_hash",
    ]

    with open(
        metadata_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=metadata_fields
        )

        writer.writeheader()

        for document in documents:

            metadata = {
                field: document[field]
                for field in metadata_fields
            }

            writer.writerow(
                metadata
            )

    # --------------------------------------------------------
    # Content
    # --------------------------------------------------------

    content_file = os.path.join(
        PROCESSED_DIR,
        "content.csv"
    )

    with open(
        content_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "doc_id",
                "content"
            ]
        )

        writer.writeheader()

        for document in documents:

            writer.writerow(
                {
                    "doc_id": document["doc_id"],
                    "content": document["content"]
                }
            )

    # --------------------------------------------------------
    # Links
    # --------------------------------------------------------

    links_file = os.path.join(
        PROCESSED_DIR,
        "links.csv"
    )

    with open(
        links_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_url",
                "target_url"
            ]
        )

        writer.writeheader()

        for link in links:

            writer.writerow(
                link
            )

    print("\nDataset saved successfully.")

    print(
        f"Metadata : {metadata_file}"
    )

    print(
        f"Content  : {content_file}"
    )

    print(
        f"Links    : {links_file}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "INFORMATION RETRIEVAL DATASET BUILDER"
    )

    print("=" * 60)

    create_directories()

    print("\nStarting crawler...")

    print(
        f"Maximum depth : {MAX_DEPTH}"
    )

    print(
        f"Maximum pages : {MAX_PAGES}"
    )

    print(
        f"Seed sources  : {len(SEED_URLS)}"
    )

    documents, links, statistics = crawl(
        SEED_URLS,
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES
    )

    save_dataset(
        documents,
        links
    )

    # --------------------------------------------------------
    # Print final statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)

    print("CRAWLING SUMMARY")

    print("=" * 60)

    print(
        f"Documents collected   : "
        f"{statistics['documents']}"
    )

    print(
        f"Unique URLs visited   : "
        f"{statistics['unique_urls']}"
    )

    print(
        f"Duplicate URLs        : "
        f"{statistics['duplicate_urls']}"
    )

    print(
        f"Duplicate documents   : "
        f"{statistics['duplicate_documents']}"
    )

    print(
        f"Links discovered      : "
        f"{statistics['links']}"
    )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()