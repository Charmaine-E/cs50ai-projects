import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    
    """
    pages = list(corpus)
    n = len(pages)
    links = corpus[page]

    # Dangling page: treat as linking to every page (including itself)
    if not links:
        return {p: 1 / n for p in pages}

    dist = {p: (1 - damping_factor) / n for p in pages}
    for link in links:
        dist[link] += damping_factor / len(links)
    return dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    at random.

    """
    pages = list(corpus)
    counts = {p: 0 for p in pages}

    page = random.choice(pages)
    counts[page] += 1

    for _ in range(n - 1):
        dist = transition_model(corpus, page, damping_factor)
        page = random.choices(list(dist), weights=list(dist.values()))[0]
        counts[page] += 1

    return {p: counts[p] / n for p in pages}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values 
    
    """
    n = len(corpus)
    ranks = {p: 1 / n for p in corpus}

    # Dangling pages are treated as linking to every page
    outlinks = {p: (corpus[p] if corpus[p] else set(corpus)) for p in corpus}

    while True:
        new_ranks = {}
        for p in corpus:
            total = sum(
                ranks[i] / len(outlinks[i])
                for i in corpus if p in outlinks[i]
            )
            new_ranks[p] = (1 - damping_factor) / n + damping_factor * total

        converged = all(abs(new_ranks[p] - ranks[p]) <= 0.001 for p in corpus)
        ranks = new_ranks
        if converged:
            break

    return ranks


if __name__ == "__main__":
    main()