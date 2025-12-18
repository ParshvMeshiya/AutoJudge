import re

# ============================================================
# Basic text statistics
# ============================================================

def text_length(text: str) -> int:
    return len(text)


def word_count(text: str) -> int:
    return len(text.split())


def math_symbol_count(text: str) -> int:
    return len(re.findall(r"[+\-*/%=<>]", text))


# ============================================================
# Keyword-based difficulty signals
# ============================================================

DIFFICULTY_KEYWORDS = [
    "dp",
    "dynamic programming",
    "graph",
    "tree",
    "dfs",
    "bfs",
    "shortest path",
    "bitmask",
    "greedy",
    "segment tree",
    "binary search",
    "mod",
    "modulo"
]


def keyword_features(text: str) -> list:
    text = text.lower()
    return [text.count(kw) for kw in DIFFICULTY_KEYWORDS]


# ============================================================
# Constraint-aware features
# ============================================================

def constraint_features(text: str) -> list:
    text = text.lower()
    return [
        int("10^5" in text or "1e5" in text),
        int("10^6" in text or "1e6" in text),
        int("2e5" in text),
        int("constraints" in text),
        int("time limit" in text)
    ]


CONSTRAINT_FEATURE_NAMES = [
    "has_1e5",
    "has_1e6",
    "has_2e5",
    "mentions_constraints",
    "mentions_time_limit"
]


# ============================================================
# Input structure complexity
# ============================================================

def input_structure_features(text: str) -> list:
    text = text.lower()
    return [
        int("test case" in text or "test cases" in text),
        int("edge" in text or "edges" in text),
        int("matrix" in text or "grid" in text),
        int("tree" in text),
        int("directed" in text or "undirected" in text)
    ]


STRUCTURE_FEATURE_NAMES = [
    "multiple_tests",
    "graph_edges",
    "matrix_or_grid",
    "tree_structure",
    "graph_direction"
]


# ============================================================
# Algorithmic depth
# ============================================================

ADVANCED_TOPICS = [
    "dp",
    "dynamic programming",
    "bitmask",
    "segment tree",
    "fenwick",
    "flow",
    "matching",
    "fft",
    "suffix",
    "binary lifting"
]


def algorithmic_depth(text: str) -> int:
    text = text.lower()
    return sum(1 for kw in ADVANCED_TOPICS if kw in text)


# ============================================================
# Story noise reduction
# ============================================================

def extract_technical_text(text: str) -> str:
    lines = text.split("\n")
    return " ".join(lines[:8])
