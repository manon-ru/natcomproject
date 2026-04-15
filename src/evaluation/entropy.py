from math import log2


def calculate_shannon_entropy(positions: list[tuple]) -> float:
    """
    Compute the Shannon entropy of a population based on agent positions.

    A higher value means agents are spread across many distinct cells
    (diverse). A value near 0 means most agents are at the same cell
    (converged / stuck).

    Args:
        positions: List of (x, y) tuples — one per agent or individual.

    Returns:
        Entropy in bits (log base 2).
    """
    if not positions:
        return 0.0

    counts: dict[tuple, int] = {}
    for pos in positions:
        counts[pos] = counts.get(pos, 0) + 1

    n = len(positions)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * log2(p)

    return entropy
