"""Small helpers for the JSON hyperparameter searches."""

from __future__ import annotations

import json
from typing import Any

from sklearn.model_selection import ParameterGrid, ParameterSampler


def parameter_candidates(
    search_space: dict[str, list[Any]],
    n_iter: int | None,
    random_state: int = 42,
    fixed_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build either the full grid or a reproducible random sample."""
    if not search_space:
        raise ValueError("tuning.search_space cannot be empty")

    empty_parameters = [name for name, values in search_space.items() if not values]
    if empty_parameters:
        raise ValueError(
            "Every tuning parameter needs at least one value: "
            f"{sorted(empty_parameters)}"
        )

    grid = ParameterGrid(search_space)
    if n_iter is None:
        sampled = list(grid)
    else:
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        sampled = list(
            ParameterSampler(
                search_space,
                n_iter=min(n_iter, len(grid)),
                random_state=random_state,
            )
        )

    candidates = list(fixed_candidates or []) + sampled
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique
