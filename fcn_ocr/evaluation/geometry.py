from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PointMatch:
    expected: float
    predicted: float

    @property
    def error(self) -> float:
        return abs(self.expected - self.predicted)


def match_sorted_points(
    expected: Iterable[float],
    predicted: Iterable[float],
    tolerance: float,
) -> list[PointMatch]:
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    expected_values = sorted(float(value) for value in expected)
    predicted_values = sorted(float(value) for value in predicted)
    rows = len(expected_values) + 1
    cols = len(predicted_values) + 1
    scores: list[list[tuple[int, float]]] = [
        [(0, 0.0) for _ in range(cols)] for _ in range(rows)
    ]
    choices = [["" for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        choices[i][0] = "expected"
    for j in range(1, cols):
        choices[0][j] = "predicted"

    def better(left: tuple[int, float], right: tuple[int, float]) -> bool:
        return left[0] > right[0] or (left[0] == right[0] and left[1] < right[1])

    for i in range(1, rows):
        for j in range(1, cols):
            best = scores[i - 1][j]
            choice = "expected"
            if better(scores[i][j - 1], best):
                best = scores[i][j - 1]
                choice = "predicted"

            error = abs(expected_values[i - 1] - predicted_values[j - 1])
            if error <= tolerance:
                previous = scores[i - 1][j - 1]
                matched = (previous[0] + 1, previous[1] + error)
                if better(matched, best):
                    best = matched
                    choice = "match"
            scores[i][j] = best
            choices[i][j] = choice

    matches: list[PointMatch] = []
    i = len(expected_values)
    j = len(predicted_values)
    while i > 0 or j > 0:
        choice = choices[i][j]
        if choice == "match":
            matches.append(PointMatch(expected_values[i - 1], predicted_values[j - 1]))
            i -= 1
            j -= 1
        elif choice == "expected":
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


def interpolate_polyline(points: list[list[float]], xs: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("A baseline polyline requires at least two points")
    ordered = sorted((float(point[0]), float(point[1])) for point in points)
    unique_x: list[float] = []
    unique_y: list[float] = []
    for x, y in ordered:
        if unique_x and x == unique_x[-1]:
            unique_y[-1] = y
        else:
            unique_x.append(x)
            unique_y.append(y)
    if len(unique_x) < 2:
        raise ValueError("A baseline polyline requires at least two distinct X coordinates")
    return np.interp(
        xs.astype(np.float64),
        np.asarray(unique_x, dtype=np.float64),
        np.asarray(unique_y, dtype=np.float64),
    )


def polyline_x_bounds(points: list[list[float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("A baseline polyline requires at least two points")
    xs = [float(point[0]) for point in points]
    return min(xs), max(xs)
