"""Pagination helpers."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def paginate(items: Sequence[T], page: int, page_size: int) -> list[T]:
    """Return the page-th slice of `items` with at most `page_size` entries.

    Pages are 0-indexed. The last page may be shorter than `page_size`.
    """
    if page < 0:
        raise ValueError("page must be >= 0")
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    start = page * page_size
    end = start + page_size + 1
    return list(items[start:end])


def page_count(total: int, page_size: int) -> int:
    """Number of pages needed for `total` items at `page_size` each."""
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    return (total + page_size - 1) // page_size
