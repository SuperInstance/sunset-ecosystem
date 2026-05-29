"""Pagination utilities for API responses.

Provides pagination helpers for offset/limit and cursor-based pagination.
Used for fleet API endpoints that return large result sets.

Usage:
    pager = PaginationHelper(items=[1,2,3,4,5], page_size=2)
    page = pager.page(1)
    assert page.items == [1, 2]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Page:
    """A page of results."""

    items: List[Any]
    page: int
    page_size: int
    total: int
    has_next: bool
    has_prev: bool


class PaginationHelper(Generic[T]):
    """
    Offset/limit pagination helper.

    :param items: Full list of items.
    :param page_size: Items per page.
    """

    def __init__(self, items: List[T], page_size: int = 20):
        self._items = list(items)
        self._page_size = page_size

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def page(self, page_num: int) -> Page:
        """
        Get a page by 1-based page number.

        :param page_num: 1-based page number.
        """
        if page_num < 1:
            page_num = 1
        start = (page_num - 1) * self._page_size
        end = start + self._page_size
        items = self._items[start:end]
        total = len(self._items)
        return Page(
            items=items,
            page=page_num,
            page_size=self._page_size,
            total=total,
            has_next=end < total,
            has_prev=page_num > 1,
        )

    def slice(self, offset: int, limit: int) -> List[T]:
        """Get a raw slice."""
        return self._items[offset:offset + limit]

    def first(self) -> Optional[T]:
        """Get first item."""
        return self._items[0] if self._items else None

    def last(self) -> Optional[T]:
        """Get last item."""
        return self._items[-1] if self._items else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def total(self) -> int:
        return len(self._items)

    def page_count(self) -> int:
        if not self._items:
            return 0
        return (len(self._items) + self._page_size - 1) // self._page_size

    def __repr__(self) -> str:
        return f"<PaginationHelper total={len(self._items)} page_size={self._page_size}>"
