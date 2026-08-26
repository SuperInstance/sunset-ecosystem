"""Tests for pagination_helper.py — Pagination utilities.

Run: python3 -m pytest tests/test_pagination_helper.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.pagination_helper import PaginationHelper


class TestPaginationHelper:
    def test_create(self):
        pager = PaginationHelper([1, 2, 3], page_size=2)
        assert pager.total() == 3
        assert pager.page_count() == 2

    def test_page(self):
        pager = PaginationHelper([1, 2, 3, 4, 5], page_size=2)
        p1 = pager.page(1)
        assert p1.items == [1, 2]
        assert p1.has_next is True
        assert p1.has_prev is False
        p2 = pager.page(2)
        assert p2.items == [3, 4]
        assert p2.has_next is True
        assert p2.has_prev is True
        p3 = pager.page(3)
        assert p3.items == [5]
        assert p3.has_next is False
        assert p3.has_prev is True

    def test_page_zero(self):
        pager = PaginationHelper([1, 2], page_size=2)
        p = pager.page(0)
        assert p.page == 1
        assert p.items == [1, 2]

    def test_slice(self):
        pager = PaginationHelper([1, 2, 3, 4, 5])
        assert pager.slice(1, 2) == [2, 3]

    def test_first_last(self):
        pager = PaginationHelper([1, 2, 3])
        assert pager.first() == 1
        assert pager.last() == 3

    def test_empty(self):
        pager = PaginationHelper([], page_size=2)
        assert pager.total() == 0
        assert pager.page_count() == 0
        assert pager.first() is None
        assert pager.last() is None

    def test_repr(self):
        pager = PaginationHelper([1, 2, 3])
        assert "PaginationHelper" in repr(pager)
