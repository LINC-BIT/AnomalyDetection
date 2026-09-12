"""Scan-order index tables for `SS2D`'s multi-directional selective scan.

MambaAD flattens an `H x W` feature grid into a 1D sequence five different
ways before feeding it to the (inherently 1D) selective-scan recurrence,
so that a state that only propagates along a 1D path still sees
2D-local neighbors as scan-neighbors under at least one ordering. This is
a straight reimplementation of the *algorithms* named in the paper
(row-major sweep, boustrophedon "scan", zigzag diagonals, Z-order/Morton,
and Hilbert curve) — not a port of upstream's `HSCANS`, which pulled
`pyzorder`/`hilbert` as extra pip dependencies for exactly this. Both are
well-known, few-line curve constructions, so they are inlined here in pure
NumPy instead of adding two more third-party packages for them.

`size` must be a power of two (`Hilbert` requires it; the others don't but
are kept to the same constraint for one shared code path).
"""

from __future__ import annotations

import numpy as np

SCAN_TYPES = ("sweep", "scan", "zigzag", "zorder", "hilbert")


def _sweep_order(size: int) -> np.ndarray:
    return np.arange(size * size)


def _boustrophedon_order(size: int) -> np.ndarray:
    indexes = np.arange(size * size).reshape(size, size)
    for row in range(1, size, 2):
        indexes[row, :] = indexes[row, :][::-1]
    return indexes.reshape(-1)


def _zigzag_order(size: int) -> np.ndarray:
    indexes = np.arange(size * size).reshape(size, size)
    order: list[int] = []
    for diagonal in range(2 * size - 1):
        lo = max(0, diagonal - size + 1)
        hi = min(diagonal, size - 1)
        rows = range(lo, hi + 1) if diagonal % 2 == 0 else range(hi, lo - 1, -1)
        for row in rows:
            order.append(int(indexes[diagonal - row, row]))
    return np.array(order)


def _zorder_order(size: int) -> np.ndarray:
    """Morton order: interleave the bits of (row, col)."""

    bits = int(np.log2(size))
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    rows, cols = rows.reshape(-1), cols.reshape(-1)
    morton = np.zeros_like(rows)
    for bit in range(bits):
        morton |= ((rows >> bit) & 1) << (2 * bit)
        morton |= ((cols >> bit) & 1) << (2 * bit + 1)
    # `morton` maps (row, col) -> its position on the curve; scan order is
    # cell indices sorted by that position.
    return np.argsort(morton)


def _hilbert_d2xy(bits: int, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard Hilbert-curve distance-to-(x, y) conversion (Wikipedia's
    `d2xy` algorithm), vectorized over `index`.

    The quadrant rotation flips against the *current sub-square size* `s`
    (not the full grid `n`) -- verified against a scalar port of the
    reference algorithm, since a wrong bound here still produces *some*
    permutation, just not a curve whose consecutive steps stay grid-adjacent.
    """

    x = np.zeros_like(index)
    y = np.zeros_like(index)
    t = index.copy()
    s = 1
    while s < (1 << bits):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        ry_zero = ry == 0
        flip = ry_zero & (rx == 1)
        x_work = np.where(flip, s - 1 - x, x)
        y_work = np.where(flip, s - 1 - y, y)
        # Swap x/y whenever ry == 0 (regardless of rx), after the flip above.
        x, y = np.where(ry_zero, y_work, x_work), np.where(ry_zero, x_work, y_work)
        x = x + s * rx
        y = y + s * ry
        t = t // 4
        s *= 2
    return x, y


def _hilbert_order(size: int) -> np.ndarray:
    bits = int(np.log2(size))
    distances = np.arange(size * size)
    x, y = _hilbert_d2xy(bits, distances)
    # d2xy already maps step -> (x, y), i.e. step -> cell -- the same
    # convention `scan_order` returns directly, unlike `_zorder_order`
    # (which computes cell -> step and has to invert it).
    return x * size + y


_BUILDERS = {
    "sweep": _sweep_order,
    "scan": _boustrophedon_order,
    "zigzag": _zigzag_order,
    "zorder": _zorder_order,
    "hilbert": _hilbert_order,
}


def scan_order(size: int, scan_type: str) -> np.ndarray:
    """`locs_flat[i]` = which flattened grid cell is visited at scan step
    `i`; `argsort(locs_flat)` is its inverse (grid cell -> scan step).
    """

    if scan_type not in _BUILDERS:
        raise ValueError(f"unknown scan_type {scan_type!r}; expected one of {SCAN_TYPES}")
    if scan_type == "hilbert" and (size & (size - 1)) != 0:
        raise ValueError(f"hilbert scan requires a power-of-two size, got {size}")
    return _BUILDERS[scan_type](size)
