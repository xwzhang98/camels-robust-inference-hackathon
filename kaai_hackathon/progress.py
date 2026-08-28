"""A progress bar that degrades to printing, so a long loop is never silent.

Reading 2700 catalogs over a mounted bucket takes minutes and produces no output at all,
which is indistinguishable from a hung kernel. That is the failure this exists to prevent.

`tqdm` is present in Colab and in most environments, and is used when it is. When it is not,
this prints a line every few seconds instead. Neither path is a dependency: nothing here is
imported unless a caller asks for it.

    for path in track(paths, "reading catalogs"):
        ...

    bar = track(range(epochs), "training")
    for epoch in bar:
        ...
        bar.set_postfix(loss=value)      # a no-op without tqdm; still prints periodically
"""
from __future__ import annotations

import sys
import time
from typing import Any, Iterable, Iterator


class _Printing:
    """The fallback. Prints at most one line every `interval` seconds, plus a final one."""

    def __init__(self, iterable: Iterable, description: str, total: int | None,
                 interval: float = 5.0):
        self._iterable = iterable
        self._description = description
        self._total = total
        self._interval = interval
        self._postfix = ""
        self._started = time.time()
        self._last = 0.0

    def set_postfix(self, **fields: Any) -> None:
        self._postfix = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in fields.items())

    def _emit(self, done: int, final: bool = False) -> None:
        elapsed = time.time() - self._started
        where = f"{done}/{self._total}" if self._total else str(done)
        rate = f"  {elapsed / done:.2f}s each" if final and done else ""
        print(f"  {self._description}: {where}  [{elapsed:5.0f}s]{rate}  {self._postfix}",
              flush=True)

    def __iter__(self) -> Iterator:
        done = 0
        for item in self._iterable:
            yield item
            done += 1
            now = time.time()
            if now - self._last >= self._interval:
                self._last = now
                self._emit(done)
        self._emit(done, final=True)


def track(iterable: Iterable, description: str = "", total: int | None = None):
    """Wrap an iterable so its progress is visible. Never raises, never adds a dependency."""
    if total is None:
        try:
            total = len(iterable)          # type: ignore[arg-type]
        except TypeError:
            total = None
    try:
        if "ipykernel" in sys.modules:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        return tqdm(iterable, desc=description, total=total, leave=True)
    except Exception:
        return _Printing(iterable, description, total)
