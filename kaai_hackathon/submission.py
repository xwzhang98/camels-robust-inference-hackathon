"""Loading and validating a team submission.

A submission is a **directory**, not a file of predictions. It contains ``predict.py``
exposing two functions::

    def load_model(model_dir: str) -> object
    def predict(model, catalog_path: str) -> dict   # {"Omega_m": ..., "sigma_8": ...}

``load_model`` is called once; ``predict`` is called once per test catalog. Anything
expensive -- unpickling, moving weights to the GPU -- belongs in ``load_model``.

A submission is a whole directory, not one file. `predict.py` may import its neighbours --
`gnn.py`, a dataloader, whatever it needs -- so the directory goes on `sys.path` and stays
there, because an import inside `predict()` has to work too, not only the ones at the top.

That has a consequence worth knowing: **score one submission per process.** Two teams whose
helper modules share a name would otherwise get each other's, since `sys.modules` is keyed
by module name and the first one loaded wins. `predict.py` itself is safe -- it is
registered under a name derived from its directory -- but `gnn.py` is not, and the failure
would be silent. The evaluation driver runs a separate process per team for this reason.

Predictions may additionally carry the four feedback parameters for the bonus track. Any
other key is ignored rather than rejected, so a team can return debug information without
failing the harness.

Teams submit code rather than predictions because the test catalogs are never published.
That is what makes the held-out simulation code genuinely unseen.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

from kaai_hackathon import PARAM_NAMES

REQUIRED_KEYS = ("Omega_m", "sigma_8")


def load_submission(sub_dir: str | Path) -> tuple[object, Callable]:
    """Import ``<sub_dir>/predict.py``, call ``load_model``, return ``(model, predict)``.

    Every team's entry point has the same filename, so the module is registered under a
    name derived from the directory. Two submissions loaded in one process do not shadow
    each other.
    """
    sub_dir = Path(sub_dir).resolve()
    entry = sub_dir / "predict.py"
    if not entry.is_file():
        raise FileNotFoundError(f"submission is missing predict.py: {entry}")

    # The submission's own directory comes first, so `import gnn` finds the team's gnn.py
    # rather than anything of ours or the environment's that happens to share the name.
    if str(sub_dir) not in sys.path:
        sys.path.insert(0, str(sub_dir))

    module_name = f"kaai_hackathon_submission_{abs(hash(str(sub_dir))):x}"
    spec = importlib.util.spec_from_file_location(module_name, entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    for name in ("load_model", "predict"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"predict.py must define {name}()")
    return module.load_model(str(sub_dir)), module.predict


def validate_prediction(pred: dict) -> dict[str, float]:
    """Check one prediction and coerce it to plain floats.

    Rejects a missing required key and any non-finite value. A NaN that reaches the scorer
    turns an entire condition into NaN, which is much harder to trace back than an error
    raised on the catalog that produced it.
    """
    if not isinstance(pred, dict):
        raise ValueError(f"predict() must return a dict, got {type(pred).__name__}")
    for key in REQUIRED_KEYS:
        if key not in pred:
            raise ValueError(f"prediction is missing required key: {key}")
    out: dict[str, float] = {}
    for key, value in pred.items():
        if key not in PARAM_NAMES:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction for {key} is not a number: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"prediction for {key} is not finite: {number}")
        out[key] = number
    return out
