"""The model architecture, per SUBMITTING.md's directory contract.

We never modified the reference de Santi et al. GNN -- ``kaai_hackathon.gnn.DeSantiGNN`` --
anywhere in this project, so there is nothing of our own to define here. This file exists
to satisfy the contract explicitly (and to make the dependency visible at a glance) by
re-exporting it, the same way the organizers' own shipped ``baseline/gnn/predict.py``
imports it directly rather than shipping a duplicate copy.
"""
from __future__ import annotations

from kaai_hackathon.gnn import DeSantiGNN, collate, moment_loss, scatter

__all__ = ["DeSantiGNN", "collate", "moment_loss", "scatter"]
