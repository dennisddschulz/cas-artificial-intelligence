"""
models_foundation.py
====================

Factory functions for simple SOTA-style linear foundation models using Darts:
 - DLinearModel
 - NLinearModel

These are configured via `FoundationConfig`.
"""

# -----------------------------------------------------------
# TODO 1 — Beispiel: Chronos 2 (Foundation Model)
# -----------------------------------------------------------
# from darts.models import Chronos2Model
#
# def load_chronos2_model():
#     """
#     Laden eines vortrainierten Chronos 2 Foundation Models.
#     Hinweis: Chronos2 kann NICHT fine-getuned werden.
#     """
#     model = Chronos2Model.from_pretrained("amazon/chronos-tiny")
#     return model


# -----------------------------------------------------------
# TODO 2 — Beispiel: TSMixer
# -----------------------------------------------------------
# from darts.models import TSMixerModel
#
# def create_tsmixer():
#     """
#     Initialisiert einen TSMixer zur Prognose.
#     """
#     return TSMixerModel(
#         input_chunk_length=64,
#         output_chunk_length=30,
#     )


# -----------------------------------------------------------
# TODO 3 — Beispiel: TiDE
# -----------------------------------------------------------
# from darts.models import TiDEModel
#
# def create_tide():
#     return TiDEModel(
#         input_chunk_length=64,
#         output_chunk_length=30,
#     )


from __future__ import annotations

from typing import Dict, Any

from darts.models import NLinearModel, DLinearModel
from src.config import FoundationConfig


def _trainer_kwargs(use_gpu: bool) -> Dict[str, Any]:
    """Return `pl_trainer_kwargs` depending on GPU flag.

    We avoid hard requirements; if GPU is not available, Lightning will fall back gracefully.
    """
    if use_gpu:
        return {"accelerator": "gpu", "devices": 1}
    else:
        return {"accelerator": "cpu", "devices": 1}


def create_dlinear(cfg: FoundationConfig) -> DLinearModel:
    """Create a DLinearModel configured from `FoundationConfig`."""
    return DLinearModel(
        input_chunk_length=cfg.input_chunk_length,
        output_chunk_length=cfg.output_chunk_length,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        random_state=cfg.random_state,
        optimizer_kwargs={"lr": cfg.lr},
        pl_trainer_kwargs=_trainer_kwargs(cfg.use_gpu),
    )


def create_nlinear(cfg: FoundationConfig) -> NLinearModel:
    """Create an NLinearModel configured from `FoundationConfig`."""
    return NLinearModel(
        input_chunk_length=cfg.input_chunk_length,
        output_chunk_length=cfg.output_chunk_length,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        random_state=cfg.random_state,
        optimizer_kwargs={"lr": cfg.lr},
        pl_trainer_kwargs=_trainer_kwargs(cfg.use_gpu),
    )
