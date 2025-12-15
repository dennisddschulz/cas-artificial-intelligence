"""
models_foundation.py
====================

Dieser Modul dient als Einstiegspunkt für moderne SOTA-Zeitreihenmodelle,
welche die Studierenden im Rahmen der Projektarbeit integrieren sollen.

Mindestens **ein** zusätzliches Modell muss eingebaut und evaluiert werden.
Beispiele:

- Chronos 2 (Amazon)
- TSMixer
- TiDE (Meta)
- NLinear / DLinear
- N-BEATS / N-HiTS
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
from darts.models import TiDEModel
#

def create_tide():
     return TiDEModel(
         input_chunk_length=64,
         output_chunk_length=30,
     )


# -----------------------------------------------------------
# TODO 4 — Beispiel: Lineare SOTA Modelle
# -----------------------------------------------------------
# from darts.models import NLinearModel, DLinearModel
#
# def create_nlinear():
#     return NLinearModel(
#         input_chunk_length=64,
#         output_chunk_length=30,
#     )
#
# def create_dlinear():
#     return DLinearModel(
#         input_chunk_length=64,
#         output_chunk_length=30,
#     )
