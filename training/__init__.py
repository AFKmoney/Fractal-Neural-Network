"""
FNN — boucles d'entrainement et composants associes.

Note: trainer.py, losses.py, agi_trainer.py et distributed.py sont en cours de
reconnexion au moteur FNN (Phase 4). Les modules auxiliaires (intrinsic, value,
continual, online_learner) sont operationnels des maintenant.
"""
try:
    from .losses import NFNLoss  # noqa: F401
except Exception:
    pass
try:
    from .trainer import NFNTrainer  # noqa: F401
except Exception:
    pass

__all__ = [
    "NFNLoss", "NFNTrainer",
    # modules auxiliaires (Phase 2) — importer explicitement si besoin
    "intrinsic", "value", "continual", "online_learner",
]
