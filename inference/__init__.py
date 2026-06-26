"""
FNN — moteurs d'inference et de generation.

Note: engine.py est en cours de reconnexion au moteur FNN (Phase 4).
streaming.py (long-contexte chunked) est operationnel des maintenant.
"""
try:
    from .engine import NFNInferenceEngine  # noqa: F401
except Exception:
    pass

__all__ = [
    "NFNInferenceEngine",
    # module auxiliaire (Phase 2) — importer explicitement si besoin
    "streaming", "fast_numpy",
]
