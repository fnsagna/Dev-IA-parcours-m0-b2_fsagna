"""Journalisation des appels à l'API, côté interface."""

import os
from pathlib import Path

from loguru import logger

# Un avis peut contenir un nom, un numéro de chambre, une adresse e-mail : jamais en entier
EXTRAIT_MAX = 80

DOSSIER_LOGS = Path(os.getenv("LOG_DIR", "logs"))

# Streamlit réexécute tout le script à chaque interaction : sans ce garde-fou,
# configurer() ajouterait un nouveau sink loguru (donc des lignes dupliquées) à chaque rerun.
_configure = False


def extrait(text: str) -> str:
    """Tronque un avis pour ne jamais journaliser une donnée personnelle complète."""
    propre = " ".join(text.split())
    return propre if len(propre) <= EXTRAIT_MAX else propre[:EXTRAIT_MAX] + "…"


def configurer() -> None:
    """Branche le fichier de logs, une seule fois par process."""
    global _configure
    if _configure:
        return
    DOSSIER_LOGS.mkdir(parents=True, exist_ok=True)
    logger.add(
        DOSSIER_LOGS / "ui.log",
        level="INFO",
        rotation="1 MB",
        retention=3,
        encoding="utf-8",
    )
    _configure = True


def tracer_predict(texte: str, sentiment: str, latence_ms: float) -> None:
    """Une ligne par analyse unitaire réussie."""
    logger.info("predict | {} | {:.1f} ms | {}", sentiment, latence_ms, extrait(texte))


def tracer_batch(total: int, repartition: dict[str, int], latence_ms: float) -> None:
    """Une ligne par analyse de lot réussie."""
    logger.info("predict_batch | {} avis | {} | {:.1f} ms", total, repartition, latence_ms)


def tracer_erreur(contexte: str, detail: str) -> None:
    """Une ligne par échec d'appel à l'API (connexion, timeout, statut d'erreur)."""
    logger.warning("erreur | {} | {}", contexte, detail)
