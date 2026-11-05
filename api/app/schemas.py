"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# Le modèle accepte 512 tokens, soit ~370 mots : 2 000 caractères restent sous la limite
LONGUEUR_MAX = 2000

# Un lot plus grand ferait tourner le modèle trop longtemps dans une seule requête HTTP
TAILLE_LOT_MAX = 200

# strip_whitespace : un texte fait de seuls espaces devient vide, donc refusé en 422
_TexteAvis = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=LONGUEUR_MAX,
    ),
]


class ReviewIn(BaseModel):
    """Un avis client à analyser."""

    text: _TexteAvis


class ReviewBatchIn(BaseModel):
    """Un lot d'avis clients à analyser (ex. import CSV depuis l'interface)."""

    avis: Annotated[list[_TexteAvis], Field(min_length=1, max_length=TAILLE_LOT_MAX)]


class SentimentOut(BaseModel):
    """Réponse de /predict."""

    sentiment: Literal["négatif", "neutre", "positif"]
    # Scores bruts du modèle, pour la traçabilité : {"1 star": 0.02, ..., "5 stars": 0.71}
    scores_etoiles: dict[str, float]
    latence_ms: float


class SentimentBatchItem(SentimentOut):
    """Une ligne du résultat de /predict/batch : l'avis d'origine, plus sa prédiction."""

    texte: str


class SentimentBatchOut(BaseModel):
    """Réponse de /predict/batch."""

    total: int
    resultats: list[SentimentBatchItem]
    # Répartition des sentiments sur le lot, prête pour un graphique côté interface
    repartition: dict[Literal["négatif", "neutre", "positif"], int]


class LogEntry(BaseModel):
    """Une prédiction journalisée, telle que la renvoie /logs."""

    horodatage: str
    # Pas d'extrait de l'avis : il reste dans logs/api.log, côté serveur
    sentiment: Literal["négatif", "neutre", "positif"]
    latence_ms: float


class LogsOut(BaseModel):
    """Réponse de /logs, les prédictions les plus récentes d'abord."""

    total: int
    entrees: list[LogEntry]
