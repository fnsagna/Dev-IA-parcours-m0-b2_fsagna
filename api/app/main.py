"""API d'analyse du sentiment des avis clients — Escale Atlantique (mission FastIA)."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from loguru import logger

from app import journal
from app.inference import load_pipeline, predict_sentiment
from app.schemas import (
    LogsOut,
    ReviewBatchIn,
    ReviewIn,
    SentimentBatchOut,
    SentimentOut,
)

MODEL_NAME = os.getenv("MODEL_NAME", "cmarkea/distilcamembert-base-sentiment")

# Objets partagés par toutes les requêtes (ici : le pipeline de classification)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    journal.configurer()
    # Au 1er lancement, le modèle est téléchargé (≈ 270 Mo) : ça peut prendre quelques minutes
    logger.info("Chargement du modèle {}…", MODEL_NAME)
    state["classifier"] = load_pipeline(MODEL_NAME)
    logger.info("Modèle chargé")
    yield
    state.clear()


app = FastAPI(
    title="Escale Atlantique — Sentiment des avis clients",
    description="Classe un avis client en négatif, neutre ou positif.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "classifier" in state}


@app.get("/info")
def info() -> dict:
    """Quel modèle tourne, et que renvoie l'API ?"""
    return {
        "model_name": MODEL_NAME,
        "sortie_modele": "1 à 5 étoiles",
        "sortie_api": ["négatif", "neutre", "positif"],
    }


@app.post("/predict", response_model=SentimentOut)
def predict(review: ReviewIn) -> SentimentOut:
    """Analyse le sentiment d'un avis client."""
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    resultat = predict_sentiment(classifier, review.text)
    journal.tracer(review.text, resultat["sentiment"], resultat["latence_ms"])
    return SentimentOut(**resultat)


@app.post("/predict/batch", response_model=SentimentBatchOut)
def predict_batch(batch: ReviewBatchIn) -> SentimentBatchOut:
    """Analyse un lot d'avis clients (ex. import CSV depuis l'interface)."""
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    repartition = {"négatif": 0, "neutre": 0, "positif": 0}
    resultats = []
    for texte in batch.avis:
        resultat = predict_sentiment(classifier, texte)
        journal.tracer(texte, resultat["sentiment"], resultat["latence_ms"])
        repartition[resultat["sentiment"]] += 1
        resultats.append({"texte": texte, **resultat})

    return SentimentBatchOut(total=len(resultats), resultats=resultats, repartition=repartition)


@app.get("/logs", response_model=LogsOut)
def logs(limite: int = Query(20, ge=1, le=journal.TAMPON_MAX)) -> LogsOut:
    """Les dernières prédictions, pour affichage dans l'interface qualité."""
    entrees = journal.dernieres(limite)
    return LogsOut(total=len(entrees), entrees=entrees)
