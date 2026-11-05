"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import os

import httpx
import pandas as pd
import streamlit as st

# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Couleur et icône associées à chaque sentiment renvoyé par l'API
SENTIMENT_STYLE = {
    "négatif": ("🔴", "error"),
    "neutre": ("🟠", "warning"),
    "positif": ("🟢", "success"),
}

st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨")
st.title("🏨 Sentiment des avis clients")
st.caption("Escale Atlantique · outil du service qualité")

text = st.text_area(
    "Avis client",
    height=160,
    placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
)

if st.button("Analyser", type="primary"):

    try:
        with st.spinner("Analyse en cours…"):
            response = httpx.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
    except httpx.ConnectError:
        st.error("Le service d'analyse est indisponible.")
        st.stop()
    except httpx.TimeoutException:
        st.error("Le service met trop de temps à répondre.")
        st.stop()

    if response.status_code == 422:
        st.warning("L'avis doit contenir entre 1 et 2 000 caractères.")
        st.stop()
    elif response.status_code != 200:
        st.error("Une erreur est survenue, réessayez.")
        st.stop()

    resultat = response.json()
    sentiment = resultat["sentiment"]
    icone, niveau = SENTIMENT_STYLE[sentiment]
    getattr(st, niveau)(f"{icone} Sentiment : **{sentiment}**")

    with st.expander("Voir le détail par nombre d'étoiles"):
        scores = dict(sorted(resultat["scores_etoiles"].items()))
        st.bar_chart(scores)
        st.caption(f"Temps de réponse de l'API : {resultat['latence_ms']:.0f} ms")

st.divider()
st.subheader("Analyser un lot d'avis (CSV)")
st.caption("Le fichier doit contenir une colonne `texte` (sinon la 1ʳᵉ colonne est utilisée).")

fichier = st.file_uploader("Fichier CSV", type="csv")

if fichier is not None and st.button("Analyser le lot", type="primary"):
    donnees = pd.read_csv(fichier)
    colonne = "texte" if "texte" in donnees.columns else donnees.columns[0]
    textes = donnees[colonne].dropna().astype(str).str.strip()
    textes = [t for t in textes if t]

    if not textes:
        st.warning("Aucun avis exploitable dans ce fichier.")
        st.stop()

    try:
        with st.spinner(f"Analyse de {len(textes)} avis…"):
            response = httpx.post(f"{API_URL}/predict/batch", json={"avis": textes}, timeout=60)
    except httpx.ConnectError:
        st.error("Le service d'analyse est indisponible.")
        st.stop()
    except httpx.TimeoutException:
        st.error("Le service met trop de temps à répondre.")
        st.stop()

    if response.status_code == 422:
        st.warning("Le fichier contient un avis vide ou trop long (max 2 000 caractères), ou plus de 200 lignes.")
        st.stop()
    elif response.status_code != 200:
        st.error("Une erreur est survenue, réessayez.")
        st.stop()

    lot = response.json()

    st.write(f"**{lot['total']} avis analysés**")
    st.bar_chart(lot["repartition"])

    tableau = pd.DataFrame(lot["resultats"])[["texte", "sentiment"]]
    st.dataframe(tableau, use_container_width=True)
