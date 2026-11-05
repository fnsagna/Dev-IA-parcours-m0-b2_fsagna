"""Evalue l'API de sentiment sur les avis annotes : matrice de confusion 3x3
et comparaison de deux conversions note -> classe metier.

Usage (API deja lancee sur 127.0.0.1:8000) :
    python scripts/evaluer.py
"""

import csv
import sys
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "data" / "avis_annotes.csv")

CLASSES = ("négatif", "neutre", "positif")
ETOILES = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")

# Conversion actuelle de l'API (cf. app/inference.py) : seuils asymetriques
SEUIL_NEG_ACTUEL, SEUIL_POS_ACTUEL = 2.8, 3.6

# Deuxieme conversion a tester : seuils symetriques autour de 3, ecartee par l'equipe
SEUIL_NEG_ALT, SEUIL_POS_ALT = 2.5, 3.5


def note_attendue(scores: dict) -> float:
    return sum((i + 1) * scores[e] for i, e in enumerate(ETOILES))


def convertir(note: float, seuil_neg: float, seuil_pos: float) -> str:
    if note < seuil_neg:
        return "négatif"
    if note < seuil_pos:
        return "neutre"
    return "positif"


def matrice_confusion(lignes):
    """lignes: liste de (vrai, predit). Retourne la matrice {vrai: {predit: compte}}."""
    m = {v: {p: 0 for p in CLASSES} for v in CLASSES}
    for vrai, predit in lignes:
        m[vrai][predit] += 1
    return m


def afficher_matrice(titre, m):
    print(f"\n{titre}")
    entete = "vrai\\prédit".ljust(12) + "".join(c.ljust(10) for c in CLASSES)
    print(entete)
    for v in CLASSES:
        ligne = v.ljust(12) + "".join(str(m[v][p]).ljust(10) for p in CLASSES)
        print(ligne)


def main():
    with open(CSV_PATH, encoding="utf-8") as f:
        avis = list(csv.DictReader(f))

    resultats_api = []  # (vrai, predit_actuel)
    resultats_alt = []  # (vrai, predit_alt)

    for row in avis:
        texte = row["texte"]
        vrai = row["sentiment_attendu"]
        resp = requests.post(API_URL, json={"text": texte}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        predit_actuel = data["sentiment"]
        note = note_attendue(data["scores_etoiles"])
        predit_alt = convertir(note, SEUIL_NEG_ALT, SEUIL_POS_ALT)

        resultats_api.append((vrai, predit_actuel))
        resultats_alt.append((vrai, predit_alt))

    n = len(avis)

    # --- Methode 1 : conversion actuelle de l'API (seuils 2.8 / 3.6) ---
    bons_api = sum(1 for v, p in resultats_api if v == p)
    m_api = matrice_confusion(resultats_api)
    afficher_matrice(f"Matrice de confusion — conversion ACTUELLE (seuils {SEUIL_NEG_ACTUEL}/{SEUIL_POS_ACTUEL})", m_api)
    print(f"Bien classés : {bons_api}/{n} ({bons_api/n:.1%})")

    rates_api = sum(1 for v, p in resultats_api if v == "négatif" and p != "négatif")
    total_neg = sum(1 for v, _ in resultats_api if v == "négatif")
    print(f"Avis négatifs ratés : {rates_api}/{total_neg}")

    # --- Methode 2 : conversion alternative (seuils symetriques 2.5 / 3.5) ---
    bons_alt = sum(1 for v, p in resultats_alt if v == p)
    m_alt = matrice_confusion(resultats_alt)
    afficher_matrice(f"Matrice de confusion — conversion ALTERNATIVE (seuils {SEUIL_NEG_ALT}/{SEUIL_POS_ALT})", m_alt)
    print(f"Bien classés : {bons_alt}/{n} ({bons_alt/n:.1%})")

    rates_alt = sum(1 for v, p in resultats_alt if v == "négatif" and p != "négatif")
    print(f"Avis négatifs ratés : {rates_alt}/{total_neg}")

    # --- Comparaison finale ---
    print("\n--- Comparaison ---")
    print(f"Conversion actuelle   : {bons_api}/{n} bons classements, {rates_api} négatifs ratés")
    print(f"Conversion alternative: {bons_alt}/{n} bons classements, {rates_alt} négatifs ratés")

    if rates_api < rates_alt:
        print("=> La conversion ACTUELLE fait moins d'erreurs graves (négatifs ratés).")
    elif rates_alt < rates_api:
        print("=> La conversion ALTERNATIVE fait moins d'erreurs graves (négatifs ratés).")
    else:
        print("=> Les deux conversions font autant d'erreurs graves (négatifs ratés).")


if __name__ == "__main__":
    main()
