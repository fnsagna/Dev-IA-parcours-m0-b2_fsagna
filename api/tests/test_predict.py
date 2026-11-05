"""Route /predict : contrat de sortie, entrées refusées, avis annotés."""

import pytest

from app.inference import predict_sentiment
from app.main import state

ETOILES = {"1 star", "2 stars", "3 stars", "4 stars", "5 stars"}

# Avis recopiés de data/avis_annotes.csv plutôt que lus depuis le fichier : le contexte
# de build de l'api est ./api, donc data/ n'est pas dans l'image et un test qui lirait
# le CSV échouerait sous docker compose exec api pytest.
AVIS_SANS_AMBIGUITE = [
    ("Séjour parfait ! Piscine magnifique, personnel aux petits soins et petit-déjeuner copieux.", "positif"),
    ("Club enfants génial, nos deux garçons ne voulaient plus partir ! Équipe d'animation au top.", "positif"),
    ("Chambre sale à notre arrivée, cheveux dans la douche et serviettes tachées.", "négatif"),
    ("Punaises de lit dans la chambre 108. Nous avons dû partir en pleine nuit.", "négatif"),
    ("Hôtel correct pour une nuit d'étape. Chambre petite mais propre, rien d'exceptionnel.", "neutre"),
    ("Chambre spacieuse et lumineuse, mais le wifi était très lent. Bon rapport qualité-prix malgré tout.", "neutre"),
]


def test_predict_respecte_son_contrat(client):
    reponse = client.post("/predict", json={"text": "Chambre sale, serviettes tachées."})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["sentiment"] in ("négatif", "neutre", "positif")
    assert set(corps["scores_etoiles"]) == ETOILES
    assert corps["latence_ms"] > 0


def test_les_cinq_scores_somment_a_un(client):
    reponse = client.post("/predict", json={"text": "Hôtel classique de centre-ville."})
    assert sum(reponse.json()["scores_etoiles"].values()) == pytest.approx(1.0, abs=1e-3)


def test_les_scores_sortent_dans_l_ordre_des_etoiles(client):
    """Le pipeline les renvoie triés par score : l'API doit les remettre dans l'ordre."""
    reponse = client.post("/predict", json={"text": "Séjour parfait, on reviendra !"})
    assert list(reponse.json()["scores_etoiles"]) == [
        "1 star", "2 stars", "3 stars", "4 stars", "5 stars",
    ]


@pytest.mark.parametrize(
    "texte, cas",
    [
        ("", "texte vide"),
        ("     ", "espaces seuls"),
        ("a" * 2001, "trop long"),
    ],
)
def test_entree_invalide_renvoie_422(client, texte, cas):
    assert client.post("/predict", json={"text": texte}).status_code == 422, cas


def test_champ_text_absent_renvoie_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_au_dela_de_512_tokens_le_texte_est_tronque(client):
    """La limite Pydantic est en caractères, celle du modèle en tokens : elle ne la couvre pas."""
    # Appel direct : 2 000 caractères ne suffisent pas toujours à dépasser 512 tokens
    resultat = predict_sentiment(state["classifier"], "Chambre sale et bruyante. " * 300)
    assert resultat["sentiment"] in ("négatif", "neutre", "positif")


@pytest.mark.parametrize("texte, attendu", AVIS_SANS_AMBIGUITE)
def test_avis_annotes_sans_ambiguite(client, texte, attendu):
    assert client.post("/predict", json={"text": texte}).json()["sentiment"] == attendu


def test_predict_batch_respecte_son_contrat(client):
    textes = [texte for texte, _ in AVIS_SANS_AMBIGUITE]
    reponse = client.post("/predict/batch", json={"avis": textes})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["total"] == len(textes)
    assert len(corps["resultats"]) == len(textes)
    assert sum(corps["repartition"].values()) == len(textes)
    assert set(corps["repartition"]) == {"négatif", "neutre", "positif"}
    for ligne, texte in zip(corps["resultats"], textes):
        assert ligne["texte"] == texte
        assert ligne["sentiment"] in ("négatif", "neutre", "positif")


def test_predict_batch_compte_juste_la_repartition(client):
    textes = [texte for texte, _ in AVIS_SANS_AMBIGUITE]
    corps = client.post("/predict/batch", json={"avis": textes}).json()
    attendue = {"négatif": 0, "neutre": 0, "positif": 0}
    for _, attendu in AVIS_SANS_AMBIGUITE:
        attendue[attendu] += 1
    assert corps["repartition"] == attendue


def test_predict_batch_lot_vide_renvoie_422(client):
    assert client.post("/predict/batch", json={"avis": []}).status_code == 422


def test_predict_batch_rejette_un_avis_invalide_du_lot(client):
    """Un seul avis vide dans le lot suffit à faire échouer toute la requête."""
    assert client.post("/predict/batch", json={"avis": ["Très bien", ""]}).status_code == 422


def test_predict_batch_champ_avis_absent_renvoie_422(client):
    assert client.post("/predict/batch", json={}).status_code == 422
