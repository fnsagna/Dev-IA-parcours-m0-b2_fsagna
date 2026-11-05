# Escale Atlantique — Sentiment des avis clients

Projet FastIA (brief M0-B2) : classer les avis clients d'une chaîne d'hôtels en **négatif, neutre ou positif**, pour que le service qualité traite les clients mécontents en priorité sans avoir à tout relire.

## Contexte

Le service qualité d'Escale Atlantique reçoit plusieurs centaines d'avis clients par mois et ne parvient plus à tous les lire. Le modèle utilisé, [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment), prédit une note de **1 à 5 étoiles** : la conversion en 3 classes métier (négatif/neutre/positif) est faite par l'API.

## Architecture

```
┌─────────────┐   HTTP :8501    ┌──────────────────┐   HTTP :8000    ┌───────────────────────┐
│  Navigateur │ ───────────────▶│  ui (Streamlit)   │────────────────▶│  api (FastAPI)        │
└─────────────┘                 │  interface de     │  POST /predict  │  /health  /info       │
                                 │  démonstration    │                 │  /predict /logs       │
                                 └──────────────────┘                 └──────────┬────────────┘
                                                                                  │ pipeline Hugging Face
                                                                                  ▼
                                                              cmarkea/distilcamembert-base-sentiment
                                                              (sortie : 1 à 5 étoiles)

Réseau docker compose "fastia" — volumes : ./models (cache du modèle HF), ./logs (logs API)
```

- **`api`** (FastAPI) : charge le modèle une seule fois au démarrage (`lifespan`), expose `/predict` (texte → sentiment), `/health`, `/info`, `/logs` (traçabilité des dernières prédictions).
- **`ui`** (Streamlit) : interface de démonstration, appelle l'API via `API_URL`.
- Le modèle est mis en cache sur le volume `./models` : téléchargé une fois, puis tout fonctionne hors ligne.

## Structure

```
docker-compose.yml     # 2 services (api, ui), réseau interne, volumes, healthcheck
.env.example           # variables d'environnement, à copier en .env
api/
  app/main.py          #   lifespan (chargement du modèle), /health, /info, /predict, /logs
  app/inference.py     #   chargement du pipeline + conversion étoiles -> sentiment
  app/journal.py       #   traçabilité des prédictions (pour /logs)
  app/schemas.py       #   schémas Pydantic
  scripts/evaluer.py   #   évaluation du modèle sur les avis annotés (tâche 6)
  scripts/evaluer.ipynb#   la même évaluation, en notebook
  tests/                #   tests unitaires (conversion, logs, predict, health)
  Dockerfile
ui/                    # service Streamlit
  app.py
  Dockerfile
data/
  avis_annotes.csv     # 30 avis annotés par l'équipe qualité
postman/
  escale_atlantique.postman_collection.json  # cas limites : vide, trop long, emojis, ironie
analyse_erreurs.md     # analyse d'avis mal classés (tâche 7)
models/                # cache du modèle Hugging Face (volume, non versionné)
logs/                  # logs de l'API (volume, non versionné, créé au lancement)
```

## Installation et lancement

```bash
cp .env.example .env
docker compose up --build
```

⏳ Le premier lancement télécharge le modèle (≈ 270 Mo) : compte quelques minutes. Les lancements suivants sont rapides, et tout fonctionne hors ligne.

- API : http://localhost:8000/health et http://localhost:8000/docs
- Interface : http://localhost:8501
- État des services : `docker compose ps` (l'API doit être *healthy*)
- Arrêt : `docker compose down`

### Sans Docker (dépannage)

Si Docker n'est pas disponible, l'API tourne aussi directement avec Python 3.12 (dépendances dans `api/requirements.txt`, torch CPU inclus) :

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Tests

```bash
docker compose exec api pytest -v
# ou, sans Docker :
cd api && pytest -v
```

## Conversion 5 étoiles → 3 classes

Le modèle renvoie une probabilité par étoile (1 à 5). `inference.py` en déduit une **note attendue** = moyenne des 5 étoiles pondérée par leurs probabilités (plus fiable que l'argmax, qui jette l'information quand deux étoiles sont proches : ex. 3★ à 0.44 et 4★ à 0.46 ne tranchent qu'à 0.02 d'écart).

Cette note est ensuite convertie en classe métier avec des **seuils volontairement asymétriques** :

| Note attendue | Classe |
|---|---|
| `< 2.8` | négatif |
| `2.8` à `3.6` | neutre |
| `≥ 3.6` | positif |

Des seuils symétriques autour de 3 vaudraient 2.5 et 3.5. Le choix métier retenu les décale vers le haut : un avis mécontent classé « neutre » reste sans réponse, alors qu'un avis neutre classé « négatif » ne coûte que deux minutes de relecture en plus — mieux vaut pencher vers le négatif.

### Justification chiffrée (tâche 6)

Comparaison sur les 30 avis annotés (`api/scripts/evaluer.py`), seuils actuels (2.8/3.6) vs seuils symétriques (2.5/3.5) :

| | Bien classés | Avis négatifs ratés |
|---|---|---|
| **Seuils actuels (2.8 / 3.6)** | 26/30 (86.7 %) | 2/12 |
| **Seuils symétriques (2.5 / 3.5)** | 25/30 (83.3 %) | 2/12 |

Les deux ratent les 2 mêmes avis négatifs (voir [analyse des erreurs](#analyse-des-erreurs-tâche-7)) : décaler le seuil bas de 2.5 à 2.8 n'aggrave donc pas le risque sur le négatif. En revanche les seuils symétriques font basculer à tort un avis neutre en positif, d'où leur moins bon score global. Les seuils asymétriques du code sont donc confirmés sur ce jeu de test.

Reproduire l'évaluation (API déjà lancée) :

```bash
python api/scripts/evaluer.py
```

## Analyse des erreurs (tâche 7)

Sur les 30 avis, 4 sont mal classés. Le détail et les hypothèses (ironie, négation, vocabulaire hôtelier mal connu) sont dans [`analyse_erreurs.md`](analyse_erreurs.md).

En résumé : les erreurs surviennent quand **le texte ne contient aucun mot de surface négatif** malgré un sens négatif — ironie (« Bravo pour l'insonorisation… ») ou double négation (« Pas une seule fausse note »). Le modèle, basé sur la polarité lexicale, ne capte pas ce second degré.

## ⭐ Bonus

- **`POST /predict/batch`** : analyse un lot d'avis (`{"avis": ["texte1", "texte2", ...]}`, 1 à 200 avis), renvoie le détail par avis et la **répartition des sentiments** sur le lot.
- **Interface** : un uploader de CSV sous le formulaire d'analyse unitaire (colonne `texte` attendue, sinon la 1ʳᵉ colonne est utilisée). Envoie le lot à `/predict/batch`, affiche un **graphique en barres de la répartition** des sentiments et un tableau des résultats.
- **Collection Postman** ([`postman/escale_atlantique.postman_collection.json`](postman/escale_atlantique.postman_collection.json)) : cas limites sur `/predict` et `/predict/batch` — texte vide, espaces seuls, texte > 2000 caractères, emojis, ironie, double négation, lot vide, avis invalide dans un lot.

## Les avis annotés

`data/avis_annotes.csv` contient 30 avis clients annotés à la main par l'équipe qualité. Ils sont censés être anonymisés… vérifie quand même.

| Colonne | Contenu |
|---|---|
| `id` | identifiant de l'avis |
| `hotel` | `City Hotel` ou `Resort Hotel` |
| `source` | Booking, Google, TripAdvisor ou Questionnaire (fin de séjour) |
| `texte` | l'avis, tel que rédigé par le client |
| `sentiment_attendu` | l'annotation de l'équipe qualité |

Consigne d'annotation donnée à l'équipe qualité :

- **négatif** : l'insatisfaction domine, ou le client signale un incident qui appelle une réponse ;
- **neutre** : avis factuel, ou mitigé sans que l'un des deux côtés domine ;
- **positif** : la satisfaction domine, même avec un petit bémol.
