# Analyse des erreurs — Tâche 7

Sur les 30 avis annotés, l'API (conversion actuelle, seuils 2.8/3.6) en classe mal 4. En voici 3, avec une hypothèse pour chacun.

## id=3 — vrai : négatif, prédit : positif (note 4.7/5)

> « Bravo pour l'insonorisation : j'ai pu suivre en direct toute la soirée de mes voisins jusqu'à 3h du matin. Merci pour l'expérience. »

**Hypothèse : ironie.** Le modèle s'appuie sur des mots lexicalement positifs (« Bravo », « Merci », « expérience ») sans capter le second degré. DistilCamemBERT, entraîné sur des avis e-commerce généralistes, n'a probablement pas assez vu ce registre sarcastique pour le désamorcer.

## id=17 — vrai : positif, prédit : négatif (note 2.1/5)

> « Pas une seule fausse note, on ne pouvait pas rêver mieux. »

**Hypothèse : négation qui inverse le sens.** « Pas une seule fausse note » contient une négation + un mot à connotation négative (« fausse note »), que le modèle associe à du négatif, alors que la double négation exprime en réalité un compliment fort — un piège classique en NLP.

## id=22 — vrai : neutre, prédit : positif (note 4.44/5)

> « RAS. »

**Hypothèse : vocabulaire/abréviation hôtelière mal connue.** L'avis est trop court et l'abréviation « RAS » (rien à signaler) n'est probablement pas associée par le modèle à de la neutralité — avec quasi aucun autre signal lexical à exploiter, il retombe sur un score par défaut plutôt positif.

## Point commun

Les deux premières erreurs partagent un trait : **aucun mot de surface négatif** dans le texte (au contraire, du vocabulaire positif pour id=3, une double négation pour id=17). Le modèle, entraîné sur la polarité lexicale de mots isolés, est aveugle à l'ironie et peu fiable sur les négations imbriquées — deux limites connues des modèles de sentiment par mots-clés/embeddings sans raisonnement contextuel profond.
