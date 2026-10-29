# Prédiction du churn de clients télécom

Projet de Machine Learning qui prédit quels clients d'un opérateur télécom risquent de se désabonner (churn), pour permettre des actions de fidélisation ciblées.

## Données
- **IBM Telco Customer Churn** (dataset public, aussi disponible sur Kaggle) : 7 043 clients, 21 variables (contrat, ancienneté, services, facturation, moyen de paiement…).
- Taux de churn : **26,6 %** (classes déséquilibrées).

## Démarche
1. Nettoyage : conversion de `TotalCharges` en numérique, suppression des 11 valeurs vides.
2. Analyse exploratoire : le churn atteint **42,7 %** pour les contrats mensuels, contre **2,8 %** pour les contrats de 2 ans.
3. Préparation : pipeline scikit-learn (standardisation des variables numériques, one-hot encoding des catégorielles), découpage 80/20 stratifié.
4. Modélisation : comparaison de 3 modèles, avec pondération des classes.
5. Interprétation : facteurs qui augmentent ou réduisent le risque de churn.

## Résultats (jeu de test, 1 407 clients)
| Modèle | ROC-AUC | Rappel | Précision | F1 |
|---|---|---|---|---|
| Gradient Boosting | 0,839 | 0,529 | 0,641 | 0,580 |
| Régression logistique | 0,835 | 0,797 | 0,490 | 0,607 |
| Random Forest | 0,833 | 0,786 | 0,514 | 0,622 |

La régression logistique pondérée détecte **80 % des clients qui partent**, ce qui est l'objectif prioritaire pour une campagne de rétention.

## Lancer le projet
```bash
pip install -r requirements.txt
python churn_prediction.py
```
Sorties : `resultats_modeles.csv`, `courbes_roc.png`, `facteurs_churn.png`.

## Pistes d'amélioration
- Ajuster le seuil de décision selon le coût d'une action de rétention.
- Tester XGBoost et l'optimisation des hyperparamètres.
- Exposer le modèle via une API FastAPI.
