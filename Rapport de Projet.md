# Rapport de Projet : Prédiction d'Actions de Maintenance

---

## Contexte du projet

Ce projet vise à prédire les actions de maintenance la plus probable à partir de descriptions textuelles de problèmes, extraites de tickets de maintenance d’aéronefs. Le principal défi réside dans l’absence de labels explicites pour l’action à prédire. Pour y répondre, une approche en deux étapes a été adoptée :

1. **Clustering non supervisé** pour regrouper automatiquement les actions similaires.
2. **Classification supervisée** pour apprendre à prédire le cluster d'action à partir du texte du problème.

---

## Données utilisées

Colonnes principales :

- `PROBLEM` : description textuelle du problème rencontré.
- `ACTION` : description textuelle de l’action réalisée.

---

## Prétraitement

- Mise en minuscules
- Suppression des chiffres et ponctuations (hors `&`)
- Suppression des *stop words*
- Vectorisation via **TF-IDF**

---

## Clustering des actions

- Algorithme : `KMeans`
- Sélection du nombre de clusters via :
  - Silhouette Score
  - Méthode du coude

---

## Classification supervisée

- Objectif : prédire `action_cluster` à partir de `PROBLEM`
- Vectorisation : `TF-IDF`
- Modèle : `RandomForestClassifier`
- Séparation entraînement/test pour évaluation

---

## Résultats du modèle

- **Accuracy globale** : `92%`
- **F1-score macro** : `88%`
- **F1-score pondéré** : `92%`

> Très bonnes performances sur les clusters fréquents. Faible rappel sur les clusters rares, ce qui est attendu.

---

## 📌 Conclusion

- L'approche de **pseudo-supervision** par clustering permet de s'affranchir de l'absence de labels.
- Le modèle `RandomForest` est performant pour relier les descriptions de problèmes à des actions typiques.

