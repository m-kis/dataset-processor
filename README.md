
# M-KIS AI Preprocessing Dataset

Ce dépôt contient un outil de traitement de données développé par M-KIS pour effectuer diverses tâches d'analyse de données, de prétraitement et d'apprentissage automatique. L'outil se compose de plusieurs scripts conçus pour être modulaires et personnalisables en fonction de vos données et de vos besoins d'analyse.

## Fonctionnalités

- **Analyse des Caractéristiques**: Identifie et affiche les caractéristiques numériques, temporelles et catégoriques.
- **Distribution et Statistiques**: Affiche la distribution des caractéristiques numériques et catégoriques, et fournit des statistiques descriptives.
- **Valeurs Manquantes**: Visualise et résume les valeurs manquantes dans l'ensemble de données.
- **Détection des Valeurs Atypiques**: Utilise des diagrammes à moustaches pour identifier les valeurs atypiques.
- **Corrélation**: Affiche une carte thermique de corrélation pour les caractéristiques numériques.
- **Analyse de Texte**: Crée un nuage de mots pour les caractéristiques textuelles.
- **Analyse Temporelle**: Effectue une décomposition saisonnière pour les caractéristiques temporelles.
- **Clustering**: Utilise K-means pour effectuer une analyse de cluster sur les caractéristiques numériques.
- **Rapport d'Analyse Exploratoire**: Génère un rapport HTML complet à l'aide de ydata-profiling.

### Étape 2 : Séparer les Données

- **Le script `separate_data_test_train.py`**: permet de séparer votre ensemble de données en ensembles d'apprentissage et de test.
- **Ajustez le ratio de séparation et autres** paramètres selon vos besoins dans le script.
3. Les ensembles de données séparés seront enregistrés sous les noms `X_train.csv`, `y_train.csv`, `X_test.csv` et `y_test.csv`.

### Étape 3 : Entraîner et Évaluer les Modèles

- **Le script `model_train.py`** est utilisé pour entraîner et évaluer des modèles d'apprentissage automatique sur votre ensemble de données.
2. Le script déterminera automatiquement le type de problème (classification ou régression) en fonction de votre colonne cible.
3. Vous serez invité à sélectionner un modèle à entraîner et à évaluer.
4. Après l'entraînement et l'évaluation, vous aurez la possibilité d'enregistrer le modèle et d'obtenir un exemple de code pour l'interroger.

## Personnalisation

- Modifiez les scripts selon vos besoins pour adapter le traitement à vos données spécifiques et à vos exigences d'analyse.
- Vous pouvez ajuster les paramètres, ajouter des étapes de prétraitement personnalisées et modifier les options de sélection de modèle.


## Comment Utiliser

1. **Installer les Dépendances**:
   Assurez-vous d'avoir installé les bibliothèques nécessaires:
   ```
   pip install pandas seaborn matplotlib ydata-profiling wordcloud statsmodels scikit-learn
   ```

2. **Exécuter le Script**:
   Exécutez le script `analyse_exploratoire_ameliore_fr_complet.py`:
   ```
   python main_scripts.py
   ```

3. **Fournir le Chemin du Fichier CSV**:
   Le script vous demandera le chemin du fichier CSV que vous souhaitez analyser.

4. **Visualiser les Résultats**:
   Le script affichera diverses visualisations et résumés dans la console et générera un rapport HTML.

## Contribution

N'hésitez pas à contribuer à ce projet en soumettant des issues ou des pull requests.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
