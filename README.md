
# Analyse Exploratoire Automatique de Données

Ce projet contient un script Python pour effectuer une analyse exploratoire automatisée sur n'importe quel ensemble de données CSV. Le script réalise diverses analyses et visualisations pour comprendre les tendances, les corrélations et la distribution des données.

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

## Comment Utiliser

1. **Installer les Dépendances**:
   Assurez-vous d'avoir installé les bibliothèques nécessaires:
   ```
   pip install pandas seaborn matplotlib ydata-profiling wordcloud statsmodels scikit-learn
   ```

2. **Exécuter le Script**:
   Exécutez le script `analyse_exploratoire_ameliore_fr_complet.py`:
   ```
   python analyse_exploratoire_ameliore_fr_complet.py
   ```

3. **Fournir le Chemin du Fichier CSV**:
   Le script vous demandera le chemin du fichier CSV que vous souhaitez analyser.

4. **Visualiser les Résultats**:
   Le script affichera diverses visualisations et résumés dans la console et générera un rapport HTML.

## Contribution

N'hésitez pas à contribuer à ce projet en soumettant des issues ou des pull requests.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
