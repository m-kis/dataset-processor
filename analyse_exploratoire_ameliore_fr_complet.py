
# Importation des bibliothèques nécessaires
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from wordcloud import WordCloud
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans

# Fonction pour charger les données depuis l'entrée utilisateur
def load_data():
    chemin_fichier = input("Veuillez entrer le chemin du fichier CSV: ")
    try:
        donnees = pd.read_csv(chemin_fichier)
        return donnees
    except FileNotFoundError:
        print("Fichier non trouvé. Veuillez réessayer.")
        return None

# Fonction pour effectuer l'analyse préliminaire
def analyze_data(donnees):
    # Analyser les types de caractéristiques
    caracteristiques_numeriques = donnees.select_dtypes(include=['float64', 'int64']).columns
    caracteristiques_temporelles = donnees.select_dtypes(include=['datetime']).columns
    caracteristiques_categoriques = donnees.select_dtypes(include=['object']).columns.difference(caracteristiques_temporelles)

    print("Caractéristiques Numériques:", caracteristiques_numeriques.tolist())
    print("Caractéristiques Temporelles:", caracteristiques_temporelles.tolist())
    print("Caractéristiques Catégoriques:", caracteristiques_categoriques.tolist())

    # Analyse préliminaire des caractéristiques numériques
    for caract in caracteristiques_numeriques:
        sns.histplot(donnees[caract], kde=True, bins=30)
        plt.title(f'Distribution de {caract}')
        plt.show()

    # Analyse préliminaire des caractéristiques catégoriques
    for caract in caracteristiques_categoriques:
        sns.countplot(x=caract, data=donnees)
        plt.title(f'Distribution de {caract}')
        plt.show()

    # Statistiques Descriptives
    print("Statistiques Descriptives:")
    print(donnees.describe())

    # Valeurs Manquantes
    print("\nValeurs Manquantes:")
    print(donnees.isnull().sum())

    # Visualisation des Valeurs Manquantes
    sns.heatmap(donnees.isnull(), cbar=False)
    plt.title("Carte Thermique des Valeurs Manquantes")
    plt.show()

    # Distribution des Caractéristiques Numériques
    for caracteristique in caracteristiques_numeriques:
        plt.figure(figsize=(8, 4))
        sns.histplot(donnees[caracteristique], kde=True, bins=30)
        plt.title(f'Distribution de {caracteristique}')
        plt.xlabel(caracteristique)
        plt.ylabel('Fréquence')
        plt.show()

        # Détection des Valeurs Atypiques
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=donnees[caracteristique])
        plt.title(f'Boxplot de {caracteristique}')
        plt.show()

    # Relations Catégorielles (si applicable)
    for caract_categ in caracteristiques_categoriques:
        sns.countplot(x=caract_categ, data=donnees, palette='viridis')
        plt.title(f'Distribution de {caract_categ}')
        plt.show()

    # Matrice de Corrélation
    matrice_correlation = donnees[caracteristiques_numeriques].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrice_correlation, annot=True, cmap='coolwarm')
    plt.title('Carte Thermique de Corrélation')
    plt.show()

    # Analyse de Texte
    for colonne in caracteristiques_categoriques:
        # Prétraitement (suppression des caractères non désirés, mise en minuscules, etc.)
        donnees[colonne] = donnees[colonne].str.lower().str.replace('[^a-z\\\\s]', '')

        # Nuage de mots
        nuage_mots = WordCloud().generate(' '.join(donnees[colonne].dropna()))
        plt.imshow(nuage_mots, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Nuage de Mots pour {colonne}')
        plt.show()

    # Analyse Temporelle (en supposant le format 'yyyy-mm-dd')
    colonnes_date = donnees.select_dtypes(include=['datetime']).columns
    for colonne in colonnes_date:
        # Supposer une colonne cible pour la décomposition (remplacer par votre choix)
        colonne_cible = 'nom_colonne_cible'  # À remplacer par le nom de la colonne cible

        # Décomposition saisonnière
        decompose_result = seasonal_decompose(donnees[colonne_cible].set_index(colonne), model='additive')
        decompose_result.plot()
        plt.title(f'Décomposition de {colonne_cible} par {colonne}')
        plt.show()

    # Analyse de Cluster
    if caracteristiques_numeriques.size > 1:  # Modifier la condition si nécessaire
        # Sélectionner les caractéristiques pour le clustering
        caract_cluster = donnees[caracteristiques_numeriques]  # Utiliser les colonnes numériques

        # Appliquer K-means
        kmeans = KMeans(n_clusters=3)  # Choisir le nombre de clusters
        donnees['cluster'] = kmeans.fit_predict(caract_cluster)

        # Visualiser les clusters
        sns.scatterplot(x=caract_cluster.columns[0], y=caract_cluster.columns[1], hue='cluster', data=donnees)
        plt.title('Clusters K-means')
        plt.show()

    # Profiling Automatique avec ydata-profiling (anciennement pandas_profiling)
    profil = ProfileReport(donnees, title="Rapport d'Analyse Exploratoire")
    profil.to_file("rapport_ae.html")

# Fonction principale pour exécuter le script
def main():
    data = load_data()
    if data is not None:
        analyze_data(data)
        print("\nAnalyse terminée.")

# Uncomment to execute the script
main()
