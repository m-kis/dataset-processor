
# Importation des bibliothèques nécessaires
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from wordcloud import WordCloud
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans

# Fonction pour charger les données depuis l'entrée utilisateur
def load_data(csv_path):
    try:
        donnees = pd.read_csv(csv_path)
        return donnees
    except FileNotFoundError:
        print("Fichier non trouvé. Veuillez réessayer.")
        return None

# Fonction principale pour exécuter le script d'analyse
def main_analyse(csv_path):
    donnees = load_data(csv_path)
    if donnees is not None:
        analyze_data(donnees)
        prepare_and_save_data(donnees)
        print("\nAnalyse et préparation terminées.")
        
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
    profil.to_file("M-KIS-analyse-rapport.html")

# Fonction principale pour exécuter le script
def main():
    data = load_data()
    if data is not None:
        analyze_data(data)
        print("\nAnalyse terminée.")


# On attaque la préaparation des datas en vue de l'exploitation 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def handle_missing_values(data):
    # Imputer for numerical features
    num_imputer = SimpleImputer(strategy="mean")
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_features] = num_imputer.fit_transform(data[numerical_features])

    # Imputer for categorical features
    cat_imputer = SimpleImputer(strategy="most_frequent")
    categorical_features = data.select_dtypes(include=['object']).columns
    
    # Print to verify the categorical features
    print("Caractéristiques Catégoriques:", categorical_features.tolist())

    # Check if there are any categorical features before imputing
    if len(categorical_features) > 0:
        data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

    return data


def encode_categorical_features(data):
    categorical_features = data.select_dtypes(include=['object']).columns
    one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_features])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_features))
    data = data.drop(columns=categorical_features)
    data = pd.concat([data, one_hot_encoded_df], axis=1)

    return data

def scale_numerical_features(data):
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

def prepare_data(data):
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = scale_numerical_features(data)

    return data
def prepare_and_save_data(data):
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = scale_numerical_features(data)
    
    # Enregistrement des données préparées dans un fichier CSV
    chemin_enregistrement = "donnees_preparees.csv" 
    data.to_csv(chemin_enregistrement, index=False)
    print(f"Données préparées enregistrées dans {chemin_enregistrement}.")

    return data

if __name__ == "__main__":
    donnees = load_data()
    if donnees is not None:
        analyze_data(donnees)
        prepare_and_save_data(donnees)
        print("\nAnalyse et préparation terminées.")
