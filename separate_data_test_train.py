from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    chemin_fichier = input("Veuillez entrer le chemin du fichier CSV: ")
    try:
        donnees = pd.read_csv(chemin_fichier)
        return donnees
    except FileNotFoundError:
        print("Fichier non trouvé. Veuillez réessayer.")
        return None

def select_target_column(donnees):
    print("Colonnes disponibles:", donnees.columns.tolist())
    target_column = input("Veuillez entrer le nom de la colonne cible ou appuyez sur 'Entrée' pour utiliser la dernière colonne: ")
    if not target_column:
        target_column = donnees.columns[-1]
        print(f"Colonne cible sélectionnée: {target_column}")
    elif target_column not in donnees.columns:
        print(f"Colonne {target_column} introuvable. Utilisation de la dernière colonne.")
        target_column = donnees.columns[-1]
    return target_column

def split_data(donnees, target_column):
    X = donnees.drop(columns=[target_column])
    y = donnees[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    print("Les ensembles d'entraînement et de test ont été enregistrés.")

def main():
    donnees = load_data()
    if donnees is not None:
        target_column = select_target_column(donnees)
        X_train, X_test, y_train, y_test = split_data(donnees, target_column)
        save_data(X_train, X_test, y_train, y_test)
        print("\nSéparation des données terminée.")

if __name__ == "__main__":
    main()
