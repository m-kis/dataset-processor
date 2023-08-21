import analyse_exploratoire_ameliore_fr_complet as analyse_script
import separate_data_test_train as separate_script
import model_train as train_script

def main():
    csv_path = input("Veuillez entrer le chemin complet vers le fichier CSV : ")
    
    print("Étape 1: Analyse Exploratoire des Données")
    print("----------------------------------------")
    # Appel de la fonction d'analyse exploratoire en transmettant le chemin du CSV
    analyse_script.main_analyse(csv_path)
    
    print("\nÉtape 2: Séparation des Données en Ensembles d'Apprentissage et de Test")
    print("---------------------------------------------------------------------")
    # Appel de la fonction de séparation des données
    # Vous pouvez adapter cette partie selon la façon dont la séparation des données est effectuée dans votre script
    separate_script.main()

    print("\nÉtape 3: Formation et Évaluation des Modèles")
    print("-------------------------------------------")
    # Appel de la fonction de formation et d'évaluation du modèle
    # Vous pouvez adapter cette partie selon la façon dont la formation et l'évaluation du modèle sont effectuées dans votre script
    train_script.main()

    print("\nProcessus terminé avec succès!")

if __name__ == "__main__":
    main()
