from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import pandas as pd

# Fonction pour charger les données
def load_data():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    
    return X_train, y_train, X_test, y_test

# Fonction pour déterminer le type de problème
def determine_problem_type(y):
    unique_values = y.nunique().values[0]
    if unique_values == 2:
        return 'classification'
    else:
        return 'regression'

# Fonction pour obtenir le modèle sélectionné
def get_model(problem_type):
    models = {
        'classification': [
            ('Logistic Regression', LogisticRegression),
            ('Random Forest Classifier', RandomForestClassifier),
            ('Support Vector Classifier', SVC)
        ],
        'regression': [
            ('Linear Regression', LinearRegression),
            ('Random Forest Regressor', RandomForestRegressor),
            ('Support Vector Regressor', SVR)
        ]
    }
    
    print(f"Je detecte un problème de type {problem_type}.")
    print("Modèles disponibles:")
    for i, (name, _) in enumerate(models[problem_type]):
        print(f"{i + 1}. {name}")

    choice = input("Choisissez un modèle en tapant le numéro qui lui corresponds: ")
    if choice == '':
        model_name, model_instance = models[problem_type][0]  # Default model
    else:
        model_name, model_instance = models[problem_type][int(choice) - 1]

    print(f"Using {model_name}.")
    return model_name, model_instance() # Return both name and instance

# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    if problem_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
    else:
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

# Fonction principale
def main():
    X_train, y_train, X_test, y_test = load_data()
    problem_type = determine_problem_type(y_train)
    
    while True:
        model_name, model_instance = get_model(problem_type)
        model_instance.fit(X_train, y_train.values.ravel())
        evaluate_model(model_instance, X_test, y_test, problem_type)
        
        save_choice = input(f"Voulez-vous enregistrer le  modèle {model_name}? (o/n): ")
        if save_choice.lower() == 'o':
            save_model(model_instance, model_name)  # Enregistre le modèle
            display_example_code(model_name, problem_type)  # Affiche un exemple de code
        
        retry = input("Voulez-vous essayer un autre modèle ? (o/n): ")
        if retry.lower() != 'o':
            break

# Fonction pour enregistrer le modèle
def save_model(model, model_name):
    filename = f'{model_name}_model.sav'
    joblib.dump(model, filename)
    print(f"Modèle enregistré sous le nom {filename}.")

# Fonction pour afficher un exemple de code
def display_example_code(model_name, problem_type):
    example_code = f"""# Exemple de code pour charger et interroger le modèle {model_name} 

from sklearn.externals import joblib

# charger le modele
filename = '{model_name}_model.sav'
loaded_model = joblib.load(filename)

# Exemple de données d'entrée
input_data = [...]  # Remplacez par vos datas

# Interroger le modèle
prediction = loaded_model.predict([input_data])

print('Prediction:', prediction)"""

    print("\nVous pouvez utiliser le code suivant pour charger et interroger le modèle:")
    print(example_code)

# Fonction principale
def main():
    X_train, y_train, X_test, y_test = load_data()
    problem_type = determine_problem_type(y_train)
    
    while True:
        model_name, model_instance = get_model(problem_type)
        model_instance.fit(X_train, y_train.values.ravel())
        evaluate_model(model_instance, X_test, y_test, problem_type)

        save_model(model_instance, model_name)  # Enregistre le modèle
        display_example_code(model_name, problem_type)  # Affiche un exemple de code
        
        retry = input("Voulez-vous essayer un autre modèle? (o/n): ")
        if retry.lower() != 'o':
            break

# Déclencheur pour exécuter le script
if __name__ == "__main__":
    main()
