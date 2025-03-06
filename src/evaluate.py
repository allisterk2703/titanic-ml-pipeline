import joblib
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

def evaluate_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Chargement du modèle
    model = joblib.load("models/model.pkl")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Calcul de la précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
