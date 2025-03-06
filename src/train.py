import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Sauvegarde du modèle
    joblib.dump(model, "models/model.pkl")
    print("Modèle entraîné et sauvegardé !")

if __name__ == "__main__":
    train_model()
