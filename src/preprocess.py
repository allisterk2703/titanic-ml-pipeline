import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="data/titanic.csv"):
    df = pd.read_csv(filepath)
    
    # Remplissage des valeurs manquantes
    df.fillna({"Age": df["Age"].median(), "Embarked": "S"}, inplace=True)

    # Encodage des variables catégoriques
    df.replace({"Sex": {"male": 0, "female": 1}}, inplace=True)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Sélection des features et de la cible
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
    X = df[features]
    y = df["Survived"]

    # Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Données préparées avec succès !")