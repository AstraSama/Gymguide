import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import glob


def main():
    # Lê todos os arquivos de estatísticas
    files = glob.glob('data/training_data_stats*.csv')
    if not files:
        print("Nenhum arquivo de treino encontrado.")
        exit()

    columns = ['mean', 'std', 'min', 'max', 'amplitude', 'reps', 'label']
    df = pd.concat([pd.read_csv(f, names=columns) for f in files], ignore_index=True)

    print("Amostras carregadas:", len(df))
    print("Exemplo dos dados:")
    print(df.head())

    if df.isnull().any().any():
        print("Aviso: dados faltando detectados, removendo...")
        df = df.dropna()

    # Define as features e rótulo
    feature_cols = ['mean', 'std', 'min', 'max', 'amplitude', 'reps']
    X = df[feature_cols]
    y = df['label']

    # Divide treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Treina modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avalia
    y_pred = model.predict(X_test)
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("Acurácia no teste:", model.score(X_test, y_test))

    # Salva modelo
    joblib.dump(model, 'pushup_quality_model.pkl')
    print("\nModelo salvo como 'pushup_quality_model.pkl'")


if __name__ == "__main__":
    main()
