import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import glob


def main():
    # Lê todos os arquivos que começam com 'training_data_'
    files = glob.glob('data/training_data_*.csv')
    if not files:
        print("Nenhum arquivo de treino encontrado.")
        exit()

    # Junta todos os CSVs em um único dataframe
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Verifica dados
    print("Amostras carregadas:", len(df))
    print("Exemplo dos dados:")
    print(df.head())

    # Confere se tem dados faltando
    if df.isnull().any().any():
        print("Aviso: dados faltando detectados, removendo...")
        df = df.dropna()

    X = df[['angle']]
    y = df['label']

    # 2. Divide treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Treina modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Avalia modelo
    y_pred = model.predict(X_test)
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("Acurácia no teste:", model.score(X_test, y_test))

    # 5. Salva o modelo
    joblib.dump(model, 'angle_quality_model.pkl')
    print("\nModelo salvo em 'angle_quality_model.pkl'")


if __name__ == "__main__":
    main()
