from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Entrenament del model de Logistic Regression i avaluació.
    """
    # Inicialitzar el model
    model = LogisticRegression(max_iter=1000)

    # Entrenar el model
    model.fit(X_train, y_train)

    # Fer prediccions
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)








    return model