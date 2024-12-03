#gridsearch, valors de c

from sklearn.linear_model import LogisticRegression

def entrena_i_prediu(X_train, y_train, X_test):
    """
    Entrena un model de regressió logística i genera les prediccions.
    """
    # Definir el model
    C_value = 1.0  # Valor fix de C
    model = LogisticRegression(C=C_value, solver='liblinear', max_iter=5000, penalty='l2')
    
    # Entrenar el model
    print("Entrenant regressió logística...")
    model.fit(X_train, y_train)
    
    # Generar prediccions
    print("Generant prediccions amb regressió logística...")
    predictions = model.predict(X_test)
    return predictions
