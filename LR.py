#gridsearch, valors de c

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
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

    #EVALUACIO
    cm = confusion_matrix(y_test, predictions)
    print("\nMatriu de confusió:")
    print(cm)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    return predictions
    #EVALUACIO




