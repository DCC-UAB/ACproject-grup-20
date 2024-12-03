#gridsearch, valors de c
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

def evaluar(y_true, y_pred):
    """
    Calcula i mostra la matriu de confusió i altres mètriques d'avaluació.
    """
    # Matriu de confusió
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriu de confusió:")
    print(cm)

    # metriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
    """
    Entrena un model de regressió logística, genera les prediccions,
    i crida la funció 'evaluar' per mostrar els resultats.
    """
    # Definir el model
    C_value = 1.0  # Valor fix de C
    model = LogisticRegression(C=C_value, solver='liblinear', max_iter=5000, penalty='l2')
    
    # Entrenar el model
    model.fit(X_train, y_train)
    
    # Generar prediccions
    predictions = model.predict(X_test)

    # Avaluar resultats
    evaluar(y_test, predictions)

    return predictions
