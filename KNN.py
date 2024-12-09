from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluar(y_true, y_pred):
    """
    Calcula i mostra la matriu de confusió i altres mètriques d'avaluació.
    """
    # Matriu de confusió
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriu de confusió:")
    print(cm)

    # Mètriques
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
    Entrena un model K-Nearest Neighbors, genera les prediccions,
    i crida la funció 'evaluar' per mostrar els resultats.
    """
    # Definir el model KNN amb un nombre de veïns predeterminat
    n_neighbors = 5  # Valor per defecte de veïns
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Entrenar el model
    model.fit(X_train, y_train)

    # Generar prediccions
    predictions = model.predict(X_test)

    # Avaluar resultats
    evaluar(y_test, predictions)

    return predictions

if __name__ == "__main__":
    print("Aquest script està destinat a ser importat al main.")

"""
Primera execució model KNN, n_neighbors=5
temps preocessar 124.6710159778595

Matriu de confusió:
[[1727  768]
 [ 481 2024]]

Accuracy: 0.7502
Precision: 0.7535
Recall: 0.7502
F1 Score: 0.7493
temps entrenament 22.811246871948242
"""