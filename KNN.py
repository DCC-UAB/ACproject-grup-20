from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Acabar de revisar implementació cross validation, revisar grafiques
# Trobar millor k (neighbors) per executar i analitzar grafiques --> XValidation

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

    # Gràfica de la matriu de confusió
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Negatiu', 'Positiu'], yticklabels=['Negatiu', 'Positiu'])
    plt.title('Matriu de Confusió')
    plt.xlabel('Prediccions')
    plt.ylabel('Valors Reals')
    plt.show()

def trobar_millor_n_neighbors(X_train, y_train, max_k, start_k):
    """
    Troba el millor valor de n_neighbors mitjançant validació creuada,
    començant des del valor especificat per start_k.

    Paràmetres:
    - X_train: Matriu de característiques d'entrenament.
    - y_train: Etiquetes d'entrenament.
    - max_k: Nombre màxim de veïns a provar.
    - start_k: Valor inicial de n_neighbors (per reprendre la cerca).

    Retorna:
    - millor_k: El valor òptim de n_neighbors.
    - scores: Accuracy per a cada valor de k.
    """
    scores = []
    for k in range(start_k, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        # Validació creuada amb accuracy com a mètrica
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        scores.append(cv_scores.mean())
        print(f"k={k}: Accuracy mitjana={cv_scores.mean():.4f}")
    
    # Seleccionar el millor k
    millor_k = np.argmax(scores) + start_k  # Ajusta l'índex amb start_k
    print(f"\nMillor k trobat: {millor_k} amb accuracy={scores[millor_k - start_k]:.4f}")
    return millor_k, scores


def mostrar_curves_roc_precisio_recall(y_true, y_prob): # Implementada al model de forma temporal per fer testing
    """
    Mostra la corba ROC i la corba de precisió-recall.
    """
    # Corba ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Corba de precisió-recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(14, 6))

    # Plot Corba ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Curva ROC')
    plt.xlabel('Tasa de falsos positius')
    plt.ylabel('Tasa de veritables positius')
    plt.legend(loc='lower right')

    # Plot Corba de precisió-recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.2f}')
    plt.title('Curva de Precisión-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test, max_k=200, start_k=101):
    """
    Troba el millor valor de n_neighbors, entrena un model K-Nearest Neighbors,
    genera les prediccions i crida les funcions per avaluar i mostrar gràfiques.

    Paràmetres:
    - X_train, y_train: Dades d'entrenament.
    - X_test, y_test: Dades de test.
    - max_k: Valor màxim de n_neighbors.
    - start_k: Valor inicial de n_neighbors (per reprendre la cerca).
    """
    # Trobar el millor valor de n_neighbors
    millor_k, _ = trobar_millor_n_neighbors(X_train, y_train, max_k, start_k=start_k)
    print(f"Entrenant amb el millor k={millor_k}...")

    # Definir el model amb el millor k
    model = KNeighborsClassifier(n_neighbors=millor_k)

    # Entrenar el model
    model.fit(X_train, y_train)

    # Generar prediccions
    predictions = model.predict(X_test)

    # Generar probabilitats de la classe positiva (si està disponible)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = predictions  # Si no hi ha probabilitats, utilitzem les prediccions directes

    # Avaluar resultats
    evaluar(y_test, predictions)

    # Mostrar corbes ROC i de precisió-recall
    if hasattr(model, "predict_proba"):
        mostrar_curves_roc_precisio_recall(y_test, y_prob)

    return predictions
