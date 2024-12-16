from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score,train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
EVALUATION_DIR = "C:/Users/twitc/ACproject-grup-20/KNN_evaluation"

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

def comparar_accuracy_per_percentatge(X_train, y_train, X_test, y_test, millor_k):
    """
    Compara l'accuracy del model KNN per diferents percentatges de les dades d'entrenament,
    tant pel conjunt d'entrenament com pel conjunt de test.
    Es seleccionen aleatòriament percentatges del 5%, 10%, 30%, 50%, 70%, 90%, 100% de les dades d'entrenament.
    """
    percentatges = [5, 10, 30, 50, 70, 90, 100]
    
    # Llistes per guardar els resultats de l'accuracy
    train_accuracies = []
    test_accuracies = []

    # Iterar sobre cada percentatge
    for percentatge in percentatges:
        # Calcular el percentatge de mostres
        train_size = percentatge / 100.0  # Convertir el percentatge a una fracció

        # Evitar que train_size sigui 1.0, ja que això vol dir que agafem tot el conjunt d'entrenament
        if train_size == 1.0:
            X_train_sub, y_train_sub = X_train, y_train
        else:
            # Seleccionar aleatòriament una part del conjunt d'entrenament
            X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)

        # Definir el model KNN amb el millor valor de k trobat
        model = KNeighborsClassifier(n_neighbors=millor_k)
        
        # Entrenar el model amb el subset seleccionat
        model.fit(X_train_sub, y_train_sub)

        # Fer les prediccions pel conjunt d'entrenament
        y_train_pred = model.predict(X_train_sub)
        # Fer les prediccions pel conjunt de test
        y_test_pred = model.predict(X_test)

        # Calcular l'accuracy pel conjunt d'entrenament
        train_accuracy = accuracy_score(y_train_sub, y_train_pred)
        # Calcular l'accuracy pel conjunt de test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Afegir els resultats a les llistes
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Crear un DataFrame amb els resultats
    results_df = pd.DataFrame({
        'Percentatge': percentatges,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies
    })

    # Mostrar els resultats
    print("\nResultats d'accuracy per diferents percentatges de dades d'entrenament:")
    print(results_df)

    # Generar gràfiques de comparació
    plt.figure(figsize=(10, 6))
    
    # Crear gràfic per a l'accuracy del conjunt d'entrenament i test
    plt.plot(percentatges, train_accuracies, marker='o', linestyle='-', label='Train Accuracy')
    plt.plot(percentatges, test_accuracies, marker='o', linestyle='-', label='Test Accuracy')
    
    # Afegir títol i etiquetes
    plt.title('Comparació de l\'Accuracy per percentatges de dades d\'entrenament (KNN)')
    plt.xlabel('Percentatge del conjunt d\'entrenament')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Modificar les etiquetes de l'eix X per mostrar percentatges
    plt.xticks(percentatges, [f'{x}%' for x in percentatges])

    # Guardar el gràfic a la carpeta d'evaluació
    plt.tight_layout()
    plt.savefig(f"{EVALUATION_DIR}/comparacio_accuracy_percentatge_KNN.png")
    plt.close()

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test): #, max_k=200, start_k=1
    """
    Entrena un model KNN i crida 'comparar_accuracy_per_percentatge' després d'entrenar.
    """
    # # Trobar el millor valor de n_neighbors
    # millor_k, _ = trobar_millor_n_neighbors(X_train, y_train, max_k, start_k=start_k)
    # print(f"Entrenant amb el millor k={millor_k}...") # millor_k=195

    # Assegurar que els arrays són editables
    X_train = np.array(X_train.todense()) if hasattr(X_train, "todense") else np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test.todense()) if hasattr(X_test, "todense") else np.array(X_test)
    y_test = np.array(y_test)

    # Definir el model amb el millor k
    model = KNeighborsClassifier(n_neighbors=195)

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

    # Cridar la comparació d'accuracy per percentatges
    comparar_accuracy_per_percentatge(X_train, y_train, X_test, y_test, 195)

    return predictions
