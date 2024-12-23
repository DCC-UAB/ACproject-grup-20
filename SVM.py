import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve

# Directorio para guardar los gráficos
EVALUATION_DIR = "C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/"

def evaluar(y_true, y_pred, y_proba):
    """
    Calcula y muestra la matriz de confusión, métricas y genera la curva ROC.
    """
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    # Métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.savefig(os.path.join(EVALUATION_DIR, "matriz_confusion.png"))
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    print(f"\nAUC (Area Under the Curve): {auc_score:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Curva ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(os.path.join(EVALUATION_DIR, "roc_curve.png"))
    plt.close()

def graficar_distribucion_probabilidades(probabilidades, y_test):
    """
    Grafica la distribución de probabilidades de predicción.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(probabilidades[y_test == 1], color='green', label='Positivos', kde=True, stat="density", bins=20)
    sns.histplot(probabilidades[y_test == 0], color='red', label='Negativos', kde=True, stat="density", bins=20)
    plt.title('Distribución de Probabilidades de Predicción')
    plt.xlabel('Probabilidad')
    plt.ylabel('Densidad')
    plt.legend()
    plt.savefig(os.path.join(EVALUATION_DIR, "distribucion_probabilidades.png"))
    plt.close()

def graficar_precision_vs_C(X_train, y_train, X_test, y_test):
    """
    Grafica la precisión del modelo en función del hiperparámetro C.
    """
    C_values = [0.01, 0.1, 1, 10, 100]
    accuracies = []

    for C in C_values:
        base_model = LinearSVC(C=C, max_iter=5000)
        model = CalibratedClassifierCV(base_model, cv=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    plt.plot(C_values, accuracies, marker='o')
    plt.title('Precisión del Modelo vs C')
    plt.xlabel('C (Parámetro de Regularización)')
    plt.ylabel('Precisión')
    plt.xscale('log')
    plt.grid()
    plt.savefig(os.path.join(EVALUATION_DIR, "precision_vs_C.png"))
    plt.close()

def graficar_curva_aprendizaje(model, X, y):
    
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Entrenamiento', marker='o')
    plt.plot(train_sizes, test_mean, label='Validación', marker='o')
    plt.title('Curva de Aprendizaje')
    plt.xlabel('Tamaño del Conjunto de Entrenamiento')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(EVALUATION_DIR, "curva_aprendizaje.png"))
    plt.close()

def graficar_curva_precision_recall(y_true, probabilidades):
    """
    Genera y guarda la Curva Precision-Recall con el AUC.
    """
    from sklearn.metrics import precision_recall_curve, auc

    # Calcular precisión, recall y el AUC
    precisions, recalls, _ = precision_recall_curve(y_true, probabilidades)
    auc_pr = auc(recalls, precisions)

    # Generar la gráfica
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='green', label=f'Precision-Recall Curve (AUC = {auc_pr:.2f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Guardar la gráfica
    plt.savefig(os.path.join(EVALUATION_DIR, "curva_precision_recall.png"))
    plt.close()
    print("Curva Precision-Recall guardada como 'curva_precision_recall.png'")



def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo LinearSVC, genera las predicciones y llama a 'evaluar'.
    """
    # Crear el modelo LinearSVC
    base_model = LinearSVC(C=1.0, max_iter=5000, random_state=42)

    # Calibrar el modelo para obtener probabilidades
    model = CalibratedClassifierCV(base_model, cv=5)
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Generar predicciones y probabilidades
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva

    # Evaluar resultados
    evaluar(y_test, predictions, probabilities)

    return predictions, probabilities 

#AMB MILLORS PARAMS
def acc_millors_params(X_train, y_train, X_test, y_test):
    print('fent SVM')
    """
    Entrena un model de SVM amb els millors paràmetres i retorna l'acc.
    """
    model = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
