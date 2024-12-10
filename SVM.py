import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV  # Para obtener probabilidades
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Directorio para guardar los gráficos
EVALUATION_DIR = "C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/"
os.makedirs(EVALUATION_DIR, exist_ok=True)

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

    # Visualizar y guardar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=set(y_true), yticklabels=set(y_true))
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    confusion_path = os.path.join(EVALUATION_DIR, "matriz_confusion.png")
    plt.savefig(confusion_path)
    print(f"Matriz de confusión guardada en {confusion_path}")
    plt.close()

    # Cálculo y visualización de la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    print(f"\nAUC (Area Under the Curve): {auc_score:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    roc_path = os.path.join(EVALUATION_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    print(f"Curva ROC guardada en {roc_path}")
    plt.close()

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

    return predictions
