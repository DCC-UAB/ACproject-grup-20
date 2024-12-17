import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV  # Para obtener probabilidades
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.inspection import DecisionBoundaryDisplay

# Directorio para guardar los gráficos
EVALUATION_DIR = "C:/Users/twitc/ACproject-grup-20/SVM_evaluation"

def plot_probability_distribution(y_true, y_proba):
    """
    Plota la distribució de les probabilitats predites per cada classe.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(y_proba[y_true == 1], color="blue", kde=True, label="Classe Positiva (1)", stat="density")
    sns.histplot(y_proba[y_true == 0], color="red", kde=True, label="Classe Negativa (0)", stat="density")
    plt.title("Distribució de les Probabilitats Preditades")
    plt.xlabel("Probabilitat Preditada")
    plt.ylabel("Densitat")
    plt.legend()
    prob_dist_path = os.path.join(EVALUATION_DIR, "probability_distribution.png")
    plt.savefig(prob_dist_path)
    print(f"Distribució de probabilitats guardada en {prob_dist_path}")
    plt.close()

def plot_decision_boundary(model, X, y):
    """
    Plota la frontera de decisió per un model entrenat en dades 2D.
    """
    plt.figure(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(
        model, X, response_method="predict", cmap="coolwarm", alpha=0.5
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    plt.title("Frontera de Decisió")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.show()

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

    # Cálculo y visualización de la curva de precisión-recall
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    print(f"AUC (Precisión-Recall): {pr_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva de Precisión-Recall')
    plt.legend(loc='lower left')
    pr_path = os.path.join(EVALUATION_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    print(f"Curva de precisión-recall guardada en {pr_path}")
    plt.close()

    # Cridar la funció per mostrar la distribució de probabilitats
    plot_probability_distribution(y_true, y_proba)


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
    
    # Mostrar la frontera de decisió (si les dades són 2D)
    if X_train.shape[1] == 2: #
        plot_decision_boundary(model, X_train.toarray(), y_train)

    return predictions
