import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Directorio para guardar visualizaciones
EVALUATION_DIR =  "C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/"

def evaluar(y_true, y_pred, y_proba):
    """
    Calcula y muestra la matriz de confusión, métricas y genera la curva ROC.
    """
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
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
    Entrena un modelo Random Forest, genera las predicciones y llama a 'evaluar'.
    """
    # Parámetros del modelo
    n_estimators = 200
    max_depth = 20
    random_state = 42

    print(f"Entrenando Random Forest con n_estimators={n_estimators} y max_depth={max_depth}")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Predicciones y probabilidades
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva

    # Evaluar resultados
    evaluar(y_test, y_pred, y_proba)
    return y_pred

##Ahora lo que se hara es hacer una funcion donde se entrene con diferentes num de estimadores (pendiente x hacer)

# # Definir el diccionario de parámetros
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# # Crear el modelo
# model = RandomForestClassifier(random_state=42)

# # Configurar GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                            scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

# # Entrenar GridSearchCV
# print("Iniciando Grid Search...")
# grid_search.fit(x_train, y_train)

# # Obtener los mejores parámetros y resultados
# print("\nMejores parámetros encontrados:")
# print(grid_search.best_params_)

# print("\nMejor precisión obtenida:")
# print(f"{grid_search.best_score_:.4f}")

# # Evaluar en el conjunto de prueba
# y_pred = grid_search.best_estimator_.predict(X_test)
# print("\nReporte de clasificación en el conjunto de prueba:")
# print(classification_report(y_test, y_pred))