import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
# Directorio para guardar visualizaciones
EVALUATION_DIR = "C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/"
#os.makedirs(EVALUATION_DIR, exist_ok=True)

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
    Entrena un modelo Random Forest optimizado, genera las predicciones y llama a 'evaluar'.
    """
    modelo= RandomForestClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=200)

    modelo_entrenado =modelo.fit(X_train, y_train)

    # Predicciones y probabilidades
    y_pred = modelo_entrenado.predict(X_test)
    y_proba = modelo_entrenado.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva

    # Evaluar resultados
    evaluar(y_test, y_pred, y_proba)
    return y_pred    


###
#Lo que se hizo aqui abajo es para encontrar los mejores parametros para este modelo donde demos encontrado lo siguiente:
# Temps trigat a processar les dades : 118.41145944595337
# model utilitzat: RF
# Iniciando Grid Search...
# Fitting 5 folds for each of 36 candidates, totalling 180 fits

# Mejores parámetros encontrados:
# {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}

# Mejor precisión obtenida:
# 0.8437

# Matriz de Confusión:
# [[2039  456]
#  [ 320 2185]]

# Accuracy: 0.8448
# Precision: 0.8458
# Recall: 0.8448
# F1 Score: 0.8447
# Matriz de confusión guardada en C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/matriz_confusion.png

# AUC (Area Under the Curve): 0.9230
# Curva ROC guardada en C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/roc_curve.png
# temps entrenament 961.2376260757446




#FUNCION PARA ENCONTRAR MEJORES PARAM
# def buscar_mejores_parametros_rf(X_train, y_train):
#     """
#     Busca los mejores parámetros para Random Forest utilizando GridSearchCV.
#     """
#     # Definir el diccionario de parámetros
#     param_grid = {
#         'n_estimators': [50, 100, 200],  # Número de árboles
#         'max_depth': [10, 20, 30],  # Profundidad máxima del árbol
#         'min_samples_split': [2, 5],  # Número mínimo de muestras para dividir un nodo
#         'min_samples_leaf': [1, 2]  # Número mínimo de muestras en una hoja
#     }

#     # Crear el modelo base
#     model = RandomForestClassifier(random_state=42)

#     # Configurar GridSearchCV
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                                scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

#     # Entrenar el modelo
#     print("Iniciando Grid Search...")
#     grid_search.fit(X_train, y_train)

#     # Mostrar los mejores parámetros
#     print("\nMejores parámetros encontrados:")
#     print(grid_search.best_params_)

#     print("\nMejor precisión obtenida:")
#     print(f"{grid_search.best_score_:.4f}")

#     return grid_search.best_estimator_

# def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
#     """
#     Entrena un modelo Random Forest optimizado, genera las predicciones y llama a 'evaluar'.
#     """
#     # Buscar los mejores parámetros y entrenar el modelo
#     mejor_modelo = buscar_mejores_parametros_rf(X_train, y_train)

#     # Predicciones y probabilidades
#     y_pred = mejor_modelo.predict(X_test)
#     y_proba = mejor_modelo.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva

#     # Evaluar resultados
#     evaluar(y_test, y_pred, y_proba)
#     return y_pred

