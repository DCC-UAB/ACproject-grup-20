from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluar(y_true, y_pred):
    """
    Calcula y muestra la matriz de confusión y otras métricas de evaluación.
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    
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
    Entrena un modelo Random Forest con parámetros predefinidos,
    genera predicciones y evalúa los resultados.
    """
    # Parámetros del modelo
    n_estimators = 200  # Número de árboles
    max_depth = 20      # Profundidad máxima de los árboles
    random_state = 42   # Estado aleatorio para reproducibilidad

    print(f"Entrenando Random Forest con n_estimators={n_estimators} y max_depth={max_depth}")
    
    # Crear y entrenar el modelo
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Generar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    evaluar(y_test, y_pred)
    
    return y_pred

###Ahora lo que se hara es hacer una funcion donde se entrene con diferentes num de estimadores