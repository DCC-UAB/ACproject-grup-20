from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluar(y_true, y_pred):
    """
    Calcula y muestra la matriz de confusión y otras métricas de evaluación.
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred))
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Matriz de Confusión')
    plt.colorbar()
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks([0, 1], ['Negativo', 'Positivo'])
    plt.yticks([0, 1], ['Negativo', 'Positivo'])
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="black")
    plt.show()

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo LinearSVC con parámetros predefinidos,
    genera predicciones y evalúa los resultados.
    """
    # Parámetros del modelo
    svm_params = {
        "C": 1.0,           # Parámetro de regularización
        "max_iter": 1000,   # Número máximo de iteraciones
        "random_state": 42  # Semilla para reproducibilidad
    }

    print(f"Entrenando LinearSVC con parámetros: {svm_params}")
    
    # Crear y entrenar el modelo
    model = LinearSVC(**svm_params)
    model.fit(X_train, y_train)
    
    # Generar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    evaluar(y_test, y_pred)
    
    return y_pred
