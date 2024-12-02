import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK
#nltk.download('punkt', quiet=True)
#nltk.download('stopwords', quiet=True)
#nltk.download('wordnet', quiet=True)

# 1. Normalización del texto
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)  # Eliminar menciones (@usuario)
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚ]', '', text)  # Eliminar caracteres especiales
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# 2. Procesar texto y lematizar
def process_text(dataset, column_name):
    dataset['cleaned_text'] = dataset[column_name].apply(normalize_text)
    lemmatizer = WordNetLemmatizer()
    dataset['lemmatized_tokens'] = dataset['cleaned_text'].apply(
        lambda text: [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    )
    return dataset

# 3. Pipeline de procesamiento
def preprocess_pipeline(filepath, column_name):
    data = pd.read_csv(filepath)
    data = process_text(data, column_name)
    return data

# Bloque principal para evitar ejecuciones duplicadas
if __name__ == "__main__":
    # 4. Cargar y procesar los datos
    data = preprocess_pipeline(
        'C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/Train.csv',
        'text'
    )

    # Usar la columna procesada 'cleaned_text' y las etiquetas 'label'
    X = data['cleaned_text']  
    y = data['label']  # Etiquetas (0 = negativo, 1 = positivo)

    # 5. Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Vectorizar el texto con TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 7. Entrenar el modelo LinearSVC
    start = time.time()
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)
    print(f'Training time: {time.time() - start:.2f} seconds')

    # 8. Hacer predicciones
    y_pred = svm_model.predict(X_test_tfidf)

    # 9. Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # 10. Visualización de la matriz de confusión
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
