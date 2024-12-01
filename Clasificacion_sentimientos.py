import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

#nltk.download('punkt')  # Descargar tokenizador de palabras
#nltk.download('stopwords')  # Descargar lista de palabras vacías
#nltk.download('wordnet')  # Descargar WordNet para lematización
#nltk.download('all')  # Descargar todos los recursos de NLTK, si es necesario

# 1. Normalización del texto
def normalize_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'@\w+', '', text)  # Eliminar menciones (@usuario)
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚ]', '', text)  # Eliminar caracteres especiales
    words = word_tokenize(text)  # Tokenizar palabras
    stop_words = set(stopwords.words('english'))  # Cargar stopwords
    filtered_words = [word for word in words if word not in stop_words]  # Eliminar stopwords
    return " ".join(filtered_words)

# 2. Procesamiento de texto y análisis léxico
def process_text(dataset, column_name):
    dataset['cleaned_text'] = dataset[column_name].apply(normalize_text)  # Normalizar texto
    dataset['tokens'] = dataset['cleaned_text'].apply(word_tokenize)  # Tokenizar
    lemmatizer = WordNetLemmatizer()
    dataset['lemmatized_tokens'] = dataset['tokens'].apply(
        lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
    )
    return dataset

# 3. Detección de polaridad
def classify_sentiment(data):
    positive_words = {"excellent", "exciting", "great", "funny", "amazing", "beautiful", "incredible", "charming"}
    negative_words = {"boring", "weak", "terrible", "confusing", "disappointing", "predictable", "dull", "awful"}

    def assign_sentiment(tokens):
        if any(word in positive_words for word in tokens):
            return 1  # Positivo
        elif any(word in negative_words for word in tokens):
            return 0  # Negativo
        return 0.5  # Neutral o desconocido

    data['sentiment'] = data['lemmatized_tokens'].apply(assign_sentiment)
    return data

# 4. Análisis de polaridad y visualización
def visualize_sentiment(data):
    sentiment_counts = data['sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['red', 'blue', 'green'])
    plt.title('Distribución de Polaridad')
    plt.xlabel('Sentimiento (0: Negativo, 1: Positivo, 0.5: Neutral)')
    plt.ylabel('Frecuencia')
    plt.show()

# 5. Pipeline de procesamiento
def sentiment_analysis_pipeline(filepath, column_name):
    # Cargar dataset
    data = pd.read_csv(filepath)
    # Procesar texto
    data = process_text(data, column_name)
    # Clasificar sentimiento
    data = classify_sentiment(data)
    # Visualizar resultados
    visualize_sentiment(data)
    return data

# Ejecución del pipeline
processed_train_data = sentiment_analysis_pipeline(
    'C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/Train.csv',
    'text'
)

# Guardar resultados procesados
#processed_train_data.to_csv('processed_train_data.csv', index=False)
