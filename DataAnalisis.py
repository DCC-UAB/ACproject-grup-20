import pandas as pd
import os
import re
import langid
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#path evaluacio models en general
EVALUATION_DIR = "C:/Users/marti/OneDrive/Escriptori/ProjecteAC/ACproject-grup-20/PREANALISIS_evaluation"

# Funció per detectar idiomes
def detectar_idiomes(data, column_name):
    detected_languages = data[column_name].apply(lambda x: langid.classify(x)[0])
    
    # Mostrar les línies i els índexs dels textos no en anglès
    non_english_rows = data[detected_languages != 'en']
    if not non_english_rows.empty:
        print("\nLínies que no estan en anglès (índex del CSV):")
        print(non_english_rows.index.tolist())  # Mostra els índexs de les línies no angleses
    
    print('noenglish:', non_english_rows)
    
    return detected_languages.value_counts()

# Funció per mostrar la distribució de les etiquetes
def mostrar_distribucio_labels(data, column_name):
    distribucio = data[column_name].value_counts()
    return distribucio

# Funció per mostrar la distribució de les etiquetes com a gràfica
def mostrar_grafica_distribucio_labels(distribucio, title):
    distribucio.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    #guardar grafica
    file_path = os.path.join(EVALUATION_DIR, str(title))
    plt.savefig(file_path)

#normalkitzar text
def normalize_text(text):   
    """
    Funció per normalitzar el text: convertir a minúscules, eliminar mencions,
    URLs, caràcters especials i eliminar paraules d'aturada.
    """
    text = text.lower()
    text = re.sub(r'@\w+', '', text)  # Eliminar mencions (@usuari)
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚ]', '', text)  # Eliminar caràcters especials
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

lemmatizer = WordNetLemmatizer()

#lemmatitzar
def lemmatize(text):
    """
    Funció per lematitzar.
    """
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

# Pipeline de preprocessament (DECIDIR: STEMMING/LEMMATIZE/LEMMATIZEpos)
def preprocess_pipeline(data, column_name):
    """
    Funció que aplica la normalització i la lematització al dataset.
    """
    data['processed_text'] = data['text'].apply(lambda x: lemmatize(normalize_text(x)))
    return data

# Funció per crear matriu TF-IDF i guardar la gràfica
# Funció per calcular les paraules més rellevants TF-IDF
def calcular_top_paraules_tfidf(data, column_name, top_n, title):
    """
    Calcula les paraules amb més pes TF-IDF i les mostra en una gràfica.
    """
    # Vectoritzar el text
    vectorizer = TfidfVectorizer(max_features=5000)  # max_features definit per TF-IDF
    X_tfidf = vectorizer.fit_transform(data[column_name])

    # Obtenir les paraules i pesos
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_weights = np.asarray(X_tfidf.mean(axis=0)).flatten()

    # Seleccionar les 10 paraules amb més pes
    sorted_indices = np.argsort(tfidf_weights)[-top_n:]
    top_words = feature_names[sorted_indices]
    top_weights = tfidf_weights[sorted_indices]

    # Guardar la gràfica
    plt.figure(figsize=(10, 6))
    plt.barh(top_words, top_weights, color='skyblue')
    plt.xlabel('Pes TF-IDF')
    plt.title(f"Top {top_n} Paraules amb més Pes TF-IDF ({title})")
    plt.tight_layout()

    file_path = os.path.join(EVALUATION_DIR, f"Top_{top_n}_TFIDF_{title}.png")
    plt.savefig(file_path)

# Funció principal
def main():
    test_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Test.csv')
    train_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Train.csv')
    valid_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Valid.csv')
    
    ######### DETECCIO IDIOMES
    print("Detectant idiomes en Test.csv...")
    test_languages = detectar_idiomes(test_data, "text")
    print("Detectant idiomes en Train.csv...")
    train_languages = detectar_idiomes(train_data, "text")
    print("Detectant idiomes en Valid.csv...")
    valid_languages = detectar_idiomes(valid_data, "text")

    # Mostrar la distribució d'idiomes
    print("\nIdiomes en Test.csv:", test_languages)
    print("Idiomes en Train.csv:", train_languages)
    print("Idiomes en Valid.csv:", valid_languages)
    
    
    ######### DISTTRIBUCIO POS I NEG 
    print("\nDistribució de les etiquetes en Test.csv:")
    test_label_dist = mostrar_distribucio_labels(test_data, 'label')
    print(test_label_dist)
    mostrar_grafica_distribucio_labels(test_label_dist, "Distribució de les etiquetes en Test")
    
    print("\nDistribució de les etiquetes en Train.csv:")
    train_label_dist = mostrar_distribucio_labels(train_data, 'label')
    print(train_label_dist)
    mostrar_grafica_distribucio_labels(train_label_dist, "Distribució de les etiquetes en Train")
    
    print("\nDistribució de les etiquetes en Valid.csv:")
    valid_label_dist = mostrar_distribucio_labels(valid_data, 'label')
    print(valid_label_dist)
    mostrar_grafica_distribucio_labels(valid_label_dist, "Distribució de les etiquetes en Valid")
    
    
    ######### 15 PARAULES MES IMPORTANTS TFIDF
    train_data = preprocess_pipeline(train_data, 'text')

    # Calcular i guardar les 10 paraules més rellevants
    calcular_top_paraules_tfidf(train_data, 'processed_text', top_n=10, title='Train')

# Cridar la funció principal
if __name__ == "__main__":
    main()
