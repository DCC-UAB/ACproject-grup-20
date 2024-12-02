import re
import pandas as pd
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

import LR
import SVM
import RF
import KNN
import NB

# Selecció del model al principi del codi
MODEL_CHOICE = 1

# path carpeta
DATA_PATH = 'C:/Users/marti/OneDrive/Escriptori/datasets_AC/'  

# Normalització del text
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

# Inicialitzar el lematitzador i el stemmer una sola vegada
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize_and_stem(text):
    """
    Funció per lematitzar i aplicar stemming.
    """
    tokens = word_tokenize(text)
    # Lematitzar
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Stemming
    lemmatized = [stemmer.stem(word) for word in lemmatized]
    return " ".join(lemmatized)

# Pipeline de preprocessament
def preprocess_pipeline(data, column_name):
    """
    Funció que aplica la normalització i la lematització/stemming al dataset.
    """
    data['cleaned_text'] = data[column_name].apply(normalize_text)
    data['processed_text'] = data['cleaned_text'].apply(lemmatize_and_stem)
    return data


def load_and_preprocess_data(data_path):
    """
    Càrrega i preprocessament de les dades.
    """
    # Carregar el dataset Train, Valid, Test
    X_train = pd.read_csv(f'{data_path}Train.csv')
    X_valid = pd.read_csv(f'{data_path}Valid.csv')
    X_test = pd.read_csv(f'{data_path}Test.csv')

    # Preprocessament
    X_train = preprocess_pipeline(X_train, 'text')
    X_valid = preprocess_pipeline(X_valid, 'text')
    X_test = preprocess_pipeline(X_test, 'text')

    y_train = X_train['label']
    y_valid = X_valid['label']
    y_test = X_test['label']

    X_train = X_train[['processed_text']]
    X_valid = X_valid[['processed_text']]
    X_test = X_test[['processed_text']]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def main():
    # Carregar i processar les dades
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)
    # Aquí es cridarien les funcions per entrenar i avaluar el model seleccionat

if __name__ == "__main__":
    main()
