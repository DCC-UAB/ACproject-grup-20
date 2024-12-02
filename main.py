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
#DATA_PATH = ''
#DATA_PATH = ''


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

# Lematització i Stemming
def lemmatize_and_stem(text):
    """
    Funció per lematitzar i aplicar stemming.
    """
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    tokens = word_tokenize(text)
    # Lematitzar les paraules
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Aplicar sempre stemming
    lemmatized = [stemmer.stem(word) for word in lemmatized]
    
    return " ".join(lemmatized)

# Pipeline de preprocessament
def preprocess_pipeline(filepath, column_name):
    """
    Funció que llegeix el dataset des del fitxer, normalitza el text,
    i aplica la lematització i stemming.
    """
    data = pd.read_csv(filepath)
    #data = data.head(10)

    data['cleaned_text'] = data[column_name].apply(normalize_text)
    data['processed_text'] = data['cleaned_text'].apply(lemmatize_and_stem)
    return data

# Funció per carregar i processar les dades
def load_and_preprocess_data(data_path):
    """
    Càrrega i preprocessament de les dades.
    """
    # Carregar el dataset Train, Valid, Test
    X_train = preprocess_pipeline(f'{data_path}Train.csv', 'text')
    X_valid = preprocess_pipeline(f'{data_path}Valid.csv', 'text')
    X_test = preprocess_pipeline(f'{data_path}Test.csv', 'text')

    y_train = X_train['label']
    y_valid = X_valid['label']
    y_test = X_test['label']
   
    # Eliminar la columna 'label' per les dades de text
    X_train = X_train.drop(columns=['label'])
    X_valid = X_valid.drop(columns=['label'])
    X_test = X_test.drop(columns=['label'])
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def main():
    # Selecciona si vols utilitzar stemming o no
    use_stemming = True  # Canvia a False si no vols usar stemming
    
    # Carregar i processar les dades
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)

    # Aquí es cridarien les funcions per entrenar i avaluar el model seleccionat
    # Per exemple:
    # train_and_evaluate(MODEL_CHOICE, X_train, y_train, X_valid, y_valid, X_test, y_test)
    print("Dades carregades i processades amb èxit.")

if __name__ == "__main__":
    main()
