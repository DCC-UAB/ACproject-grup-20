#FER: veure dataset % de cada un. per veure si millor accuracy o precision.
#FER: accuracy amb max features de tfidf

#FER: word embedding (relacions semantiques(sig) i sintactiques)
#FER: BOOSTRAP, BAGGING, BOOSTING, CROSS VALIDATION



#NEGATIVE COMMENT: 0, POSITIVE COMMENT: 1
#'text', 'label'
#40.000, 5.000, 5.000

from langdetect import detect
import time
import re
import pandas as pd
import sys
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import LR
import SVM
import RF
import KNN
import NB

# Selecció model: KNN, LR, NB, RF, SVM
MODEL_CHOICE = 'LR' 
model_modules = {
    "LR": LR,
    "SVM": SVM,
    "KNN": KNN,
    "RF": RF,
    "NB": NB
}

# path carpeta
DATA_PATH = 'C:/Users/marti/OneDrive/Escriptori/datasets_AC/'  
#DATA_PATH = 'C:/Users/twitc/OneDrive/Desktop/Dataset/'
#DATA_PATH = "C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/"

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

# Inicialitzar el lematitzador i stemmatitzador
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Funció per convertir etiquetes de POS al format de WordNet
def get_wordnet_pos(treebank_tag):
    """
    Converteix etiquetes POS de Penn Treebank al format WordNet.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  

# Funció per aplicar lematització amb POS
def lemmatizePOS(text):
    """
    Aplica lematització al text utilitzant POS tags per millorar la precisió.
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)  # Obtenim les etiquetes de tipus de paraula
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    return " ".join(lemmatized)

#lemmatitzar
def lemmatize(text):
    """
    Funció per lematitzar.
    """
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

#stemmatitzar
def stemmatize(text):
    """
    Funció per stemmatitzar.
    """
    tokens = word_tokenize(text)
    stemmatized = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmatized)

# Pipeline de preprocessament
def preprocess_pipeline(data, column_name):
    """
    Funció que aplica la normalització i la lematització al dataset.
    """
    data['processed_text'] = data['text'].apply(lambda x: lemmatize(normalize_text(x)))
    return data

#carregar i preprocessar dades
def load_and_preprocess_data(data_path):
    """
    Càrrega i preprocessament de les dades.
    """
    # Carregar el dataset Train, Valid, Test
    X_train = pd.read_csv(f'{data_path}Train.csv')
    X_valid = pd.read_csv(f'{data_path}ValidFAKE.csv')
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

# Funcions de desar i carregar dades amb pickle
def save_data(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def convert_to_numeric_matrices(X_train, X_valid, X_test):
    """
    Converteix els textos preprocessats en matrius numèriques utilitzant TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_matrix = vectorizer.fit_transform(X_train['processed_text'])
    X_valid_matrix = vectorizer.transform(X_valid['processed_text'])
    X_test_matrix = vectorizer.transform(X_test['processed_text'])
    return X_train_matrix, X_valid_matrix, X_test_matrix, vectorizer

def main():
    start_time = time.time()

    # Comprovar si els fitxers pickle de les matrius TF-IDF i les etiquetes ja existeixen
    if os.path.exists('X_train_matrix.pkl'):
        # Carregar les matrius TF-IDF i les etiquetes des de pickle
        X_train_matrix = load_data('X_train_matrix.pkl')
        X_valid_matrix = load_data('X_valid_matrix.pkl')
        X_test_matrix = load_data('X_test_matrix.pkl')
        y_train = load_data('y_train.pkl')
        y_valid = load_data('y_valid.pkl')
        y_test = load_data('y_test.pkl')

        print('DADES CARREGADES')
    
    else:
        # Carregar i processar les dades
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)

        # Convertir a matrius numèriques
        X_train_matrix, X_valid_matrix, X_test_matrix, _ = convert_to_numeric_matrices(X_train, X_valid, X_test)

        # Guardar les matrius TF-IDF i les etiquetes
        save_data(X_train_matrix, 'X_train_matrix.pkl')
        save_data(X_valid_matrix, 'X_valid_matrix.pkl')
        save_data(X_test_matrix, 'X_test_matrix.pkl')
        save_data(y_train, 'y_train.pkl')
        save_data(y_valid, 'y_valid.pkl')
        save_data(y_test, 'y_test.pkl')

        print('DADES PROCESSADESS i DESCARGADES')

    processar_time = time.time()
    print('Temps trigat a processar les dades:', processar_time - start_time)
    
    # Obtenir el model
    if MODEL_CHOICE not in model_modules:
        raise ValueError(f"Model no reconegut: {MODEL_CHOICE}")
    model_module = model_modules[MODEL_CHOICE]
    print('Model utilitzat:', MODEL_CHOICE)
    
    # Entrenar i predir
    y_pred = getattr(model_module, "entrena_prediu_i_evaluaMaxIter")(X_train_matrix, y_train, X_test_matrix, y_test)
    entrenaripredir_time = time.time()
    print('Temps entrenament:', entrenaripredir_time - processar_time)

if __name__ == "__main__":
    main()
