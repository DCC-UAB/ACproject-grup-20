#podem treure directament els tres elements amb altre idioma???
#NEGATIVE COMMENT: 0, POSITIVE COMMENT: 1

import time
import re
import pandas as pd
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import MiniLR
#import SVM
#import RF
#import KNN
#import NB

# Selecció model: KNN, LR, NB, RF, SVM
MODEL_CHOICE = 'MiniLR' 
model_modules = {
    "MiniLR": MiniLR
}

# path carpeta
DATA_PATH = 'C:/Users/marti/OneDrive/Escriptori/datasetsMini_AC/'  
#DATA_PATH = 
#DATA_PATH = 

# Normalització del text
def normalize_text(text):   #REVISAR FUNCIO
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

# Inicialitzar el lematitzador i el stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Funció per aplicar només lematització
def apply_lemmatize(text):
    """
    Funció per aplicar lematització al text.
    """
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

# Funció per aplicar només stemming
def apply_stem(text):
    """
    Funció per aplicar stemming al text.
    """
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed)

# Pipeline de preprocessament que aplica les dues tècniques
def preprocess_pipeline(data, column_name):
    """
    Funció que aplica normalització, lematització i després stemming al dataset.
    """
    # Normalitzar el text abans de processar
    text_normalitzat = data[column_name].apply(normalize_text)
    data['normalized_text'] = text_normalitzat

    # Aplicar lematització
    text_lemmatitzat =  data['normalized_text'].apply(apply_lemmatize)
    data['lemmatized_text'] = text_lemmatitzat

    # Aplicar stemming
    text_stemmatitzat = data['lemmatized_text'].apply(apply_stem)
    data['stemmatized_text'] = text_stemmatitzat

    '''
    # Aplicar stemming
    text_stemmatitzat = data['normalized_text'].apply(apply_stem)
    data['stemmatized_text'] = text_stemmatitzat
    '''
    return data

#carregar i preprocessar dades
def load_and_preprocess_data(data_path):
    """
    Càrrega i preprocessament de les dades.
    """
    # Carregar el dataset Train, Valid, Test
    X_train = pd.read_csv(f'{data_path}MiniTrain.csv')
    X_valid = pd.read_csv(f'{data_path}MiniValid.csv')
    X_test = pd.read_csv(f'{data_path}MiniTest.csv')

    print('\nDOcument train original:\n',X_train)

    # Preprocessament
    X_train = preprocess_pipeline(X_train, 'text')
    X_valid = preprocess_pipeline(X_valid, 'text')
    X_test = preprocess_pipeline(X_test, 'text')

    print('\ntrain normalitzat:\n', X_train['normalized_text'])
    print('\ntrain lemmatitzat:\n', X_train['lemmatized_text'])
    print('\ntrain stemmatized:\n', X_train['stemmatized_text'])

    y_train = X_train['label']
    y_valid = X_valid['label']
    y_test = X_test['label']

    X_train = X_train[['stemmatized_text']]
    X_valid = X_valid[['stemmatized_text']]
    X_test = X_test[['stemmatized_text']]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def convert_to_numeric_matrices(X_train, X_valid, X_test):
    """
    Converteix els textos preprocessats en matrius numèriques utilitzant TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_matrix = vectorizer.fit_transform(X_train['stemmatized_text'])
    X_valid_matrix = vectorizer.transform(X_valid['stemmatized_text'])
    X_test_matrix = vectorizer.transform(X_test['stemmatized_text'])

    print('\nmatriu train vectoritzada (matriu esccasa: vector(paraula), pes):\n', X_train_matrix)

    return X_train_matrix, X_valid_matrix, X_test_matrix, vectorizer

def main():
    start_time = time.time()

    # Carregar i processar les dades
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)
    processar_time = time.time()
    print('temps preocessar', processar_time - start_time)

    # Convertir a matrius numèriques
    X_train_matrix, X_valid_matrix, X_test_matrix, vectorizer = convert_to_numeric_matrices(X_train, X_valid, X_test)
    
    #obtenir model
    if MODEL_CHOICE not in model_modules:
        raise ValueError(f"Model no reconegut: {MODEL_CHOICE}")
    model_module = model_modules[MODEL_CHOICE]
    
    #entrenar i predir
    y_pred = getattr(model_module, "entrena_prediu_i_evalua")(X_train_matrix, y_train, X_test_matrix, y_test)
    entrenaripredir_time = time.time()
    print('temps entrenament', entrenaripredir_time - processar_time)

if __name__ == "__main__":
    main()
