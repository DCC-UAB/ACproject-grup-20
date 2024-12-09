#podem treure directament els tres elements amb altre idioma???
#accuracy/precision o quin metrica a assolir i a quan
#stemming: no aporta info fent lemm(LR: acc 89,00 stemming, acc 88,98 sense stemming)
#evaluacio respecte % en train fa falta??
#com implementar validation
#word embedding (relacions semantiques(sig) i sintactiques)

#BOOSTRAP, BAGGING, BOOSTING, CROSS VALIDATION

#Fer final: funcio que compari amb grafiques la forma mes optima de cada model

#NEGATIVE COMMENT: 0, POSITIVE COMMENT: 1
import time
import re
import pandas as pd
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import LR
import SVM
import RF
import KNN
import NB

# Selecció model: KNN, LR, NB, RF, SVM
MODEL_CHOICE = 'RF' 
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

# Inicialitzar el lematitzador
lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    """
    Funció per lematitzar.
    """
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

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
    X_valid = pd.read_csv(f'{data_path}Valid.csv')
    X_test = pd.read_csv(f'{data_path}Test.csv')

    # Preprocessament
    X_train = preprocess_pipeline(X_train, 'text')
    X_valid = preprocess_pipeline(X_valid, 'text')
    X_test = preprocess_pipeline(X_test, 'text')

    #X_train = X_train.head(2000)
    #X_valid = X_valid.head(1000)
    #X_test = X_test.head(1000)

    y_train = X_train['label']
    y_valid = X_valid['label']
    y_test = X_test['label']

    X_train = X_train[['processed_text']]
    X_valid = X_valid[['processed_text']]
    X_test = X_test[['processed_text']]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

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
    # Carregar i processar les dades
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)
    processar_time = time.time()
    print('Temps trigat a processar les dades :', processar_time - start_time)

    # Convertir a matrius numèriques
    X_train_matrix, X_valid_matrix, X_test_matrix, vectorizer = convert_to_numeric_matrices(X_train, X_valid, X_test)
    
    #obtenir model
    if MODEL_CHOICE not in model_modules:
        raise ValueError(f"Model no reconegut: {MODEL_CHOICE}")
    model_module = model_modules[MODEL_CHOICE]
    print('model utilitzat:', MODEL_CHOICE)
    
    #entrenar i predecir
    y_pred = getattr(model_module, "entrena_prediu_i_evalua")(X_train_matrix, y_train, X_test_matrix, y_test)
    #y_pred = getattr(model_module, "entrena_prediu_i_evalua")(X_train_matrix, y_train, X_test_matrix, y_test,max_k=200)
    entrenaripredir_time = time.time()
    print('temps entrenament', entrenaripredir_time - processar_time)

if __name__ == "__main__":
    main()
