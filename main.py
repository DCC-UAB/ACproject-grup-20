import time
import re
import sys
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split

import LR
import SVM
import RF
import KNN
import NB

#NEGATIVE COMMENT: 0, POSITIVE COMMENT: 1
#'text', 'label'
#40.000, 5.000, 5.000


#################################################################################
#PARÀMETRES
#################################################################################
# Selecció model: KNN, LR, NB, RF, SVM
MODEL_CHOICE = 'LR' 
model_modules = {
    "LR": LR,
    "SVM": SVM,
    "KNN": KNN,
    "RF": RF,
    "NB": NB
}

#Selecció method: {tfidf, word_embedding}
EMBEDDING_METHOD = 'tfidf'
    
# path carpeta
DATA_PATH = 'C:/Users/marti/OneDrive/Escriptori/datasets_AC/'  
#DATA_PATH = 'C:/Users/twitc/OneDrive/Desktop/Dataset/'
#DATA_PATH = "C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/"

#path model word embedding
WORD_EMBEDDING_MODEL_PATH = 'C:/Users/marti/OneDrive/Escriptori/ProjecteAC/GoogleNews-vectors-negative300.bin'

#path evaluacio models en general
EVALUATION_DIR = "C:/Users/marti/OneDrive/Escriptori/ProjecteAC/ACproject-grup-20/MODELS_evaluation"

#################################################################################
#EXECUTION
#################################################################################
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

'''
#################   LEMMATIZE Part Of Speech -> NO APORTA MILLORA   ##############################
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
##################################################################################################
'''

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

# Pipeline de preprocessament (utilitza STEMMING)
def preprocess_pipeline(data, column_name):
    """
    Funció que aplica la normalització i la lematització al dataset.
    """
    data['processed_text'] = data[column_name].apply(lambda x: lemmatize(normalize_text(x)))
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

# Funció per carregar un model de Word2Vec
def load_word_embedding_model(model_path):
    """
    Carrega un model Word2Vec o GloVe des de la ruta especificada.
    """
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

# Funció per obtenir el vector mitjà d'un text utilitzant Word2Vec
def get_word2vec_vector(text, model):
    """
    Funció per obtenir el vector mitjà d'un text utilitzant Word2Vec.
    """
    tokens = word_tokenize(text)
    valid_tokens = [word for word in tokens if word in model]
    if valid_tokens:
        word_vectors = [model[word] for word in valid_tokens]
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # Vector buit si no trobem paraules en el model

# Funció per convertir els textos en matrius de Word Embedding
def convert_to_word_embeddings(X_train, X_valid, X_test, model):
    """
    Converteix els textos en matrius de Word Embedding.
    """
    X_train_matrix = np.array([get_word2vec_vector(text, model) for text in X_train['processed_text']])
    X_valid_matrix = np.array([get_word2vec_vector(text, model) for text in X_valid['processed_text']])
    X_test_matrix = np.array([get_word2vec_vector(text, model) for text in X_test['processed_text']])
    return X_train_matrix, X_valid_matrix, X_test_matrix

# Funció per convertir els textos en matrius TF-IDF
def convert_to_tfidf(X_train, X_valid, X_test, mida_matriu):
    """
    Converteix els textos en matrius numèriques utilitzant TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features = mida_matriu)
    X_train_matrix = vectorizer.fit_transform(X_train['processed_text'])
    X_valid_matrix = vectorizer.transform(X_valid['processed_text'])
    X_test_matrix = vectorizer.transform(X_test['processed_text'])
    return X_train_matrix, X_valid_matrix, X_test_matrix

# Funció per carregar i processar dades segons el mètode d'embedding: SHA UTLITZAT LEMMATIZE
def carregar_i_processar_dades(DATA_PATH, EMBEDDING_METHOD, WORD_EMBEDDING_MODEL_PATH):
    if EMBEDDING_METHOD == 'word_embedding':
        if os.path.exists('X_train_word_embedding.pkl'):
            # Carregar les matrius de Word Embedding i les etiquetes des de pickle
            X_train_matrix = load_data('X_train_word_embedding.pkl')
            X_valid_matrix = load_data('X_valid_word_embedding.pkl')
            X_test_matrix = load_data('X_test_word_embedding.pkl')
            y_train = load_data('y_train.pkl')
            y_valid = load_data('y_valid.pkl')
            y_test = load_data('y_test.pkl')
            print('DADES CARREGADES')
        else:
            # Carregar i processar les dades
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)

            # Carregar model de Word Embedding
            model = load_word_embedding_model(WORD_EMBEDDING_MODEL_PATH)

            # Convertir els textos en matrius de Word Embedding
            X_train_matrix, X_valid_matrix, X_test_matrix = convert_to_word_embeddings(X_train, X_valid, X_test, model)

            # Guardar les matrius de Word Embedding
            save_data(X_train_matrix, 'X_train_word_embedding.pkl')
            save_data(X_valid_matrix, 'X_valid_word_embedding.pkl')
            save_data(X_test_matrix, 'X_test_word_embedding.pkl')
            save_data(y_train, 'y_train.pkl')
            save_data(y_valid, 'y_valid.pkl')
            save_data(y_test, 'y_test.pkl')

            print('DADES PROCESSADES I DESCARGADES')

    # TF-IDF
    elif EMBEDDING_METHOD == 'tfidf':
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

            # Convertir a matrius TF-IDF
            X_train_matrix, X_valid_matrix, X_test_matrix = convert_to_tfidf(X_train, X_valid, X_test, 5000)

            # Guardar les matrius TF-IDF i les etiquetes
            save_data(X_train_matrix, 'X_train_matrix.pkl')
            save_data(X_valid_matrix, 'X_valid_matrix.pkl')
            save_data(X_test_matrix, 'X_test_matrix.pkl')
            save_data(y_train, 'y_train.pkl')
            save_data(y_valid, 'y_valid.pkl')
            save_data(y_test, 'y_test.pkl')

            print('DADES PROCESSADES I DESCARGADES')

    else:
        raise ValueError('SELECCIONAR CORRECTAMENT EL MODEL')

    return X_train_matrix, X_valid_matrix, X_test_matrix, y_train, y_valid, y_test

#robustesa diferents max features tfidf
def evaluar_robustesa_amb_tfidf(DATA_PATH, model_modules, max_features_range):
    """
    Avalua tots elss models amb diferents valors de max_features (TF-IDF) i genera una gràfica.
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)
    
    results = {model_name: [] for model_name in model_modules.keys()}  # Per emmagatzemar els resultats

    for mida_matriu in max_features_range:
        # Crear matrius TF-IDF
        X_train_matrix, X_valid_matrix, X_test_matrix = convert_to_tfidf(X_train, X_valid, X_test, mida_matriu)

        for model_name, model_module in model_modules.items():
            accuracy = getattr(model_module, "acc_millors_params")(X_train_matrix, y_train, X_test_matrix, y_test)
            results[model_name].append(accuracy)
            print(f"{model_name} (max_features={mida_matriu}): Accuracy = {accuracy:.4f}")
    
    # Graficar els resultats
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in results.items():
        plt.plot(max_features_range, accuracies, label=model_name)

    # Modificar les etiquetes de l'eix X per mostrar els valors de max_features
    plt.xticks(max_features_range, [str(x) for x in max_features_range])

    plt.xlabel('max_features (TF-IDF)')
    plt.ylabel('Accuracy')
    plt.title('Rendiment dels models segons max_features (TF-IDF)')
    plt.legend()
    plt.grid(True)

    #guardar grafica
    file_path = os.path.join(EVALUATION_DIR, "acc_max_features.png")
    plt.savefig(file_path)

#robustesa diferents % train
def evaluar_robustesa_amb_subsets(DATA_PATH, model_modules, train_percentages):
    """
    Avalua tots els models amb diferents percentatges del conjunt de dades d'entrenament i genera una gràfica.
    """
    # Carregar i processar les dades
    X_train_matrix, X_valid_matrix, X_test_matrix, y_train, y_valid, y_test = carregar_i_processar_dades(DATA_PATH, EMBEDDING_METHOD, WORD_EMBEDDING_MODEL_PATH)
    
    results = {model_name: [] for model_name in model_modules.keys()}  # Per emmagatzemar els resultats

    for train_pct in train_percentages:
        print(f"Processant amb {train_pct}% del conjunt d'entrenament.")

        # Reduir el conjunt de dades d'entrenament
        if train_pct < 100:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train_matrix, y_train, train_size=train_pct / 100, random_state=42)
        else:
            X_train_subset, y_train_subset = X_train_matrix, y_train

        for model_name, model_module in model_modules.items():
            accuracy = getattr(model_module, "acc_millors_params")(X_train_subset, y_train_subset, X_test_matrix, y_test)
            results[model_name].append(accuracy)
            print(f"{model_name} ({train_pct}% train): Accuracy = {accuracy:.4f}")
    
    # Graficar els resultats
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in results.items():
        plt.plot(train_percentages, accuracies, label=model_name)

    # Modificar les etiquetes de l'eix X per mostrar percentatges
    plt.xticks(train_percentages, [f'{x}%' for x in train_percentages])

    plt.xlabel('Percentatge del conjunt d\'entrenament (%)')
    plt.ylabel('Accuracy')
    plt.title('Rendiment dels models segons percentatge del conjunt d\'entrenament')
    plt.legend()
    plt.grid(True)

    #guardar grafica
    file_path = os.path.join(EVALUATION_DIR, "acc_perc_train.png")
    plt.savefig(file_path)

def get_stemmed_tfidf(DATA_PATH):
    # Carregar el dataset Train, Valid, Test
    X_train = pd.read_csv(f'{DATA_PATH}Train.csv')
    X_valid = pd.read_csv(f'{DATA_PATH}ValidFAKE.csv')
    X_test = pd.read_csv(f'{DATA_PATH}Test.csv')

    X_train['processed_text'] = X_train['text'].apply(lambda x: stemmatize(normalize_text(x)))
    X_valid['processed_text'] = X_valid['text'].apply(lambda x: stemmatize(normalize_text(x)))
    X_test['processed_text'] = X_test['text'].apply(lambda x: stemmatize(normalize_text(x)))

    y_train = X_train['label']
    y_valid = X_valid['label']
    y_test = X_test['label']

    X_train = X_train[['processed_text']]
    X_valid = X_valid[['processed_text']]
    X_test = X_test[['processed_text']]

    X_train_matrix, X_valid_matrix, X_test_matrix = convert_to_tfidf(X_train, X_valid, X_test, 5000)
    print('dades processaddes amb stemming')
    return X_train_matrix, X_valid_matrix, X_test_matrix, y_train, y_valid, y_test

#stemming/lemmatitzacio
def compare_stemming_vs_lemmatization(DATA_PATH, model_modules):
    """
    Compara els rendiments de les tècniques de stemming i lemmatization per tots els models.
    Genera un gràfic de barres amb els resultats.
    """
    # Obtenir les matrius TF-IDF amb lemmatization (ja en TF-IDF)
    X_train_lemmatized, X_valid_lemmatized, X_test_lemmatized, y_train, y_valid, y_test = carregar_i_processar_dades(DATA_PATH, EMBEDDING_METHOD, WORD_EMBEDDING_MODEL_PATH)
    
    # Obtenir les matrius TF-IDF amb stemming
    X_train_stemmed_matrix, X_valid_stemmed_matrix, X_test_stemmed_matrix, y_train, y_valid, y_test = get_stemmed_tfidf(DATA_PATH)
    
    # Inicialitzar un diccionari per emmagatzemar els resultats dels models
    results = {model_name: {'lemmatization': 0, 'stemming': 0} for model_name in model_modules.keys()}

    # Comparar per a cada model
    for model_name, model_module in model_modules.items():
        print(f"Model: {model_name}")
        
        # Accuracy per a lemmatization (ja en TF-IDF)
        accuracy_lemmatize = getattr(model_module, "acc_millors_params")(X_train_lemmatized, y_train, X_test_lemmatized, y_test)
        results[model_name]['lemmatization'] = accuracy_lemmatize
        print(f"Accuracy (lemmatization): {accuracy_lemmatize:.4f}")

        # Accuracy per a stemming (ja en TF-IDF)
        accuracy_stemming = getattr(model_module, "acc_millors_params")(X_train_stemmed_matrix, y_train, X_test_stemmed_matrix, y_test)
        results[model_name]['stemming'] = accuracy_stemming
        print(f"Accuracy (stemming): {accuracy_stemming:.4f}")

    # Crear un gràfic de barres per comparar els resultats
    model_names = list(results.keys())
    lemmatization_accuracies = [results[model]['lemmatization'] for model in model_names]
    stemming_accuracies = [results[model]['stemming'] for model in model_names]

    # Graficar els resultats
    x = np.arange(len(model_names))  # Posicions en l'eix X
    width = 0.35  # Amplada de les barres

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, lemmatization_accuracies, width, label='Lemmatization')
    ax.bar(x + width/2, stemming_accuracies, width, label='Stemming')

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Comparativa entre Lemmatization i Stemming per a cada model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    #guardar grafica
    file_path = os.path.join(EVALUATION_DIR, "acc_stemmlemm.png")
    plt.savefig(file_path)

def main():
    print("Selecciona una opció:")
    print("1. Executar i evalura el model seleccionat")
    print("2. Evaluar robustesa dels models (AMB MILLORS PARÀMETERS)")
    print('3. Evaluació stemming/lemmatize (AMB MILLORS PARÀMETERS)')
    print('4. Evaluació tfidf/word embedding (AMB MILLORS PARÀMETERS)')
    
    try:
        opcio = int(input("Introdueix el número de l'opció (1 o 2): "))
    except ValueError:
        print("Opció no vàlida. Introdueix un número enter.")
        return
    
    #EXECUCIO MODEL SELECCIONAT
    if opcio == 1:
        print('embedding method:', EMBEDDING_METHOD)
        print('Model:', MODEL_CHOICE)

        # Carregar i processar les dades
        X_train_matrix, X_valid_matrix, X_test_matrix, y_train, y_valid, y_test = carregar_i_processar_dades(DATA_PATH, EMBEDDING_METHOD, WORD_EMBEDDING_MODEL_PATH)
        
        # Obtenir el model
        if MODEL_CHOICE not in model_modules:
            raise ValueError(f"Model no reconegut: {MODEL_CHOICE}")
        model_module = model_modules[MODEL_CHOICE]

        # Entrenar i predir
        y_pred = getattr(model_module, "entrena_prediu_i_evaluaMaxIter")(X_train_matrix, y_train, X_test_matrix, y_test)

    #GRAFIQUES ROBUSTESA
    elif opcio == 2:
        max_features_range = [100, 500, 1000, 3000, 5000]  # Valors per provar TF-IDF
        train_percentages = [10, 25, 50, 75, 100]  # Percentatges a utilitzar
        evaluar_robustesa_amb_tfidf(DATA_PATH, model_modules, max_features_range)
        evaluar_robustesa_amb_subsets(DATA_PATH, model_modules, train_percentages)

    #COMPARAR STEMMING I LEMMATIZE
    elif opcio == 3:
        compare_stemming_vs_lemmatization(DATA_PATH, model_modules)

    elif opcio == 4:
        print('hola')
    else:
        print("Opció no vàlida. Si us plau, selecciona 1 o 2.")

if __name__ == "__main__":
    main()
